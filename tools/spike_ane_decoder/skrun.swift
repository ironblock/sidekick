// skrun <model_dir> <cpu_only|cpu_and_ne|cpu_and_gpu> <job.json> <out_prefix>
//
// The spike's native runtime: loads every chunk's functions (prefill_<S>,
// decode) of a converted decoder (convert.py), one MLState per chunk shared by
// all of a chunk's functions, and runs the job's tasks:
//   teacher  prefill the first ids with one prefill call, then decode the rest
//            teacher-forced; writes every returned logits row (fp32) to
//            <out_prefix>_<name>.f32
//   greedy   prefill a prompt, then greedy-decode n tokens; returns the tokens
//   bench    prefill ms per bucket; decode ms per token at several positions
//   gen      a served-style loop: prompt prefill, then n greedy tokens, wall clock
// Results go to <out_prefix>.json. Inputs are MLMultiArrays that own their
// memory (no aliasing of caller buffers).
import CoreML
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
let argv = CommandLine.arguments
let mdir = URL(fileURLWithPath: argv[1])
let unitsName = argv[2]
let job = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: argv[3]))) as! [String: Any]
let outPrefix = argv[4]

let meta = try! JSONSerialization.jsonObject(with: Data(contentsOf: mdir.appendingPathComponent("meta.json"))) as! [String: Any]
let C = meta["C"] as! Int, D = meta["hidden"] as! Int, Dh = meta["head_dim"] as! Int
let theta = (meta["rope_theta"] as! NSNumber).doubleValue
let funcs = (ProcessInfo.processInfo.environment["FUNCS"].map { $0.split(separator: ",").map(String.init) }) ?? (meta["funcs"] as! [String])
let nChunks = (meta["chunks"] as! [Any]).count
let buckets = funcs.filter { $0.hasPrefix("prefill_") }.map { Int($0.split(separator: "_")[1])! }.sorted()

// embed.npy: fp16 [V, D], C order
let npy = try! Data(contentsOf: mdir.appendingPathComponent("embed.npy"))
let hlen = Int(npy[8]) | (Int(npy[9]) << 8)
let embed: [Float16] = npy.subdata(in: (10 + hlen)..<npy.count).withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }

// RoPE tables [C, Dh], fp64 -> fp16
var cosT = [Float16](repeating: 0, count: C * Dh), sinT = cosT
for p in 0..<C {
    for i in 0..<(Dh / 2) {
        let inv = 1.0 / pow(theta, Double(2 * i) / Double(Dh))
        let a = Double(p) * inv
        cosT[p * Dh + i] = Float16(cos(a)); cosT[p * Dh + i + Dh / 2] = Float16(cos(a))
        sinT[p * Dh + i] = Float16(sin(a)); sinT[p * Dh + i + Dh / 2] = Float16(sin(a))
    }
}

func config(_ fn: String) -> MLModelConfiguration {
    let c = MLModelConfiguration()
    c.functionName = fn
    switch unitsName {
    case "cpu_only": c.computeUnits = .cpuOnly
    case "cpu_and_gpu": c.computeUnits = .cpuAndGPU
    case "all": c.computeUnits = .all
    default: c.computeUnits = .cpuAndNeuralEngine
    }
    return c
}

// Optional (PLAN=1): Core ML's compute plan per chunk and function, read in this
// process before loading, counted as tools/sidekick_convert/plan.py counts.
func planSummary(_ ci: Int, _ fn: String) -> [String: Any] {
    var ane = 0, gpu = 0, cpu = 0, none = 0
    var off: [String: Int] = [:]
    func walk(_ plan: MLComputePlan, _ block: MLModelStructure.Program.Block) {
        for op in block.operations {
            let name = String(op.operatorName.split(separator: ".").last ?? "")
            if let u = plan.deviceUsage(for: op) {
                switch u.preferred {
                case .neuralEngine: ane += 1
                case .gpu: gpu += 1; off[name, default: 0] += 1
                default: cpu += 1; off[name, default: 0] += 1
                }
            } else { none += 1 }
            for b in op.blocks { walk(plan, b) }
        }
    }
    let sem = DispatchSemaphore(value: 0)
    var err = ""
    Task {
        do {
            let plan = try await MLComputePlan.load(contentsOf: mdir.appendingPathComponent("chunk_\(ci).mlmodelc"), configuration: config(fn))
            if case .program(let p) = plan.modelStructure, let f = p.functions[fn] { walk(plan, f.block) }
        } catch { err = "\(error)" }
        sem.signal()
    }
    sem.wait()
    let assigned = ane + gpu + cpu
    return ["ane": ane, "gpu": gpu, "cpu": cpu, "unassigned": none, "share": assigned > 0 ? Double(ane) / Double(assigned) : 0,
            "off": off, "error": err]
}
var plans: [String: Any] = [:]
if ProcessInfo.processInfo.environment["PLAN"] == "1" {
    for ci in 0..<nChunks {
        let pf = (ProcessInfo.processInfo.environment["PLAN_FUNCS"] ?? funcs.joined(separator: ",")).split(separator: ",").map(String.init)
        for fn in funcs where pf.contains(fn) {
            let t = Date()
            var s = planSummary(ci, fn)
            s["seconds"] = Date().timeIntervalSince(t)
            plans["chunk\(ci).\(fn)"] = s
            print("plan chunk \(ci) \(fn): \(s)")
        }
    }
}

var loadS: [String: Double] = [:]
var models: [[String: MLModel]] = []
let tLoad = Date()
for ci in 0..<nChunks {
    var fm: [String: MLModel] = [:]
    for fn in funcs {
        let t = Date()
        fm[fn] = try! MLModel(contentsOf: mdir.appendingPathComponent("chunk_\(ci).mlmodelc"), configuration: config(fn))
        loadS["chunk\(ci).\(fn)"] = Date().timeIntervalSince(t)
        print("loaded chunk \(ci) \(fn) in \(String(format: "%.1f", loadS["chunk\(ci).\(fn)"]!)) s")
    }
    models.append(fm)
}
let loadTotal = Date().timeIntervalSince(tLoad)

func newStates() -> [MLState] { models.map { $0[funcs[0]]!.makeState() } }

func f16(_ shape: [Int]) -> MLMultiArray {
    try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
}

var chunkTimes: [String: [Double]] = [:]
var traceOut: [[Float]]? = nil      // "trace" task: every chunk's output of the last call, fp32

// Per-prediction timeout (PREDICT_TIMEOUT s, default 60): a watcher thread
// exits the process (code 4) at the first prediction that runs past it, so a
// hung ANE call stops the run instead of being retried.
let predictTimeout = Double(ProcessInfo.processInfo.environment["PREDICT_TIMEOUT"] ?? "60")!
let callLock = NSLock()
var callStarted: Date? = nil
var callWhat = ""
Thread.detachNewThread {
    while true {
        Thread.sleep(forTimeInterval: 1)
        callLock.lock(); let s = callStarted; let w = callWhat; callLock.unlock()
        if let s = s, Date().timeIntervalSince(s) > predictTimeout {
            print("HANG: \(w) ran past \(predictTimeout) s"); fflush(stdout); exit(4)
        }
    }
}

/// One call of `fn` (S tokens) through every chunk; returns the logits (fp32).
func call(_ fn: String, _ S: Int, _ ids: [Int], _ pos: Int, _ st: [MLState], sel: Int = 0) -> [Float] {
    precondition(pos + S <= C, "past the cache")
    let hidden = f16([1, D, 1, S])
    let hp = hidden.dataPointer.bindMemory(to: Float16.self, capacity: D * S)
    for s in 0..<S {
        let tok = s < ids.count ? ids[s] : 0
        for d in 0..<D { hp[d * S + s] = embed[tok * D + d] }
    }
    let posA = try! MLMultiArray(shape: [1], dataType: .int32); posA[0] = NSNumber(value: pos)
    let mask = f16([1, 1, S, C])
    let mp = mask.dataPointer.bindMemory(to: Float16.self, capacity: S * C)
    for q in 0..<S { for k in 0..<C { mp[q * C + k] = k <= pos + q ? 0 : -30000 } }
    let cosA = f16([S, Dh]), sinA = f16([S, Dh])
    let cp = cosA.dataPointer.bindMemory(to: Float16.self, capacity: S * Dh)
    let sp = sinA.dataPointer.bindMemory(to: Float16.self, capacity: S * Dh)
    for i in 0..<(S * Dh) { cp[i] = cosT[pos * Dh + i]; sp[i] = sinT[pos * Dh + i] }
    var h: MLMultiArray = hidden
    var logits: MLMultiArray? = nil
    for ci in 0..<nChunks {
        var d: [String: Any] = ["hidden": h, "pos": posA, "mask": mask, "cos": cosA, "sin": sinA]
        let last = ci == nChunks - 1
        if last && S > 1 && models[ci][fn]!.modelDescription.inputDescriptionsByName["sel"] != nil {
            let s = f16([S]); let p = s.dataPointer.bindMemory(to: Float16.self, capacity: S)
            for i in 0..<S { p[i] = i == sel ? 1 : 0 }
            d["sel"] = s
        }
        let t = Date()
        callLock.lock(); callStarted = t; callWhat = "\(fn) chunk \(ci) pos \(pos)"; callLock.unlock()
        let o = try! models[ci][fn]!.prediction(from: try! MLDictionaryFeatureProvider(dictionary: d), using: st[ci])
        callLock.lock(); callStarted = nil; callLock.unlock()
        chunkTimes["\(fn).chunk\(ci)", default: []].append(Date().timeIntervalSince(t) * 1000)
        if last, let lv = o.featureValue(for: "logits") { logits = lv.multiArrayValue! } else {
            h = o.featureValue(for: "hidden_out")!.multiArrayValue!
            if last { logits = h }
        }
        if traceOut != nil {
            let a = last ? logits! : h
            let n = a.count
            let p = a.dataPointer.bindMemory(to: Float16.self, capacity: n)
            traceOut![ci] = (0..<n).map { Float(p[$0]) }
        }
    }
    let L = logits!
    let n = L.shape.last!.intValue
    let stride = L.strides.last!.intValue
    var out = [Float](repeating: 0, count: n)
    let lp = L.dataPointer.bindMemory(to: Float16.self, capacity: n * stride)
    for i in 0..<n { out[i] = Float(lp[i * stride]) }
    return out
}

// MULTI=T: one T-token function (prefill_T) serves prompt and generation.
// A decode step pads the call to T tokens and picks the real token's logits
// with sel; the padding writes junk keys at pos+1..pos+T-1, which no query
// attends to before the next real token overwrites them.
let multiT = Int(ProcessInfo.processInfo.environment["MULTI"] ?? "0")!
func decodeCall(_ tok: Int, _ pos: Int, _ st: [MLState]) -> [Float] {
    if multiT > 0 { return call("prefill_\(multiT)", multiT, [tok], pos, st, sel: 0) }
    return call("decode", 1, [tok], pos, st)
}

func argmax(_ x: [Float]) -> Int { var b = 0; for i in 1..<x.count where x[i] > x[b] { b = i }; return b }

func prefill(_ ids: [Int], _ st: [MLState], bucket: Int? = nil) -> [Float] {
    if buckets.isEmpty {
        // a decode-only build (FUNCS=decode or no prefill function): one token per call
        var lg: [Float] = []
        for (i, t) in ids.enumerated() { lg = decodeCall(t, i, st) }
        return lg
    }
    let b = bucket.flatMap { buckets.contains($0) ? $0 : nil } ?? (buckets.first { $0 >= ids.count } ?? buckets.last!)
    var pos = 0
    var lg: [Float] = []
    while pos < ids.count {
        let piece = Array(ids[pos..<min(pos + b, ids.count)])
        lg = call("prefill_\(b)", b, piece, pos, st, sel: piece.count - 1)
        pos += piece.count
    }
    return lg
}

func median(_ x: [Double]) -> Double { let s = x.sorted(); return s[s.count / 2] }
func pct(_ x: [Double], _ p: Double) -> Double { let s = x.sorted(); return s[min(s.count - 1, Int(Double(s.count) * p))] }

var results: [String: Any] = ["units": unitsName, "load_s": loadS, "load_total_s": loadTotal, "plans": plans]
var outTasks: [[String: Any]] = []
for task in job["tasks"] as! [[String: Any]] {
    let type = task["type"] as! String
    var r: [String: Any] = ["type": type, "name": task["name"] ?? type]
    let t0 = Date()
    switch type {
    case "teacher":
        let ids = task["ids"] as! [Int], p = task["p"] as! Int
        let st = newStates()
        var rows: [[Float]] = [prefill(Array(ids[0..<p]), st, bucket: task["bucket"] as? Int)]
        for i in p..<ids.count { rows.append(call("decode", 1, [ids[i]], i, st)) }
        var flat = [Float](); for r in rows { flat += r }
        let url = URL(fileURLWithPath: "\(outPrefix)_\(task["name"] as! String).f32")
        try! flat.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: url)
        r["rows"] = rows.count; r["vocab"] = rows[0].count
    case "greedy":
        let ids = task["ids"] as! [Int], n = task["n"] as! Int
        let st = newStates()
        var toks = [argmax(prefill(ids, st))]
        var pos = ids.count
        while toks.count < n { toks.append(argmax(decodeCall(toks.last!, pos, st))); pos += 1 }
        r["tokens"] = toks
    case "bench":
        let st = newStates()
        var pre: [String: Any] = [:]
        for b in (task["buckets"] as? [Int]) ?? buckets {
            let ids = (0..<b).map { 100 + ($0 * 37) % 900 }
            _ = call("prefill_\(b)", b, ids, 0, st, sel: b - 1)
            var ts: [Double] = []
            for _ in 0..<((task["reps"] as? Int) ?? 5) {
                let t = Date(); _ = call("prefill_\(b)", b, ids, 0, st, sel: b - 1); ts.append(Date().timeIntervalSince(t) * 1000)
            }
            pre["\(b)"] = ["ms": median(ts), "tok_s": Double(b) / median(ts) * 1000]
        }
        var dec: [String: Any] = [:]
        let steps = (task["steps"] as? Int) ?? 32
        for p in (task["positions"] as! [Int]) where p + steps + max(multiT, 1) <= C {
            var ts: [Double] = []
            for i in 0..<steps {
                let t = Date(); _ = decodeCall(100 + i, p + i, st); ts.append(Date().timeIntervalSince(t) * 1000)
            }
            let w = Array(ts.dropFirst(2))
            dec["\(p)"] = ["ms": median(w), "p90_ms": pct(w, 0.9), "tok_s": 1000 / median(w)]
        }
        r["prefill"] = pre; r["decode"] = dec
    case "loop":
        // the companion's load: back-to-back generations (prompt prefill + n greedy
        // tokens) for `seconds`, or until the file `stop_file` exists
        let ids = task["ids"] as! [Int], n = task["n"] as! Int, secs = (task["seconds"] as? Double) ?? 60
        let stopFile = task["stop_file"] as? String
        let st = newStates()
        var runs: [[String: Double]] = []
        let tStart = Date()
        while Date().timeIntervalSince(tStart) < secs {
            if let f = stopFile, FileManager.default.fileExists(atPath: f) { break }
            let t1 = Date()
            var tok = argmax(prefill(ids, st))
            let t2 = Date()
            var pos = ids.count
            for _ in 0..<n { tok = argmax(decodeCall(tok, pos, st)); pos += 1 }
            let t3 = Date()
            runs.append(["start": t1.timeIntervalSince1970, "end": t3.timeIntervalSince1970,
                         "ttft_ms": t2.timeIntervalSince(t1) * 1000, "decode_tok_s": Double(n) / t3.timeIntervalSince(t2)])
        }
        r["runs"] = runs
    case "fresh":
        // first prefill (and first decode) on a fresh MLState vs on a reused one
        let b = (task["bucket"] as? Int).flatMap { buckets.contains($0) ? $0 : nil } ?? buckets[0]
        let ids = (0..<b).map { 100 + ($0 * 37) % 900 }
        var fresh: [Double] = [], freshDec: [Double] = [], reused: [Double] = []
        let keep = newStates()
        for _ in 0..<((task["reps"] as? Int) ?? 4) {
            let st = newStates()
            var t = Date(); _ = call("prefill_\(b)", b, ids, 0, st, sel: b - 1); fresh.append(Date().timeIntervalSince(t) * 1000)
            t = Date(); _ = decodeCall(5, b, st); freshDec.append(Date().timeIntervalSince(t) * 1000)
            t = Date(); _ = call("prefill_\(b)", b, ids, 0, keep, sel: b - 1); reused.append(Date().timeIntervalSince(t) * 1000)
        }
        r["fresh_prefill_ms"] = fresh; r["fresh_first_decode_ms"] = freshDec; r["reused_prefill_ms"] = reused
    case "tbench":
        // per call of each loaded prefill_T function at a few positions: ms, and
        // tokens/s if all T tokens are real (prompt) or calls/s if one is (decode)
        let st = newStates()
        var res: [String: Any] = [:]
        for fn in funcs where fn.hasPrefix("prefill_") {
            let T = Int(fn.split(separator: "_")[1])!
            let ids = (0..<T).map { 100 + ($0 * 37) % 900 }
            var per: [String: Any] = [:]
            for p in (task["positions"] as? [Int]) ?? [128, 1024] where p + T <= C {
                _ = call(fn, T, ids, p, st, sel: T - 1)
                var ts: [Double] = []
                for _ in 0..<((task["reps"] as? Int) ?? 20) {
                    let t = Date(); _ = call(fn, T, ids, p, st, sel: T - 1); ts.append(Date().timeIntervalSince(t) * 1000)
                }
                per["\(p)"] = ["ms": median(ts), "p90_ms": pct(ts, 0.9), "prompt_tok_s": Double(T) / median(ts) * 1000,
                               "decode_tok_s": 1000 / median(ts)]
            }
            res[fn] = per
        }
        r["tbench"] = res
    case "switch":
        // the prefill <-> decode function-switch stall: reps of (one prefill, then
        // `steps` decodes) on one state; per call, the time of every prefill, of
        // the first decode after it, and of the steady decodes, with per-chunk
        // times for the switched calls
        let b = (task["bucket"] as? Int).flatMap { buckets.contains($0) ? $0 : nil } ?? buckets[0]
        let steps = (task["steps"] as? Int) ?? 8
        let ids = (0..<b).map { 100 + ($0 * 37) % 900 }
        let st = newStates()
        var pre: [Double] = [], first: [Double] = [], steady: [Double] = []
        var firstChunks: [[Double]] = [], preChunks: [[Double]] = []
        for _ in 0..<((task["reps"] as? Int) ?? 6) {
            var n0 = chunkTimes["prefill_\(b).chunk0"]?.count ?? 0
            var t = Date(); _ = call("prefill_\(b)", b, ids, 0, st, sel: b - 1); pre.append(Date().timeIntervalSince(t) * 1000)
            preChunks.append((0..<nChunks).map { chunkTimes["prefill_\(b).chunk\($0)"]![n0] })
            n0 = chunkTimes["decode.chunk0"]?.count ?? 0
            t = Date(); _ = decodeCall(5, b, st); first.append(Date().timeIntervalSince(t) * 1000)
            firstChunks.append((0..<nChunks).map { chunkTimes["decode.chunk\($0)"]![n0] })
            for i in 1..<steps {
                t = Date(); _ = decodeCall(5 + i, b + i, st); steady.append(Date().timeIntervalSince(t) * 1000)
            }
        }
        r["prefill_ms"] = pre; r["first_decode_ms"] = first; r["steady_decode_median_ms"] = median(steady)
        r["first_decode_chunk_ms"] = firstChunks; r["prefill_chunk_ms"] = preChunks
    case "trace":
        // decode the ids one token per call; dump every chunk's output per position
        let ids = task["ids"] as! [Int]
        let st = newStates()
        var dumps = [[Float]](repeating: [], count: nChunks)
        for (i, t) in ids.enumerated() {
            traceOut = [[Float]](repeating: [], count: nChunks)
            _ = decodeCall(t, i, st)
            for ci in 0..<nChunks { dumps[ci] += traceOut![ci] }
        }
        traceOut = nil
        for ci in 0..<nChunks {
            let url = URL(fileURLWithPath: "\(outPrefix)_trace\(ci).f32")
            try! dumps[ci].withUnsafeBufferPointer { Data(buffer: $0) }.write(to: url)
        }
        r["rows"] = ids.count
    case "gen":
        let ids = task["ids"] as! [Int], n = task["n"] as! Int
        let st = newStates()
        let t1 = Date()
        var tok = argmax(prefill(ids, st))
        let t2 = Date()
        var pos = ids.count
        var toks = [tok]
        for _ in 0..<n { tok = argmax(decodeCall(tok, pos, st)); pos += 1; toks.append(tok) }
        let t3 = Date()
        r["prompt_tokens"] = ids.count; r["ttft_ms"] = t2.timeIntervalSince(t1) * 1000
        r["decode_tokens"] = n; r["decode_tok_s"] = Double(n) / t3.timeIntervalSince(t2)
        r["start_unix"] = t1.timeIntervalSince1970; r["end_unix"] = t3.timeIntervalSince1970
    default:
        fatalError("unknown task \(type)")
    }
    r["wall_s"] = Date().timeIntervalSince(t0)
    outTasks.append(r)
    print("task \(r["name"]!) done in \(String(format: "%.1f", r["wall_s"] as! Double)) s")
}
results["tasks"] = outTasks
results["chunk_median_ms"] = chunkTimes.mapValues { median($0) }
if ProcessInfo.processInfo.environment["DUMP"] == "1" { results["chunk_times_ms"] = chunkTimes }
let data = try! JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
try! data.write(to: URL(fileURLWithPath: "\(outPrefix).json"))
print("done")
