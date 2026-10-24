// statecheck <model.mlmodelc> <cpu_only|cpu_and_ne|cpu_and_gpu> <steps> <C> <D> <Dh> [prefill_fn]
// One MLState made by the decode function, used by a prefill function once and
// then by decode for <steps> steps: does cross-function state sharing hold up?
// Prints per-step latency summary and a checksum of the last logits.
import CoreML
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
let a = CommandLine.arguments
let url = URL(fileURLWithPath: a[1])
let steps = Int(a[3])!, C = Int(a[4])!, D = Int(a[5])!, Dh = Int(a[6])!
let prefillFn = a.count > 7 ? a[7] : "prefill_128"
let S = Int(prefillFn.split(separator: "_").last!)!

func cfg(_ fn: String) -> MLModelConfiguration {
    let c = MLModelConfiguration()
    c.functionName = fn
    switch a[2] {
    case "cpu_only": c.computeUnits = .cpuOnly
    case "cpu_and_gpu": c.computeUnits = .cpuAndGPU
    default: c.computeUnits = .cpuAndNeuralEngine
    }
    return c
}

func arr(_ shape: [Int], _ fill: (Int) -> Float) -> MLMultiArray {
    let m = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
    let p = m.dataPointer.bindMemory(to: Float16.self, capacity: m.count)
    for i in 0..<m.count { p[i] = Float16(fill(i)) }
    return m
}

func feed(_ S: Int, _ pos: Int, last: Bool) -> MLDictionaryFeatureProvider {
    var d: [String: Any] = [:]
    d["hidden"] = arr([1, D, 1, S]) { i in Float((i * 7919) % 200) / 100.0 - 1.0 }
    let p = try! MLMultiArray(shape: [1], dataType: .int32); p[0] = NSNumber(value: pos); d["pos"] = p
    d["mask"] = arr([1, 1, S, C]) { i in (i % C) <= pos + i / C ? 0 : -30000 }
    d["cos"] = arr([S, Dh]) { _ in 1 }
    d["sin"] = arr([S, Dh]) { _ in 0 }
    if S > 1 && last { d["sel"] = arr([S]) { i in i == S - 1 ? 1 : 0 } }
    return try! MLDictionaryFeatureProvider(dictionary: d)
}

let t0 = Date()
let dec = try! MLModel(contentsOf: url, configuration: cfg("decode"))
let pre = try! MLModel(contentsOf: url, configuration: cfg(prefillFn))
print("loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s")
let last = dec.modelDescription.outputDescriptionsByName["logits"] != nil
let seqs = Int(ProcessInfo.processInfo.environment["SEQS"] ?? "1")!
let stateFrom = ProcessInfo.processInfo.environment["STATE_FROM"] ?? "decode"
var ts: [Double] = []
var out: MLFeatureProvider? = nil
let env = ProcessInfo.processInfo.environment
var keptStates: [MLState] = [], keptOut: [MLFeatureProvider] = [], keptIn: [MLFeatureProvider] = []
for q in 0..<seqs {
    let state = stateFrom == "prefill" ? pre.makeState() : dec.makeState()
    if env["KEEP_STATE"] == "1" { keptStates.append(state) }
    if env["PREFILL"] != "0" {
        let f = feed(S, 0, last: last)
        let o = try! pre.prediction(from: f, using: state)
        if env["KEEP_OUT"] == "1" { keptOut.append(o) }
        if env["KEEP_IN"] == "1" { keptIn.append(f) }
    }
    print("sequence \(q) prefilled")
    for i in 0..<steps {
        let f = feed(1, S + i, last: last)
        let t = Date()
        out = try! dec.prediction(from: f, using: state)
        ts.append(Date().timeIntervalSince(t) * 1000)
        if env["KEEP_OUT"] == "1" { keptOut.append(out!) }
        if env["KEEP_IN"] == "1" { keptIn.append(f) }
    }
    print("sequence \(q) decoded \(steps)")
    withExtendedLifetime(state) {}
}
print("kept \(keptStates.count) \(keptOut.count) \(keptIn.count)")
ts.sort()
let name = last ? "logits" : "hidden_out"
let o = out!.featureValue(for: name)!.multiArrayValue!
var sum: Double = 0
for i in 0..<min(o.count, 1000) { sum += o[i].doubleValue }
print(String(format: "steps %d median %.3f ms p90 %.3f ms checksum %.4f", steps, ts[ts.count / 2], ts[ts.count * 9 / 10], sum))
