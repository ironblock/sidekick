// coreml_run: predict with a compiled Core ML model on chosen compute units.
//
// For running a repro's programs on a Mac without Python or coremltools (an
// Intel Mac, say): tools/repro_cpu_matmul_tail.py --emit writes the programs
// and feeds, this runs them, and --compare reads the outputs. Compile it
// once with `swiftc -O -o coreml_run coreml_run.swift`, or run it with
// `swift coreml_run.swift ...`.
//
//   swift coreml_run.swift <model.mlmodelc> <cpu|gpu|ane|all> <feeds.json> <out.json> [repeats]
//
// feeds.json: [{"name": {"shape": [1, 128], "dtype": "int32"|"float32", "data": [...]}, ...}, ...]
// One prediction per element (a case), each repeated `repeats` times; out.json holds every
// output of every case as float64 lists, plus whether the repeats were bit-identical.
import CoreML
import Foundation

let args = CommandLine.arguments
guard args.count >= 5 else { fatalError("usage: coreml_run <model.mlmodelc> <units> <feeds.json> <out.json> [repeats]") }
let config = MLModelConfiguration()
config.computeUnits = ["cpu": .cpuOnly, "gpu": .cpuAndGPU, "ane": .cpuAndNeuralEngine, "all": .all][args[2]]!
let repeats = args.count > 5 ? Int(args[5])! : 1
let t0 = Date()
let model = try MLModel(contentsOf: URL(fileURLWithPath: args[1]), configuration: config)
let loadMs = Date().timeIntervalSince(t0) * 1000
let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[3]))) as! [[String: [String: Any]]]

func array(_ spec: [String: Any]) throws -> MLMultiArray {
    let shape = (spec["shape"] as! [Int]).map { NSNumber(value: $0) }
    let isInt = (spec["dtype"] as? String) == "int32"
    let a = try MLMultiArray(shape: shape, dataType: isInt ? .int32 : .float32)
    let data = spec["data"] as! [Double]
    if isInt {
        let p = a.dataPointer.bindMemory(to: Int32.self, capacity: data.count)
        for (i, v) in data.enumerated() { p[i] = Int32(v) }
    } else {
        let p = a.dataPointer.bindMemory(to: Float.self, capacity: data.count)
        for (i, v) in data.enumerated() { p[i] = Float(v) }
    }
    return a
}

// IEEE half to Double from its bits (Swift's Float16 isn't available on x86-64).
func half(_ h: UInt16) -> Double {
    let sign = (h & 0x8000) != 0 ? -1.0 : 1.0
    let exp = Int((h >> 10) & 0x1f), frac = Double(h & 0x3ff)
    if exp == 0 { return sign * frac * pow(2.0, -24.0) }
    if exp == 31 { return frac == 0 ? sign * Double.infinity : Double.nan }
    return sign * (1.0 + frac / 1024.0) * pow(2.0, Double(exp - 15))
}

func values(_ a: MLMultiArray) -> [Double] {
    // Read through strides, so non-contiguous outputs come out in logical order.
    let n = a.count, shape = a.shape.map { $0.intValue }, strides = a.strides.map { $0.intValue }
    var out = [Double](repeating: 0, count: n)
    var idx = [Int](repeating: 0, count: shape.count)
    for flat in 0..<n {
        var rem = flat
        for d in stride(from: shape.count - 1, through: 0, by: -1) { idx[d] = rem % shape[d]; rem /= shape[d] }
        var off = 0
        for d in 0..<shape.count { off += idx[d] * strides[d] }
        switch a.dataType {
        case .float16: out[flat] = half(a.dataPointer.bindMemory(to: UInt16.self, capacity: off + 1)[off])
        case .float32: out[flat] = Double(a.dataPointer.bindMemory(to: Float.self, capacity: off + 1)[off])
        case .double: out[flat] = a.dataPointer.bindMemory(to: Double.self, capacity: off + 1)[off]
        case .int32: out[flat] = Double(a.dataPointer.bindMemory(to: Int32.self, capacity: off + 1)[off])
        default: fatalError("unsupported output type \(a.dataType.rawValue)")
        }
    }
    return out
}

var results: [[String: Any]] = []
var allBitwise = true
var ms: [Double] = []
for c in cases {
    var dict: [String: MLFeatureValue] = [:]
    for (name, spec) in c { dict[name] = MLFeatureValue(multiArray: try array(spec)) }
    let provider = try MLDictionaryFeatureProvider(dictionary: dict)
    var first: [String: [Double]] = [:]
    for r in 0..<repeats {
        let t = Date()
        let out = try model.prediction(from: provider)
        ms.append(Date().timeIntervalSince(t) * 1000)
        var got: [String: [Double]] = [:]
        for name in out.featureNames { if let a = out.featureValue(for: name)?.multiArrayValue { got[name] = values(a) } }
        if r == 0 { first = got } else if got != first { allBitwise = false }
    }
    results.append(first.mapValues { $0 as Any })
}
let report: [String: Any] = ["load_ms": loadMs, "repeat_bitwise": allBitwise, "median_ms": ms.sorted()[ms.count / 2],
                             "outputs": results]
try JSONSerialization.data(withJSONObject: report).write(to: URL(fileURLWithPath: args[4]))
print("ok: \(cases.count) cases, repeats bitwise: \(allBitwise), median \(String(format: "%.2f", ms.sorted()[ms.count / 2])) ms")
