// Foundation Models C-ABI shim for sidekick.
//
// Exposes a deliberately tiny surface: availability probing, session
// create/free, and a blocking respond() with optional guided generation
// from a JSON Schema subset. All strings cross the boundary as
// (pointer, length) UTF-8 buffers; every buffer returned to Rust is
// malloc'd here and freed by Rust via sk_fm_buf_free/sk_fm_string_free.
//
// NEEDS-HARDWARE-VERIFICATION: written against the macOS 26 (Tahoe)
// FoundationModels API surface; compile and exercise on a real machine
// before trusting edge cases (see docs/DECISIONS.md).

import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

// Availability codes shared with Rust (see src/ffi.rs).
private let SK_AVAILABLE: Int32 = 0
private let SK_DEVICE_NOT_ELIGIBLE: Int32 = 1
private let SK_AI_NOT_ENABLED: Int32 = 2
private let SK_MODEL_NOT_READY: Int32 = 3
private let SK_OTHER: Int32 = 4
private let SK_OS_TOO_OLD: Int32 = 5

private func mallocBuffer(_ s: String) -> (UnsafeMutablePointer<UInt8>, Int) {
    let bytes = Array(s.utf8)
    let ptr = UnsafeMutablePointer<UInt8>.allocate(capacity: max(bytes.count, 1))
    if !bytes.isEmpty {
        ptr.update(from: bytes, count: bytes.count)
    }
    return (ptr, bytes.count)
}

private func setError(_ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?, _ message: String) {
    guard let err else { return }
    err.pointee = strdup(message)
}

private func takeString(_ ptr: UnsafePointer<UInt8>?, _ len: UInt) -> String {
    guard let ptr, len > 0 else { return "" }
    return String(decoding: UnsafeBufferPointer(start: ptr, count: Int(len)), as: UTF8.self)
}

@_cdecl("sk_fm_buf_free")
public func sk_fm_buf_free(_ ptr: UnsafeMutablePointer<UInt8>?, _ len: UInt) {
    ptr?.deallocate()
}

@_cdecl("sk_fm_string_free")
public func sk_fm_string_free(_ ptr: UnsafeMutablePointer<CChar>?) {
    free(ptr)
}

@_cdecl("sk_fm_availability")
public func sk_fm_availability() -> Int32 {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        switch SystemLanguageModel.default.availability {
        case .available:
            return SK_AVAILABLE
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible:
                return SK_DEVICE_NOT_ELIGIBLE
            case .appleIntelligenceNotEnabled:
                return SK_AI_NOT_ENABLED
            case .modelNotReady:
                return SK_MODEL_NOT_READY
            @unknown default:
                return SK_OTHER
            }
        }
    }
    return SK_OS_TOO_OLD
    #else
    return SK_OS_TOO_OLD
    #endif
}

#if canImport(FoundationModels)

/// Box holding a session so it can cross the C boundary as an opaque pointer.
/// Access is serialized on the Rust side (one respond at a time per session).
@available(macOS 26.0, *)
private final class SessionBox: @unchecked Sendable {
    let session: LanguageModelSession
    init(instructions: String) {
        if instructions.isEmpty {
            self.session = LanguageModelSession()
        } else {
            self.session = LanguageModelSession(instructions: instructions)
        }
    }
}

/// Convert a JSON Schema subset into a DynamicGenerationSchema.
/// Supported: object/string/integer/number/boolean, string enums, arrays of
/// the above, nested objects, required lists (absent => optional property).
@available(macOS 26.0, *)
private func dynamicSchema(from json: [String: Any], name: String) throws -> DynamicGenerationSchema {
    let type = json["type"] as? String ?? "object"
    let description = json["description"] as? String

    if let anyOf = json["enum"] as? [String] {
        return DynamicGenerationSchema(name: name, description: description, anyOf: anyOf)
    }

    switch type {
    case "object":
        let props = json["properties"] as? [String: Any] ?? [:]
        let required = Set(json["required"] as? [String] ?? [])
        var properties: [DynamicGenerationSchema.Property] = []
        // Sort for deterministic order.
        for key in props.keys.sorted() {
            guard let sub = props[key] as? [String: Any] else { continue }
            let subSchema = try dynamicSchema(from: sub, name: key)
            properties.append(
                DynamicGenerationSchema.Property(
                    name: key,
                    description: sub["description"] as? String,
                    schema: subSchema,
                    isOptional: !required.contains(key)
                )
            )
        }
        return DynamicGenerationSchema(name: name, description: description, properties: properties)
    case "array":
        let items = json["items"] as? [String: Any] ?? ["type": "string"]
        let itemSchema = try dynamicSchema(from: items, name: name + "Item")
        let minItems = json["minItems"] as? Int
        let maxItems = json["maxItems"] as? Int
        return DynamicGenerationSchema(
            arrayOf: itemSchema,
            minimumElements: minItems,
            maximumElements: maxItems
        )
    case "string":
        return DynamicGenerationSchema(type: String.self)
    case "integer":
        return DynamicGenerationSchema(type: Int.self)
    case "number":
        return DynamicGenerationSchema(type: Double.self)
    case "boolean":
        return DynamicGenerationSchema(type: Bool.self)
    default:
        throw NSError(
            domain: "sidekick.fm", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "unsupported JSON schema type: \(type)"]
        )
    }
}

#endif

@_cdecl("sk_fm_session_create")
public func sk_fm_session_create(
    _ instructions: UnsafePointer<UInt8>?,
    _ instructionsLen: UInt,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let box = SessionBox(instructions: takeString(instructions, instructionsLen))
        return Unmanaged.passRetained(box).toOpaque()
    }
    #endif
    setError(err, "Foundation Models requires macOS 26 or later")
    return nil
}

@_cdecl("sk_fm_session_free")
public func sk_fm_session_free(_ session: UnsafeMutableRawPointer?) {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard let session else { return }
        Unmanaged<SessionBox>.fromOpaque(session).release()
    }
    #endif
}

@_cdecl("sk_fm_respond")
public func sk_fm_respond(
    _ session: UnsafeMutableRawPointer?,
    _ prompt: UnsafePointer<UInt8>?,
    _ promptLen: UInt,
    _ schemaJson: UnsafePointer<UInt8>?,
    _ schemaLen: UInt,
    _ temperature: Double,
    _ maxTokens: Int64,
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outLen: UnsafeMutablePointer<UInt>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        guard let session, let out, let outLen else {
            setError(err, "null argument")
            return 1
        }
        let box = Unmanaged<SessionBox>.fromOpaque(session).takeUnretainedValue()
        let promptText = takeString(prompt, promptLen)
        let schemaText = takeString(schemaJson, schemaLen)

        var options = GenerationOptions()
        if temperature >= 0 {
            options = GenerationOptions(temperature: temperature)
        }
        if maxTokens > 0 {
            options.maximumResponseTokens = Int(maxTokens)
        }

        let semaphore = DispatchSemaphore(value: 0)
        var resultText: String?
        var resultError: String?

        Task {
            defer { semaphore.signal() }
            do {
                if schemaText.isEmpty {
                    let response = try await box.session.respond(to: promptText, options: options)
                    resultText = response.content
                } else {
                    guard
                        let data = schemaText.data(using: .utf8),
                        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    else {
                        resultError = "schema is not a JSON object"
                        return
                    }
                    let root = try dynamicSchema(from: parsed, name: "Response")
                    let schema = try GenerationSchema(root: root, dependencies: [])
                    let response = try await box.session.respond(
                        to: promptText, schema: schema, options: options
                    )
                    resultText = response.content.jsonString
                }
            } catch {
                resultError = String(describing: error)
            }
        }
        semaphore.wait()

        if let resultText {
            let (ptr, len) = mallocBuffer(resultText)
            out.pointee = ptr
            outLen.pointee = UInt(len)
            return 0
        }
        setError(err, resultError ?? "unknown Foundation Models error")
        return 1
    }
    #endif
    setError(err, "Foundation Models requires macOS 26 or later")
    return 1
}
