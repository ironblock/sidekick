// Foundation Models C-ABI shim for sidekick.
//
// Exposes a deliberately tiny surface: availability probing, model info,
// session create/free, and a blocking respond() with optional guided
// generation from a JSON Schema subset. All strings cross the boundary as
// (pointer, length) UTF-8 buffers; every buffer returned to Rust is
// malloc'd here and freed by Rust via sk_fm_buf_free/sk_fm_string_free.
//
// respond() returns a JSON envelope rather than bare text so it can carry
// token usage, a truncation flag, and a *typed* error classification (see
// `classify`) without a C header for Swift and Rust to agree on. The Rust
// side parses it in src/envelope.rs.
//
// SDK gating: the runtime floor is macOS 26.0. APIs that only exist in the
// macOS 27 SDK (per-response usage, model variant and capabilities, typed
// LanguageModelError) are compiled only when build.rs defines SK_SDK_27, and
// run only under `#available(macOS 27.0, *)`. build.rs requires the 26.4 SDK
// or newer, so 26.4 APIs (tokenCount, contextSize) compile unconditionally.

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

/// Serialize a JSON-compatible dictionary and hand it to Rust as a
/// malloc'd buffer. Returns false if serialization failed.
private func writeJSON(
    _ object: [String: Any],
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>,
    _ outLen: UnsafeMutablePointer<UInt>
) -> Bool {
    guard
        JSONSerialization.isValidJSONObject(object),
        let data = try? JSONSerialization.data(withJSONObject: object),
        let text = String(data: data, encoding: .utf8)
    else { return false }
    let (ptr, len) = mallocBuffer(text)
    out.pointee = ptr
    outLen.pointee = UInt(len)
    return true
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

// MARK: - Error classification

/// Map a Foundation Models error to a stable kind the Rust side maps to an
/// HTTP status. Typed matching first (macOS 27 `LanguageModelError` and
/// friends, then macOS 26 `GenerationError`); the only string matching left
/// is a fallback for context overflow, for binaries built without the 27
/// SDK running on macOS 27, where the typed error is invisible.
///
/// Kinds: context_overflow, content_filter, rate_limited, transient,
/// model_not_ready, unsupported_guide, unsupported_language, other.
@available(macOS 26.0, *)
private func classify(_ error: Error) -> [String: Any] {
    var out: [String: Any] = ["kind": "other", "message": String(describing: error)]

    #if SK_SDK_27
    if #available(macOS 27.0, *) {
        if let e = error as? LanguageModelError {
            switch e {
            case .contextSizeExceeded(let info):
                out["kind"] = "context_overflow"
                out["token_count"] = info.tokenCount
                out["context_size"] = info.contextSize
            case .rateLimited(let info):
                out["kind"] = "rate_limited"
                if let reset = info.resetDate {
                    out["retry_after_secs"] = max(0, Int(reset.timeIntervalSinceNow.rounded(.up)))
                }
            case .guardrailViolation, .refusal:
                out["kind"] = "content_filter"
            case .timeout:
                out["kind"] = "transient"
            case .unsupportedGenerationGuide:
                out["kind"] = "unsupported_guide"
            case .unsupportedLanguageOrLocale:
                out["kind"] = "unsupported_language"
            case .unsupportedCapability, .unsupportedTranscriptContent:
                break
            @unknown default:
                break
            }
            return out
        }
        if let e = error as? LanguageModelSession.Error {
            switch e {
            case .concurrentRequests, .transcriptMutationWhileResponding:
                out["kind"] = "transient"
            @unknown default:
                break
            }
            return out
        }
        if let e = error as? SystemLanguageModel.Error {
            switch e {
            case .assetsUnavailable:
                out["kind"] = "model_not_ready"
            @unknown default:
                break
            }
            return out
        }
    }
    #endif

    if let e = error as? LanguageModelSession.GenerationError {
        switch e {
        case .exceededContextWindowSize:
            out["kind"] = "context_overflow"
        case .assetsUnavailable:
            out["kind"] = "model_not_ready"
        case .guardrailViolation, .refusal:
            out["kind"] = "content_filter"
        case .unsupportedGuide:
            out["kind"] = "unsupported_guide"
        case .unsupportedLanguageOrLocale:
            out["kind"] = "unsupported_language"
        case .rateLimited:
            out["kind"] = "rate_limited"
        case .concurrentRequests:
            out["kind"] = "transient"
        case .decodingFailure:
            break
        @unknown default:
            break
        }
        return out
    }

    // Untyped fallback: macOS 27's context-overflow message, e.g. "Content
    // contains 4459 tokens, which exceeds the maximum allowed context size
    // of 4096."
    let text = "\(String(describing: error)) \(error.localizedDescription)"
    if text.contains("exceeds the maximum allowed context size") {
        out["kind"] = "context_overflow"
        if let regex = try? NSRegularExpression(
            pattern: #"contains (\d+) tokens.*context size of (\d+)"#
        ),
            let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
            let tokens = Range(match.range(at: 1), in: text).flatMap({ Int(text[$0]) }),
            let limit = Range(match.range(at: 2), in: text).flatMap({ Int(text[$0]) })
        {
            out["token_count"] = tokens
            out["context_size"] = limit
        }
    }
    return out
}

// MARK: - Truncation

/// `tokenCount(for:)` of the empty string: the constant framing the count
/// adds to any text (1 on macOS 27). Measured once per process.
private final class TokenOverhead: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Int?

    @available(macOS 26.4, *)
    func get() async throws -> Int {
        if let cached = lock.withLock({ value }) {
            return cached
        }
        let measured = try await SystemLanguageModel.default.tokenCount(for: "")
        lock.withLock { value = measured }
        return measured
    }
}

private let tokenOverhead = TokenOverhead()

/// Did generation stop because it hit `maxTokens`? Foundation Models has no
/// finish reason, so this is inferred; nil means "can't tell".
/// - Plain text: fewer UTF-8 bytes than the limit can't have hit it (every
///   token covers at least one byte); on macOS 27 an output count under the
///   limit can't either; otherwise re-count the text (~45 ms) and compare
///   net of the counting overhead. Measured on macOS 27: truncated replies
///   re-count to exactly the limit, natural ones well below it.
/// - Constrained output doesn't re-tokenize to what was generated (structure
///   tokens), and a truncated one still reports `isComplete`, so the macOS 27
///   per-response output count is the only signal.
@available(macOS 26.0, *)
private func isTruncated(text: String, constrained: Bool, maxTokens: Int, outputTokens: Int?) async -> Bool? {
    if let output = outputTokens, output < maxTokens {
        return false
    }
    if constrained {
        return outputTokens.map { $0 >= maxTokens }
    }
    if text.utf8.count < maxTokens {
        return false
    }
    guard #available(macOS 26.4, *) else { return nil }
    do {
        let overhead = try await tokenOverhead.get()
        let count = try await SystemLanguageModel.default.tokenCount(for: text)
        return count - overhead >= maxTokens
    } catch {
        return nil
    }
}

// MARK: - Respond

#if SK_SDK_27
@available(macOS 27.0, *)
private func usageJSON(_ usage: LanguageModelSession.Usage) -> [String: Any] {
    [
        "input": usage.input.totalTokenCount,
        "cached": usage.input.cachedTokenCount,
        "output": usage.output.totalTokenCount,
        "reasoning": usage.output.reasoningTokenCount,
    ]
}
#endif

private func errorEnvelope(_ error: [String: Any]) -> [String: Any] {
    ["text": NSNull(), "usage": NSNull(), "truncated": NSNull(), "error": error]
}

@available(macOS 26.0, *)
private func respondEnvelope(
    session: LanguageModelSession,
    prompt: String,
    schemaText: String,
    options: GenerationOptions,
    maxTokens: Int
) async -> [String: Any] {
    // Parse the schema first: a bad schema is the caller's error, not the
    // model's, and must not be classified as a generation failure.
    var schema: GenerationSchema?
    if !schemaText.isEmpty {
        do {
            guard
                let data = schemaText.data(using: .utf8),
                let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                return errorEnvelope(["kind": "unsupported_guide", "message": "schema is not a JSON object"])
            }
            let root = try dynamicSchema(from: parsed, name: "Response")
            schema = try GenerationSchema(root: root, dependencies: [])
        } catch {
            return errorEnvelope(["kind": "unsupported_guide", "message": String(describing: error)])
        }
    }

    do {
        let text: String
        var usage: Any = NSNull()
        var outputTokens: Int?
        if let schema {
            let response = try await session.respond(to: prompt, schema: schema, options: options)
            text = response.content.jsonString
            #if SK_SDK_27
            if #available(macOS 27.0, *) {
                usage = usageJSON(response.usage)
                outputTokens = response.usage.output.totalTokenCount
            }
            #endif
        } else {
            let response = try await session.respond(to: prompt, options: options)
            text = response.content
            #if SK_SDK_27
            if #available(macOS 27.0, *) {
                usage = usageJSON(response.usage)
                outputTokens = response.usage.output.totalTokenCount
            }
            #endif
        }
        var truncated: Any = NSNull()
        if maxTokens > 0,
           let hit = await isTruncated(
               text: text, constrained: schema != nil, maxTokens: maxTokens, outputTokens: outputTokens
           )
        {
            truncated = hit
        }
        return ["text": text, "usage": usage, "truncated": truncated, "error": NSNull()]
    } catch {
        return errorEnvelope(classify(error))
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

/// Blocking respond. Returns 0 with a JSON envelope in `out`:
///   {"text": str|null,
///    "usage": {"input","cached","output","reasoning"}|null,   (macOS 27)
///    "truncated": bool|null,
///    "error": {"kind","message","token_count"?,"context_size"?,
///              "retry_after_secs"?}|null}
/// Returns 1 with `err` set only for failures of the shim itself.
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
        var envelope: [String: Any] = [:]
        Task {
            defer { semaphore.signal() }
            envelope = await respondEnvelope(
                session: box.session,
                prompt: promptText,
                schemaText: schemaText,
                options: options,
                maxTokens: Int(maxTokens)
            )
        }
        semaphore.wait()

        if writeJSON(envelope, out, outLen) {
            return 0
        }
        setError(err, "could not encode the response envelope")
        return 1
    }
    #endif
    setError(err, "Foundation Models requires macOS 26 or later")
    return 1
}

/// Model facts as JSON: {"variant", "variant_id", "context_size",
/// "capabilities"} — each null when unknown. Variant and capabilities are
/// macOS 27 APIs; `capabilities` is built by probing `contains(_:)`, since
/// `LanguageModelCapabilities` doesn't enumerate. Only read while the model
/// is available. Returns 0 on success.
@_cdecl("sk_fm_model_info")
public func sk_fm_model_info(
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outLen: UnsafeMutablePointer<UInt>?
) -> Int32 {
    guard let out, let outLen else { return 1 }
    var info: [String: Any] = [
        "variant": NSNull(), "variant_id": NSNull(), "context_size": NSNull(), "capabilities": NSNull(),
    ]
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        let model = SystemLanguageModel.default
        if model.isAvailable {
            let size = model.contextSize
            if size > 0 {
                info["context_size"] = size
            }
            #if SK_SDK_27
            if #available(macOS 27.0, *) {
                let variant = model.variant
                info["variant"] = variant.displayName
                info["variant_id"] =
                    variant == .core3 ? "core3" : variant == .coreAdvanced3 ? "core_advanced3" : "other"
                let capabilities = model.capabilities
                let probes: [(String, LanguageModelCapabilities.Capability)] = [
                    ("guided_generation", .guidedGeneration),
                    ("tool_calling", .toolCalling),
                    ("vision", .vision),
                    ("reasoning", .reasoning),
                ]
                info["capabilities"] = probes.filter { capabilities.contains($0.1) }.map { $0.0 }
            }
            #endif
        }
    }
    #endif
    return writeJSON(info, out, outLen) ? 0 : 1
}

/// Self-test for `classify`: runs it over constructed errors (no model
/// needed, so it runs in CI) and returns JSON {"sdk27": bool, "runtime27":
/// bool, "cases": [{"case", "kind", ...}]} for a Rust test to check.
@_cdecl("sk_fm_selftest")
public func sk_fm_selftest(
    _ out: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>?>?,
    _ outLen: UnsafeMutablePointer<UInt>?
) -> Int32 {
    guard let out, let outLen else { return 1 }
    var result: [String: Any] = ["sdk27": false, "runtime27": false, "cases": [[String: Any]]()]
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        var cases: [[String: Any]] = []
        func add(_ name: String, _ error: Error) {
            var entry = classify(error)
            entry["case"] = name
            cases.append(entry)
        }
        typealias GenerationError = LanguageModelSession.GenerationError
        let context = GenerationError.Context(debugDescription: "selftest")
        add("gen.exceededContextWindowSize", GenerationError.exceededContextWindowSize(context))
        add("gen.assetsUnavailable", GenerationError.assetsUnavailable(context))
        add("gen.guardrailViolation", GenerationError.guardrailViolation(context))
        add("gen.refusal", GenerationError.refusal(.init(transcriptEntries: []), context))
        add("gen.unsupportedGuide", GenerationError.unsupportedGuide(context))
        add("gen.unsupportedLanguageOrLocale", GenerationError.unsupportedLanguageOrLocale(context))
        add("gen.decodingFailure", GenerationError.decodingFailure(context))
        add("gen.rateLimited", GenerationError.rateLimited(context))
        add("gen.concurrentRequests", GenerationError.concurrentRequests(context))
        add("message.contextOverflow", NSError(
            domain: "selftest", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                "Content contains 4459 tokens, which exceeds the maximum allowed context size of 4096."]
        ))
        add("message.other", NSError(
            domain: "selftest", code: 2, userInfo: [NSLocalizedDescriptionKey: "something else"]
        ))
        #if SK_SDK_27
        result["sdk27"] = true
        if #available(macOS 27.0, *) {
            result["runtime27"] = true
            add("lm.contextSizeExceeded", LanguageModelError.contextSizeExceeded(
                .init(contextSize: 4096, tokenCount: 5000, debugDescription: "selftest")))
            add("lm.rateLimited", LanguageModelError.rateLimited(
                .init(resetDate: Date().addingTimeInterval(30), debugDescription: "selftest")))
            add("lm.guardrailViolation", LanguageModelError.guardrailViolation(
                .init(debugDescription: "selftest")))
            add("lm.refusal", LanguageModelError.refusal(
                .init(explanation: "selftest", debugDescription: "selftest")))
            add("lm.timeout", LanguageModelError.timeout(.init(debugDescription: "selftest")))
            add("lm.unsupportedLanguageOrLocale", LanguageModelError.unsupportedLanguageOrLocale(
                .init(languageCode: Locale.LanguageCode("xx"), debugDescription: "selftest")))
            add("lm.unsupportedGenerationGuide", LanguageModelError.unsupportedGenerationGuide(
                .init(schemaName: "Response", debugDescription: "selftest")))
            add("session.concurrentRequests", LanguageModelSession.Error.concurrentRequests)
            add("session.transcriptMutationWhileResponding",
                LanguageModelSession.Error.transcriptMutationWhileResponding)
            add("system.assetsUnavailable", SystemLanguageModel.Error.assetsUnavailable(
                .init(debugDescription: "selftest")))
        }
        #endif
        result["cases"] = cases
    }
    #endif
    return writeJSON(result, out, outLen) ? 0 : 1
}
