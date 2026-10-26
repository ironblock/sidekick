// Foundation Models' guardrails on the chat benchmark's refused requests:
// each request run with the default guardrails and with
// `.permissiveContentTransformations`, as plain text and, when it carries a
// fixed-vocabulary tags schema, guided by that schema. Prints one JSON line
// per (request, guardrails, mode): whether the model refused, and its reply.
//
//     xcrun swiftc -O -parse-as-library tools/chat_bench/guardrails_probe.swift -o <out>/guardrails_probe
//     <out>/guardrails_probe <requests.json>
//
// <requests.json> is a list of {"id", "system", "user", "vocabulary"?}
// (`python tools/chat_bench/tasks.py refused <run file> [controls]` writes it).
// Temperature 0 (as sidekick passes it), one fresh session per request, macOS 26 or later.

import Foundation
import FoundationModels

struct Probe: Decodable {
    let id: String
    let system: String
    let user: String
    let vocabulary: [String]?
}

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

@main
struct Main {
    static func main() async throws {
        let path = CommandLine.arguments[1]
        let probes = try JSONDecoder().decode([Probe].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        let models: [(String, SystemLanguageModel)] = [
            ("default", SystemLanguageModel.default),
            ("permissive", SystemLanguageModel(guardrails: .permissiveContentTransformations)),
        ]
        for p in probes {
            for (gname, model) in models {
                var modes = ["text"]
                if p.vocabulary != nil { modes.append("guided") }
                for mode in modes {
                    let session = LanguageModelSession(model: model, instructions: p.system)
                    let options = GenerationOptions(temperature: 0)   // as sidekick sends temperature 0
                    do {
                        var reply: String
                        if mode == "guided", let vocab = p.vocabulary {
                            let item = DynamicGenerationSchema(name: "tagsItem", anyOf: vocab)
                            let tags = DynamicGenerationSchema(arrayOf: item, minimumElements: 1, maximumElements: 3)
                            let root = DynamicGenerationSchema(name: "Response", properties: [
                                .init(name: "tags", schema: tags),
                            ])
                            let schema = try GenerationSchema(root: root, dependencies: [])
                            let r = try await session.respond(to: p.user, schema: schema, options: options)
                            reply = r.content.jsonString
                        } else {
                            let r = try await session.respond(to: p.user, options: options)
                            reply = r.content
                        }
                        emit(["id": p.id, "guardrails": gname, "mode": mode, "refused": false,
                              "reply": String(reply.prefix(160))])
                    } catch {
                        emit(["id": p.id, "guardrails": gname, "mode": mode, "refused": true,
                              "error": String(describing: error).prefix(160).description])
                    }
                }
            }
        }
    }
}
