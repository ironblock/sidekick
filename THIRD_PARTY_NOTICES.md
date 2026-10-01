# Third-party notices

sidekick is MIT-licensed (see `LICENSE`). It includes, or is derived from, the following third-party work under other licenses. Rust crate dependencies carry their own licenses and are not listed here.

## laya

- **Source:** [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
- **License:** Apache License 2.0 (full text below)
- **What sidekick uses:** `crates/sidekick-embed/src/laya.rs` is a Rust port of `render_options` and `build_sequence` from laya's `rl_common.py`. It is compiled into `sidekickd` and `libsidekick.dylib`.
- **Changes:** translated from Python to Rust. Tokenization goes through the `tokenizers` crate, and the special tokens are resolved from the model's `tokenizer.json`. The option, question and state budgets, cut points and default texts are unchanged. Additions: each text is capped at 16 bytes per token of its budget before tokenizing, which bounds the tokenizer's work; noul labels are parsed from sidekick's `"false: …"` / `"true: …"` request form; and labels that become identical at the token level after shrinking are rejected.

`tools/convert_laya.py` imports `rl_common.py` from a local copy of the checkpoint at the same revision. It does not vendor it.

## gliner2

- **Source:** [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2), the `gliner2` Python package, version 2.0.0
- **License:** Apache License 2.0 (full text below)
- **What sidekick uses:** `crates/sidekick-embed/src/gliner2.rs` is a Rust port of the classification input layout from `processor.py` (`_transform_schema`, `_format_input_with_mapping`, and the terminal-punctuation rule of `collate_fn_inference`) and of `WhitespaceTokenSplitter`'s pattern from `processing/word_splitter.py`. It is compiled into `sidekickd` and `libsidekick.dylib`.
- **Changes:** translated from Python to Rust. Tokenization goes through the `tokenizers` crate, and the marker tokens are resolved from the model's `tokenizer.json`. The word-splitter pattern's character classes are spelled out to keep Python's `re` semantics under Rust's `regex` crate. The schema layout, marker positions, per-item tokenization, lowercasing and terminal punctuation are unchanged. Additions: one task per request; labels sent as `"key: description"` become gliner2 label descriptions; an over-length text is truncated at the word level before its terminal punctuation is added (gliner2 doesn't truncate by default); each text is capped at 16 bytes per token of the model's maximum before splitting; and empty names, duplicate names and labels identical at the token level are rejected.

`tools/classifier_reference.py` imports the installed `gliner2` package to generate references and the token-id fixture. It does not vendor it.

## Julia-1

- **Source:** [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), revision `a85b127321d580d65176c89ced8273f305745d85`
- **License:** Apache License 2.0 (full text below)
- **What sidekick uses:** `crates/sidekick-embed/src/laya.rs`'s Julia-1 option rendering (`option_rendering = "julia"`) reproduces the option texts Julia-1's typed API builds (`julia/typed.py`): a choice option's description, score items as given, and `"false"`/`"true"` or both descriptions for noul. It is compiled into `sidekickd` and `libsidekick.dylib`. The sequence around the options is laya's (above), which Julia-1's `julia/data.py` also follows.
- **Changes:** sidekick takes options as candidate labels, so a choice option's description is the text after a label's first `": "`, and the label itself when it has none, which Julia-1's API, where every choice has a description, doesn't allow. An over-long state is truncated, as Julia-1's non-strict mode does; its strict mode, which rejects one, isn't implemented.

`tools/convert_julia.py` and `tools/classifier_reference.py` import `julia/data.py` and `julia/model.py` from a local copy of the checkpoint at the same revision, after checking their sha256. They don't vendor them. `fixtures/classify/julia-1.tokens.json` holds token ids that `julia/data.py` produced.

## Lumma-fev

- **Source:** [FrontiersMind/Lumma-fev-0.1b](https://huggingface.co/FrontiersMind/Lumma-fev-0.1b), revision `085f4705aa860a6404d6cc3ff17de8a2969ac0f4`. Its NOTICE reads: "Lumma-Fev. Copyright 2026 FrontiersMind. Portions of the model and serving code are adapted from work Copyright 2026 Jared Palmer, licensed under the Apache License, Version 2.0."
- **License:** Apache License 2.0 (full text below)
- **What sidekick uses:** the fev format's input builder (docs/design/classify.md, "The fev format") reproduces the row `modeling_fev.py` builds: its `render`, `option_text`, `pack` and `encode`, including the rewrite of `<|name|>` to `<¦name¦>` in request text.
- **Changes:** sidekick takes options as candidate labels, so a choice option is the label as given and a noul question's labels are `false` and `true` (rendered as fev's `no` and `yes`), and a request carries one question. States are strings only.

`tools/classifier_reference.py` imports `modeling_fev.py`, `modeling_nandi.py` and their configuration modules from a local copy of the checkpoint at the same revision, after checking their sha256. It doesn't vendor them. `fixtures/classify/lumma-fev-0.1b.tokens.json` holds token ids that `modeling_fev.py` produced.

## fast-decisions

- **Source:** [fastino/fast-decisions](https://huggingface.co/datasets/fastino/fast-decisions), revision `1a33070`
- **License:** Apache License 2.0 (full text below)
- **What sidekick uses:** `fixtures/classify/laya-en.corpus.toml` describes a mechanical translation of the dataset into laya's question format, used to measure laya-en's accuracy on each compute path; `fixtures/classify/laya-typed-decisions.corpus.toml`, `fixtures/classify/julia-1.corpus.toml` and `fixtures/classify/lumma-fev-0.1b.corpus.toml` reuse that translation for laya-typed-decisions, Julia-1 and Lumma-fev-0.1b. `fixtures/classify/gliner2.5-decide.corpus.toml` sends each of its classification heads as a gliner2-format request, to measure GLiNER2.5-Decide's parity on each compute path. The dataset is fetched when a reference is generated, not committed. As text and as token ids, `fixtures/classify/laya-en.tokens.json`, `laya-typed-decisions.tokens.json`, `julia-1.tokens.json` and `lumma-fev-0.1b.tokens.json` each include three of its rows, and `gliner2.5-decide.tokens.json` two.

---

                                     Apache License
                               Version 2.0, January 2004
                            http://www.apache.org/licenses/

       TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

       1. Definitions.

          "License" shall mean the terms and conditions for use, reproduction,
          and distribution as defined by Sections 1 through 9 of this document.

          "Licensor" shall mean the copyright owner or entity authorized by
          the copyright owner that is granting the License.

          "Legal Entity" shall mean the union of the acting entity and all
          other entities that control, are controlled by, or are under common
          control with that entity. For the purposes of this definition,
          "control" means (i) the power, direct or indirect, to cause the
          direction or management of such entity, whether by contract or
          otherwise, or (ii) ownership of fifty percent (50%) or more of the
          outstanding shares, or (iii) beneficial ownership of such entity.

          "You" (or "Your") shall mean an individual or Legal Entity
          exercising permissions granted by this License.

          "Source" form shall mean the preferred form for making modifications,
          including but not limited to software source code, documentation
          source, and configuration files.

          "Object" form shall mean any form resulting from mechanical
          transformation or translation of a Source form, including but
          not limited to compiled object code, generated documentation,
          and conversions to other media types.

          "Work" shall mean the work of authorship, whether in Source or
          Object form, made available under the License, as indicated by a
          copyright notice that is included in or attached to the work
          (an example is provided in the Appendix below).

          "Derivative Works" shall mean any work, whether in Source or Object
          form, that is based on (or derived from) the Work and for which the
          editorial revisions, annotations, elaborations, or other modifications
          represent, as a whole, an original work of authorship. For the purposes
          of this License, Derivative Works shall not include works that remain
          separable from, or merely link (or bind by name) to the interfaces of,
          the Work and Derivative Works thereof.

          "Contribution" shall mean any work of authorship, including
          the original version of the Work and any modifications or additions
          to that Work or Derivative Works thereof, that is intentionally
          submitted to Licensor for inclusion in the Work by the copyright owner
          or by an individual or Legal Entity authorized to submit on behalf of
          the copyright owner. For the purposes of this definition, "submitted"
          means any form of electronic, verbal, or written communication sent
          to the Licensor or its representatives, including but not limited to
          communication on electronic mailing lists, source code control systems,
          and issue tracking systems that are managed by, or on behalf of, the
          Licensor for the purpose of discussing and improving the Work, but
          excluding communication that is conspicuously marked or otherwise
          designated in writing by the copyright owner as "Not a Contribution."

          "Contributor" shall mean Licensor and any individual or Legal Entity
          on behalf of whom a Contribution has been received by Licensor and
          subsequently incorporated within the Work.

       2. Grant of Copyright License. Subject to the terms and conditions of
          this License, each Contributor hereby grants to You a perpetual,
          worldwide, non-exclusive, no-charge, royalty-free, irrevocable
          copyright license to reproduce, prepare Derivative Works of,
          publicly display, publicly perform, sublicense, and distribute the
          Work and such Derivative Works in Source or Object form.

       3. Grant of Patent License. Subject to the terms and conditions of
          this License, each Contributor hereby grants to You a perpetual,
          worldwide, non-exclusive, no-charge, royalty-free, irrevocable
          (except as stated in this section) patent license to make, have made,
          use, offer to sell, sell, import, and otherwise transfer the Work,
          where such license applies only to those patent claims licensable
          by such Contributor that are necessarily infringed by their
          Contribution(s) alone or by combination of their Contribution(s)
          with the Work to which such Contribution(s) was submitted. If You
          institute patent litigation against any entity (including a
          cross-claim or counterclaim in a lawsuit) alleging that the Work
          or a Contribution incorporated within the Work constitutes direct
          or contributory patent infringement, then any patent licenses
          granted to You under this License for that Work shall terminate
          as of the date such litigation is filed.

       4. Redistribution. You may reproduce and distribute copies of the
          Work or Derivative Works thereof in any medium, with or without
          modifications, and in Source or Object form, provided that You
          meet the following conditions:

          (a) You must give any other recipients of the Work or
              Derivative Works a copy of this License; and

          (b) You must cause any modified files to carry prominent notices
              stating that You changed the files; and

          (c) You must retain, in the Source form of any Derivative Works
              that You distribute, all copyright, patent, trademark, and
              attribution notices from the Source form of the Work,
              excluding those notices that do not pertain to any part of
              the Derivative Works; and

          (d) If the Work includes a "NOTICE" text file as part of its
              distribution, then any Derivative Works that You distribute must
              include a readable copy of the attribution notices contained
              within such NOTICE file, excluding those notices that do not
              pertain to any part of the Derivative Works, in at least one
              of the following places: within a NOTICE text file distributed
              as part of the Derivative Works; within the Source form or
              documentation, if provided along with the Derivative Works; or,
              within a display generated by the Derivative Works, if and
              wherever such third-party notices normally appear. The contents
              of the NOTICE file are for informational purposes only and
              do not modify the License. You may add Your own attribution
              notices within Derivative Works that You distribute, alongside
              or as an addendum to the NOTICE text from the Work, provided
              that such additional attribution notices cannot be construed
              as modifying the License.

          You may add Your own copyright statement to Your modifications and
          may provide additional or different license terms and conditions
          for use, reproduction, or distribution of Your modifications, or
          for any such Derivative Works as a whole, provided Your use,
          reproduction, and distribution of the Work otherwise complies with
          the conditions stated in this License.

       5. Submission of Contributions. Unless You explicitly state otherwise,
          any Contribution intentionally submitted for inclusion in the Work
          by You to the Licensor shall be under the terms and conditions of
          this License, without any additional terms or conditions.
          Notwithstanding the above, nothing herein shall supersede or modify
          the terms of any separate license agreement you may have executed
          with Licensor regarding such Contributions.

       6. Trademarks. This License does not grant permission to use the trade
          names, trademarks, service marks, or product names of the Licensor,
          except as required for reasonable and customary use in describing the
          origin of the Work and reproducing the content of the NOTICE file.

       7. Disclaimer of Warranty. Unless required by applicable law or
          agreed to in writing, Licensor provides the Work (and each
          Contributor provides its Contributions) on an "AS IS" BASIS,
          WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
          implied, including, without limitation, any warranties or conditions
          of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
          PARTICULAR PURPOSE. You are solely responsible for determining the
          appropriateness of using or redistributing the Work and assume any
          risks associated with Your exercise of permissions under this License.

       8. Limitation of Liability. In no event and under no legal theory,
          whether in tort (including negligence), contract, or otherwise,
          unless required by applicable law (such as deliberate and grossly
          negligent acts) or agreed to in writing, shall any Contributor be
          liable to You for damages, including any direct, indirect, special,
          incidental, or consequential damages of any character arising as a
          result of this License or out of the use or inability to use the
          Work (including but not limited to damages for loss of goodwill,
          work stoppage, computer failure or malfunction, or any and all
          other commercial damages or losses), even if such Contributor
          has been advised of the possibility of such damages.

       9. Accepting Warranty or Additional Liability. While redistributing
          the Work or Derivative Works thereof, You may choose to offer,
          and charge a fee for, acceptance of support, warranty, indemnity,
          or other liability obligations and/or rights consistent with this
          License. However, in accepting such obligations, You may act only
          on Your own behalf and on Your sole responsibility, not on behalf
          of any other Contributor, and only if You agree to indemnify,
          defend, and hold each Contributor harmless for any liability
          incurred by, or claims asserted against, such Contributor by reason
          of your accepting any such warranty or additional liability.

       END OF TERMS AND CONDITIONS

       APPENDIX: How to apply the Apache License to your work.

          To apply the Apache License to your work, attach the following
          boilerplate notice, with the fields enclosed by brackets "[]"
          replaced with your own identifying information. (Don't include
          the brackets!)  The text should be enclosed in the appropriate
          comment syntax for the file format. We also recommend that a
          file or class name and description of purpose be included on the
          same "printed page" as the copyright notice for easier
          identification within third-party archives.

       Copyright [yyyy] [name of copyright owner]

       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at

           http://www.apache.org/licenses/LICENSE-2.0

       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.
