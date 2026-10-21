"""The assembly stage on a tiny synthetic capture set: determinism, slot
splitting, and refusal of deliberately broken triples by every hazard rule.

The captures here are written for the test, not produced by a tool; they
only need the shapes real captures have."""

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from tasks import canon, gen_verification as gen, hazards


_SEQ = [0]


def cap(cid, cmd, stdout="", stderr="", exit=0, kind="stdout", world="w"):
    _SEQ[0] += 1
    return {"schema": "sidekick-verification-capture/1", "id": cid, "seq": _SEQ[0], "world": world, "tool": "demo",
            "kind": kind, "cmd": cmd, "stdout": stdout, "stderr": stderr, "exit": exit}


HELP = ("Move a marker\n\nUsage: demo move [OPTIONS]\n\nOptions:\n"
        "  -B, --allow-back\n          Allow moving the marker back\n\n"
        "  -h, --help\n          Print help\n")
FILLER_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]


def captures():
    caps = [
        cap("demo.help.move", "demo help move", HELP, kind="help"),
        cap("demo.err.move", "demo move m --to old", stderr="Error: refusing to move m back\nHint: pass --allow-back\n",
            exit=1, kind="stderr"),
        cap("demo.cat.cfg", "cat cfg.toml", '[[svc]]\nname = "api"\nreplicas = 3\n\n[[svc]]\nname = "worker"\nreplicas = 5\n',
            kind="toml"),
        cap("demo.sum", "demo sum replicas", "8\n"),
        cap("demo.w6.cat.cfg", "cat cfg.toml", '[[svc]]\nname = "api"\nreplicas = 3\n\n[[svc]]\nname = "worker"\nreplicas = 6\n',
            kind="toml", world="w6"),
        cap("demo.w6.sum", "demo sum replicas", "9\n", world="w6"),
        cap("demo.cat.old", "cat port.txt", "port = 7000\n", world="old"),
        cap("demo.cat.new", "cat port.txt", "port = 7001\n"),
        cap("demo.cat.json", "cat app.json", '{\n  "port": 7001,\n  "name": "x"\n}\n', kind="json"),
        cap("demo.cat.yaml", "cat app.yaml", "server:\n  port: 7001\n  pool:\n    max: 4\n", kind="yaml"),
        cap("demo.err.version", "demo --version", stderr="demo: unknown flag --version\n", exit=80, kind="stderr"),
    ]
    for i, w in enumerate(FILLER_WORDS * 6):
        caps.append(cap(f"demo.filler.{i:02d}", f"demo echo {w} {i}", (f"{w} line {i}\n" * 12), world="f"))
    return {c["id"]: c for c in caps}


CORPUS = {"seed": 7, "lengths": {"1k": 400, "4k": 800, "12k": 1200, "32k": 1600},
          "positions": {"start": 0.05, "middle": 0.5, "end": 0.95}, "dev_share": 0.2, "max_per_shape": 3,
          "filler": {"exclude_worlds": ["w6", "old"]}}
CFG = {"length": "1k", "position": "middle", "obstacle": "none"}


def flag_claim():
    return {
        "id": "demo-move-flag", "shape": "tool.flag", "type": "tool", "mechanism": "plain", "fact": "demo-move-back",
        "text": "gotcha: demo move refuses to move a marker back unless given --allow-back.",
        "value": "--allow-back", "value_c": "--force", "guard": ["--allow-back", "--force"],
        "probes": {"support": ["pass --allow-back", "^  -B, --allow-back$"],
                   "contradict": ["pass --force", "^  -F, --force$"],
                   "mention": ["allow-back", "move m back"]},
        "repeat": 0,
        "block": [
            {"capture": "demo.err.move", "edit": [
                {"find": "Hint: pass --allow-back\n", "c": "Hint: pass --force\n"},
                {"find": "Error: refusing to move m back\nHint: pass --allow-back\n", "nm": ""}]},
            {"capture": "demo.help.move", "edit": [
                {"find": "  -B, --allow-back\n", "c": "  -F, --force\n"},
                {"find": "  -B, --allow-back\n          Allow moving the marker back\n\n", "nm": ""}]},
        ],
    }


def sum_claim():
    return {
        "id": "demo-worker-replicas", "shape": "file.replicas", "type": "file", "mechanism": "plain",
        "fact": "demo-replicas",
        "text": "fact: cfg.toml gives the worker 5 replicas.", "guard": [],
        "probes": {"support": ['^name = "worker"\nreplicas = 5$'], "contradict": ['^name = "worker"\nreplicas = 6$'],
                   "mention": ["replicas"]},
        "invariants": [{"type": "sum", "items": r"^replicas = (\d+)$", "total": r"demo sum replicas\n(\d+)\n"}],
        "block": [
            {"capture": "demo.cat.cfg", "c_from": "demo.w6.cat.cfg", "nm_drop": True},
            {"capture": "demo.sum", "c_from": "demo.w6.sum", "nm_drop": True},
        ],
    }


def sequence_claim():
    return {
        "id": "demo-port-now", "shape": "file.port-now", "type": "file", "mechanism": "sequence",
        "fact": "demo-port",
        "text": "fact: port.txt now sets port = 7001.", "guard": [], "anchor": "b",
        "probes": {"support": ["^port = 7001$"], "contradict": ["^port = 7000$"], "mention": [r"^port = \d+$"]},
        "block": [
            {"capture": "demo.cat.old", "c_from": "demo.cat.new", "nm_drop": True, "role": "loser", "group": "a"},
            {"capture": "demo.cat.new", "c_from": "demo.cat.old", "nm_drop": True, "role": "winner", "group": "b"},
        ],
    }


def belief_claim():
    """SUPPORTED rests on a read of the file; CONTRADICTED marks the belief
    refuted beside a later read."""
    return {
        "id": "demo-port-belief", "shape": "file.belief", "type": "file", "mechanism": "supersede", "role": "belief",
        "fact": "demo-port-old",
        "text": "fact: port.txt sets port = 7000.", "guard": [],
        "probes": {"support": ["^port = 7000$", "sets port = 7000"], "contradict": ["^port = 7001$"],
                   "mention": [r"^port = \d+$"]},
        "block": [
            {"capture": "demo.cat.old", "c_drop": True, "nm_drop": True},
            {"capture": "demo.cat.new", "call": "c9", "s_drop": True, "nm_drop": True},
            {"notes": [],
             "c_notes": ["- was: port.txt sets port = 7000 (held at 6/10), wrong because a later read shows 7001 [call c9]"]},
        ],
    }


class Assembly(unittest.TestCase):
    def setUp(self):
        self.caps = captures()

    def build(self, claim, cfg=CFG):
        return gen.build(claim, dict(cfg), self.caps, CORPUS)

    def test_variants_differ_only_in_slots_and_expand_exactly(self):
        t = self.build(flag_claim())
        for v in gen.VARIANTS:
            self.assertEqual(canon.sha256_text(gen.expand(t, v)), t["evidence_sha256"][v])
        self.assertIn("Hint: pass --force", gen.expand(t, "c"))
        self.assertNotIn("allow-back", gen.expand(t, "nm"))
        self.assertIn("[exit 1]", gen.expand(t, "nm"), "NM keeps the command and its exit status")

    def test_deterministic_across_runs(self):
        a = canon.canonical_line(self.build(flag_claim()))
        b = canon.canonical_line(gen.build(flag_claim(), dict(CFG), captures(), copy.deepcopy(CORPUS)))
        self.assertEqual(a, b)

    def test_generate_twice_into_different_roots_is_byte_identical(self):
        claims_toml = "\n".join([
            '[[shape]]', 'id = "tool.flag"', 'type = "tool"',
            '[[shape.claim]]', 'id = "demo-move-flag"', 'fact = "demo-move-back"', 'repeat = 0',
            'text = "gotcha: demo move refuses to move a marker back unless given --allow-back."',
            'value = "--allow-back"', 'value_c = "--force"',
            "probes.support = ['pass --allow-back', '^  -B, --allow-back$']",
            "probes.contradict = ['pass --force', '^  -F, --force$']",
            "probes.mention = ['allow-back', 'move m back']",
            '[[shape.claim.block]]', 'capture = "demo.err.move"',
            'edit = [{ find = "Hint: pass --allow-back\\n", c = "Hint: pass --force\\n" },'
            ' { find = "Error: refusing to move m back\\nHint: pass --allow-back\\n", nm = "" }]',
            '[[shape.claim.block]]', 'capture = "demo.help.move"',
            'edit = [{ find = "  -B, --allow-back\\n", c = "  -F, --force\\n" },'
            ' { find = "  -B, --allow-back\\n          Allow moving the marker back\\n\\n", nm = "" }]',
        ])
        corpus_toml = "\n".join([
            "seed = 7", "max_per_shape = 3", "dev_share = 0.2",
            "[filler]", 'exclude_worlds = ["w6", "old"]',
            "[lengths]", "1k = 400", "4k = 800", "12k = 1200", "32k = 1600",
            "[positions]", "start = 0.05", "middle = 0.5", "end = 0.95",
            "[review]", "per_type = 2", 'lengths = ["1k", "4k"]', 'positions = ["start", "end"]',
            'worked = { length = "1k", position = "middle", obstacle = "none" }',
        ])
        outs = []
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            src = Path(d1) / "in"
            src.mkdir()
            (src / "claims.toml").write_text(claims_toml)
            (src / "corpus.toml").write_text(corpus_toml)
            canon.write_jsonl(src / "captures.jsonl", list(self.caps.values()))
            for root in (Path(d1) / "a", Path(d2) / "b"):
                triples, _, lock = gen.generate(src / "captures.jsonl", root, "review", src / "claims.toml",
                                                src / "corpus.toml", None)
                outs.append(((root / "verification-review.triples.jsonl").read_bytes(), lock))
                self.assertEqual(len(triples), 2)  # per_type = 2 for the one type present
        self.assertEqual(outs[0][0], outs[1][0])
        self.assertEqual(outs[0][1], outs[1][1])
        self.assertEqual(gen.compare(outs[0][1], outs[1][1]), [])

    def test_obstacles_and_mechanisms_build(self):
        self.build(flag_claim(), {**CFG, "obstacle": "repeated"})
        self.build(sum_claim())
        t = self.build(sequence_claim(), {**CFG, "position": "end"})
        self.assertEqual(t["obstacles"], ["current:sequence"])
        self.build(belief_claim())

    def test_edit_kind_follows_one_rule(self):
        # an edit of a tool's own message is fabricated; of a file view, textual
        self.assertEqual(self.build(flag_claim())["edit"], "fabricated")
        self.assertEqual(gen.edit_kind({"block": [{"capture": "demo.cat.json", "edit": [
            {"find": "7001", "c": "7002"}]}]}, self.caps), "textual")
        # another capture's output: same command line, sibling-world; another command, spliced
        self.assertEqual(self.build(sum_claim())["edit"], "sibling-world")
        self.assertEqual(self.build(sequence_claim())["edit"], "sibling-world")
        self.assertEqual(gen.edit_kind({"block": [{"capture": "demo.sum", "c_from": "demo.cat.new"}]}, self.caps),
                         "spliced")
        # the least real source wins
        self.assertEqual(gen.edit_kind({"block": [{"capture": "demo.sum", "c_from": "demo.w6.sum"},
                                                  {"capture": "demo.err.move", "edit": [{"find": "m", "c": "n"}]}]},
                                       self.caps), "fabricated")

    def test_position_is_measured_in_thirds_of_the_text(self):
        for pos in ("start", "middle", "end"):
            t = self.build(flag_claim(), {**CFG, "length": "4k", "position": pos})
            self.assertEqual(t["config"]["position"], gen.POSITION_THIRDS[min(2, int(t["position_at"] * 3))])
            self.assertEqual(t["config"]["position"], pos)

    def test_nm_kind_labels_empty_output_with_nonzero_exit(self):
        self.assertEqual(self.build(sum_claim())["nm_kind"], "removed")
        # NOT MENTIONED keeps the refused command and its exit 1, with no output
        self.assertEqual(self.build(flag_claim())["nm_kind"], "empty-output-nonzero-exit")
        c = {"id": "demo-version-refuted", "shape": "r.p", "type": "refutation", "mechanism": "plain", "fact": "dv",
             "text": "was: demo --version prints the version (held at 6/10), wrong because it is an unknown flag [call c8]",
             "guard": [],
             "probes": {"support": ["unknown flag --version"], "contradict": [r"\$ demo --version\n8\n"],
                        "mention": ["unknown flag"]},
             "block": [{"capture": "demo.err.version", "call": "c8", "c_from": "demo.sum", "nm_empty": True}]}
        self.assertEqual(self.build(c)["nm_kind"], "empty-output-nonzero-exit")

    def refutation_claim(self):
        return {"id": "demo-version-refuted", "shape": "r.p", "type": "refutation", "mechanism": "plain", "fact": "dv",
                "text": "was: demo --version prints the version (held at 6/10), wrong because it is an unknown flag "
                        "[call c8]",
                "guard": [],
                "probes": {"support": ["unknown flag --version"], "contradict": [r"\[call c8\] \$ demo --version\n8\n"],
                           "mention": ["unknown flag"]},
                "block": [{"capture": "demo.err.version", "call": "c8", "c_from": "demo.sum", "nm_drop": True},
                          {"notes": ["- was: demo --version prints the version, wrong because unknown flag [call c8]"],
                           "c_notes": [], "nm_notes": []}]}

    def test_monotonic_call_ids_and_pointers(self):
        corpus = {**CORPUS, "monotonic_calls": True}
        for pos in ("start", "middle", "end"):
            t = gen.build(self.refutation_claim(), {**CFG, "length": "4k", "position": pos}, self.caps, corpus)
            for v in gen.VARIANTS:
                text = gen.expand(t, v)
                ids = [int(x) for x in re.findall(r"^\[call c(\d+)\] \$", text, re.M)]
                self.assertEqual(ids, sorted(ids), "ids increase through the haystack")
                self.assertEqual(len(ids), len(set(ids)))
            ptr = re.search(r"\[call (c\d+)\]$", t["claim"]["text"]).group(1)
            for v in ("s", "c"):
                # the renumbered pointer still names the pointed capture's call
                m = re.search(r"^\[call " + ptr + r"\] \$ (.*)$", gen.expand(t, v), re.M)
                self.assertEqual(m.group(1), self.caps["demo.err.version"]["cmd"])
            self.assertIn(f"[call {ptr}]", gen.expand(t, "s").split("[notes]")[1].split("\n")[1])

    def test_a_pointer_to_a_missing_call_is_refused(self):
        c = self.refutation_claim()
        c["text"] = c["text"].replace("[call c8]", "[call c99]")
        with self.assertRaises(gen.Refusal) as ctx:
            gen.build(c, dict(CFG), self.caps, CORPUS)
        self.assertIn("not in the haystack", str(ctx.exception))

    def test_claims_for_a_later_batch_stay_out_of_the_plan(self):
        later = {**sum_claim(), "id": "demo-later", "since": 3}
        corpus = {**CORPUS, "review": {"per_type": 2, "lengths": ["1k"], "positions": ["start", "end"],
                                        "worked": {"length": "1k", "position": "middle", "obstacle": "none"}}}
        ids = {c["id"] for c, _ in gen.review_plan([sum_claim(), later], corpus)}
        self.assertEqual(ids, {"demo-worker-replicas"})
        ids = {c["id"] for c, _ in gen.review_plan([sum_claim(), later], {**corpus, "batch": 3})}
        self.assertEqual(ids, {"demo-worker-replicas", "demo-later"})

    def test_touch_keeps_same_key_fillers_out(self):
        c = flag_claim()
        c["touch"] = [r"\bbravo\b"]
        pool = gen.filler_pool(c, self.caps)
        self.assertTrue(pool)
        self.assertFalse([p for p in pool if "bravo" in self.caps[p]["cmd"]])

    def test_chronology_within_a_world_family(self):
        caps = dict(self.caps)
        # a mutation of world w2, whose captures are filler for a needle from w
        for i in range(40):
            caps[f"demo.w2.read{i}"] = cap(f"demo.w2.read{i}", f"demo read {i}", f"state {i}\n" * 8, world="w2")
        caps["demo.mut.abandon"] = cap("demo.mut.abandon", "demo abandon", stderr="Abandoned m\n", world="w-mut")
        corpus = {**CORPUS, "_parents": {"w-mut": "w"}, "filler": {"exclude_worlds": ["w6", "old"]}}
        c = {"id": "demo-abandon", "shape": "t.a", "type": "tool", "mechanism": "plain", "fact": "da",
             "text": "fact: demo abandon abandons m.", "guard": [],
             "probes": {"support": ["^Abandoned m$"], "contradict": ["^Abandoned n$"], "mention": ["Abandoned"]},
             "block": [{"capture": "demo.mut.abandon", "edit": [{"find": "Abandoned m", "c": "Abandoned n"}],
                        "nm_drop": True}]}
        for pos in ("start", "middle"):
            t = gen.build(c, {**CFG, "length": "32k", "position": pos}, caps, corpus)
            fill = [caps[b["source"]] for b in t["layout"] if b["item"] == "filler"]
            self.assertEqual([f["id"] for f in fill if f["world"] == "w"], [],
                             "no capture of the needle's own world family pads it")
            seqs = [f["seq"] for f in fill if f["world"] == "w2"]
            self.assertTrue(seqs)
            self.assertEqual(seqs, sorted(seqs), "one world's captures keep their order")

    def test_compact_review_is_small_and_complete(self):
        triples = [self.build(flag_claim()), self.build(sum_claim(), {**CFG, "position": "end"})]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "review-compact.md"
            gen.export_compact(triples, path, "test")
            text = path.read_text()
        for t in triples:
            self.assertIn(t["id"], text)
            self.assertIn(f"**edit:** {t['edit']}", text)
        self.assertIn("Hint: pass --force", text)
        self.assertIn("filler blocks: demo.filler", text)
        self.assertLess(len(text), 20000)

    def test_sibling_worlds_never_pad(self):
        pool = gen.filler_pool(flag_claim(), self.caps, CORPUS["filler"]["exclude_worlds"])
        self.assertNotIn("demo.w6.sum", pool)
        self.assertNotIn("demo.cat.old", pool)
        self.assertNotIn("demo.err.move", pool, "a needle capture is never filler")

    def test_pending_claim(self):
        c = flag_claim()
        c["block"][0]["capture"] = "demo.not.captured"
        with self.assertRaises(gen.Refusal):
            self.build(c)
        c["pending"] = "a recipe that has not run"
        with self.assertRaises(gen.Pending):
            self.build(c)


class SlotSplitting(unittest.TestCase):
    def test_three_way_line_slots(self):
        s = "a\nb\nc\nd\n"
        pieces = gen.split3(s, "a\nB\nc\nd\n", "a\nc\n")
        self.assertEqual(pieces, ["a\n", {"s": "b\n", "c": "B\n", "nm": ""}, "c\n",
                                  {"s": "d\n", "c": "d\n", "nm": ""}])

    def test_dropped_and_empty(self):
        self.assertEqual(gen.split3("", "x\n", ""), [{"s": "", "c": "x\n", "nm": ""}])
        self.assertEqual(gen.split3("", "", ""), [])


class Hazards(unittest.TestCase):
    """Each rule refuses a triple broken on purpose."""

    def setUp(self):
        self.caps = captures()

    def refused(self, claim, cfg=CFG, *, why):
        with self.assertRaises(gen.Refusal) as ctx:
            gen.build(claim, dict(cfg), self.caps, CORPUS)
        self.assertIn(why, str(ctx.exception))
        return str(ctx.exception)

    def test_restatement_every_copy_must_be_edited(self):
        c = flag_claim()
        c["block"][1]["edit"] = c["block"][1]["edit"][1:]  # the help's option line keeps --allow-back in C
        self.refused(c, why="restatement")

    def test_uniqueness_support_outside_the_needle(self):
        c = flag_claim()
        c["block"][1].pop("edit")  # the help page is now fixed text that still states the claim
        msg = self.refused(c, why="uniqueness")
        self.assertIn("outside the needle", msg)

    def test_nm_must_remove_every_mention(self):
        c = flag_claim()
        c["block"][0]["edit"] = c["block"][0]["edit"][:1]  # NM keeps the Error and Hint lines
        self.refused(c, why="nm:")

    def test_derived_values_must_stay_consistent(self):
        c = sum_claim()
        del c["block"][1]["c_from"]  # C changes a replica but leaves the sum at 8
        self.refused(c, why="derived")

    def test_edited_bodies_must_still_parse(self):
        c = {
            "id": "demo-json-port", "shape": "file.json-port", "type": "file", "mechanism": "plain",
            "text": "fact: app.json sets port 7001.", "guard": [],
            "probes": {"support": ['"port": 7001'], "contradict": ['"port": 7002'], "mention": ['"port"']},
            "block": [{"capture": "demo.cat.json", "nm_drop": True, "edit": [
                {"find": '"port": 7001,', "c": '"port": ,'}]}],
        }
        self.refused(c, why="no longer parses")

    def test_defined_synonyms_are_not_contradictions(self):
        c = flag_claim()
        c["value_c"] = "-B"
        c["guard"] = ["--allow-back"]
        c["probes"]["contradict"] = ["pass -B$"]
        c["block"][0]["edit"][0] = {"find": "Hint: pass --allow-back\n", "c": "Hint: pass -B\n"}
        c["block"][1]["edit"] = [{"find": "  -B, --allow-back\n          Allow moving the marker back\n\n", "nm": ""},
                                 {"find": "  -B, --allow-back\n", "c": "  -B, --allow-back\n"}]
        self.refused(c, why="synonym")

    def test_sequence_needs_the_current_record_last(self):
        c = sequence_claim()
        c["block"][0]["role"], c["block"][1]["role"] = "winner", "loser"
        c["probes"]["support"], c["probes"]["contradict"] = ["^port = 7000$"], ["^port = 7001$"]
        self.refused(c, why="sequence: the deciding record must come later")

    def test_supersede_belief_cannot_stay_live(self):
        c = belief_claim()
        c["block"][2]["c_notes"] = ["- fact: port.txt sets port = 7000"]
        self.refused(c, why="restatement")

    def test_notes_are_never_the_only_support(self):
        c = belief_claim()
        c["block"][0]["capture"] = "demo.sum"  # SUPPORTED's tool output no longer states it
        c["block"][2]["notes"] = ["- fact: port.txt sets port = 7000"]
        self.refused(c, why="rests only on notes")

    def test_nm_must_not_keep_an_exit_status_the_claim_states(self):
        c = {"id": "demo-version-refuted", "shape": "r.p", "type": "refutation", "mechanism": "plain", "fact": "dv",
             "text": "was: demo --version prints the version (held at 6/10), wrong because it exited 80 [call c8]",
             "guard": [],
             "probes": {"support": ["unknown flag --version"], "contradict": [r"\$ demo --version\n8\n"],
                        "mention": ["unknown flag"]},
             "block": [{"capture": "demo.err.version", "call": "c8", "c_from": "demo.sum", "nm_empty": True}]}
        self.refused(c, why="keeps exit 80")

    def test_nm_must_not_leave_an_empty_container(self):
        c = {"id": "demo-yaml-pool", "shape": "f.y", "type": "file", "mechanism": "plain", "fact": "dy",
             "text": "fact: app.yaml sets the pool max to 4.", "guard": [],
             "probes": {"support": ["^    max: 4$"], "contradict": ["^    max: 5$"], "mention": ["max:"]},
             "block": [{"capture": "demo.cat.yaml", "nm_closed_world": "a test of the declared shape",
                        "edit": [{"find": "    max: 4\n", "c": "    max: 5\n", "nm": ""}]}]}
        self.refused(c, why="empty container")
        c["block"][0]["edit"] = [{"find": "    max: 4\n", "c": "    max: 5\n"},
                                 {"find": "  pool:\n    max: 4\n", "nm": ""}]
        self.assertEqual(gen.build(c, dict(CFG), self.caps, CORPUS)["nm_kind"], "closed-world-absence")

    def test_nm_may_not_keep_a_file_with_the_key_removed(self):
        c = {"id": "demo-yaml-pool", "shape": "f.y", "type": "file", "mechanism": "plain", "fact": "dy",
             "text": "fact: app.yaml sets the pool max to 4.", "guard": [],
             "probes": {"support": ["^    max: 4$"], "contradict": ["^    max: 5$"], "mention": ["max:"]},
             "block": [{"capture": "demo.cat.yaml", "edit": [{"find": "    max: 4\n", "c": "    max: 5\n"},
                                                              {"find": "  pool:\n    max: 4\n", "nm": ""}]}]}
        self.refused(c, why="whole with the claimed key removed")
        del c["block"][0]["edit"][1]
        c["block"][0]["nm_drop"] = True
        self.assertEqual(gen.build(c, dict(CFG), self.caps, CORPUS)["nm_kind"], "removed")

    def test_a_note_must_follow_the_call_it_cites(self):
        c = belief_claim()
        c["block"][1], c["block"][2] = c["block"][2], c["block"][1]
        self.refused(c, why="points forward")

    def test_the_mechanism_exempts_the_losing_slot_from_the_value_rule(self):
        # sequence: the claimed value (7001) sits in the losing (early) slot of C
        c = sequence_claim()
        c["value"], c["value_c"] = "7001", "7000"
        c["guard"] = ["7001"]
        t = gen.build(c, dict(CFG), self.caps, CORPUS)
        self.assertIn("port = 7001", gen.expand(t, "c"))
        # the same value outside the losing slot is refused
        c["c_forbid"] = ["^port = 7001$"]
        c["block"].append({"capture": "demo.cat.new"})  # a fixed, third read showing 7001
        with self.assertRaises(gen.Refusal):
            gen.build(c, dict(CFG), self.caps, CORPUS)
        # supersede belief: the value on a `was:` line of C is the refutation itself
        b = belief_claim()
        b["value"], b["value_c"] = "7000", "7001"
        b["guard"] = ["7000"]
        self.assertIn("was: port.txt sets port = 7000", gen.expand(gen.build(b, dict(CFG), self.caps, CORPUS), "c"))

    def test_review_export_is_named_after_the_set_and_ordered_by_number(self):
        triples = [gen.build(flag_claim(), dict(CFG), self.caps, CORPUS),
                   gen.build(sum_claim(), {**CFG, "position": "end"}, self.caps, CORPUS)]
        with tempfile.TemporaryDirectory() as d:
            gen.export_review(triples, Path(d), "review-9z")
            names = sorted(p.name for p in (Path(d) / "review").iterdir())
            rows = [json.loads(x) for x in (Path(d) / "review" / "review-9z.jsonl").read_text().splitlines()]
        self.assertIn("review-9z.md", names)
        self.assertIn("review-9z-compact.md", names)
        self.assertTrue(all(n.startswith("review-9z") for n in names))
        self.assertEqual([r["n"] for r in rows], list(range(1, len(rows) + 1)))

    def test_contradicted_may_not_still_state_the_value(self):
        c = flag_claim()
        c["block"][0]["edit"][0] = {"find": "Hint: pass --allow-back\n", "c": "Hint: pass --force (or --allow-back)\n"}
        self.refused(c, why="the claim's value '--allow-back' is still stated")
        c["c_allow"] = [{"text": "--allow-back", "reason": "a test of the opt-out"}]
        gen.build(c, dict(CFG), self.caps, CORPUS)
        c = flag_claim()
        c["c_forbid"] = ["Allow moving the marker back"]  # the help's description stays in C
        self.refused(c, why="still states the claim")

    def test_nm_must_drop_every_conjunct(self):
        c = flag_claim()
        c["conjuncts"] = ["^\\[exit 1\\]$"]  # a conjunct NOT MENTIONED keeps
        self.refused(c, why="conjunct still matches")

    def test_a_state_changing_command_is_never_repeated(self):
        corpus = {**CORPUS, "_mutating": {"demo.err.move"}}
        with self.assertRaises(gen.Refusal) as ctx:
            gen.build(flag_claim(), {**CFG, "obstacle": "repeated"}, self.caps, corpus)
        self.assertIn("state-changing", str(ctx.exception))
        plan = gen.review_plan([flag_claim()], {**corpus, "review": {
            "per_type": 3, "lengths": ["1k"], "positions": ["start", "middle", "end"],
            "worked": {"length": "1k", "position": "middle", "obstacle": "none"}}})
        self.assertNotIn("repeated", {cfg["obstacle"] for _, cfg in plan})

    def test_unedited_find_is_refused(self):
        c = flag_claim()
        c["block"][0]["edit"][0]["find"] = "Hint: text that is not there\n"
        self.refused(c, why="edit:")


class TimestampRule(unittest.TestCase):
    """The checker on hand-made evidence: the later timestamp must decide and
    come first in the text."""

    claim = {"mechanism": "timestamp", "time_re": r'"ts":"[^"]+"',
             "probes": {"support": ['"status":"passed"'], "contradict": ['"status":"failed"'], "mention": ['"job"']}}

    @staticmethod
    def evidence(first, second):
        rows = {v: [] for v in hazards.VARIANTS}
        spans = {v: [] for v in hazards.VARIANTS}
        for i, (ts, s_status, c_status) in enumerate((first, second)):
            for v, status in (("s", s_status), ("c", c_status), ("nm", None)):
                start = sum(len(x) for x in rows[v])
                line = "" if status is None else json.dumps({"ts": ts, "job": "b7", "status": status},
                                                            separators=(",", ":")) + "\n"
                rows[v].append(line)
                spans[v].append((start, start + len(line), i))
        return {v: hazards.Evidence("".join(rows[v]), spans[v]) for v in hazards.VARIANTS}

    def test_accepts_later_time_written_first(self):
        ev = self.evidence(("12:00:09", "passed", "failed"), ("12:00:04", "failed", "passed"))
        self.assertEqual(hazards.check(self.claim, ev, ["winner", "loser"]), [])

    def test_refuses_when_position_agrees_with_time(self):
        ev = self.evidence(("12:00:04", "failed", "passed"), ("12:00:09", "passed", "failed"))
        problems = hazards.check(self.claim, ev, ["loser", "winner"])
        self.assertTrue(any("position can't decide" in p for p in problems), problems)

    def test_refuses_when_the_winner_is_older(self):
        ev = self.evidence(("12:00:01", "passed", "failed"), ("12:00:04", "failed", "passed"))
        problems = hazards.check(self.claim, ev, ["winner", "loser"])
        self.assertTrue(any("later time" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
