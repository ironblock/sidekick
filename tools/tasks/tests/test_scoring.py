"""The verification scoring harness: windows, anchors and deciding spans, the
verdict rules, the control gate, the bootstrap and MDE, the controls, the
committed ceilings, and the judge export's format.

Standard library only: a whitespace tokenizer stands in for tokenizer.json
(the ceilings themselves are computed with the real tokenizers and
committed; these tests check that file against the corpus lock)."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tasks import canon, gen_verification as gen, scoring_controls as sc, scoring_export as ex
from tasks import scoring_judge as judge, scoring_report as report, scoring_rules as rules
from tasks import scoring_set, scoring_windows as sw, scrub

REPO = Path(__file__).resolve().parents[3]
TOK = sw.WordTokenizer()


def lines(n, words=5, tag="w"):
    return "".join(" ".join(f"{tag}{i}x{j}" for j in range(words)) + "\n" for i in range(n))


class Windows(unittest.TestCase):
    def test_every_window_fits_and_overlaps(self):
        text = lines(400)
        claim = "the claim has six words here"
        ws = sw.windows(text, TOK, claim, 512)
        budget = sw.text_budget(TOK, claim, 512)
        self.assertEqual(budget, 512 - 6 - 3 - 8)
        lens = sw.fits(ws, TOK, claim, 512)
        self.assertTrue(all(n <= 512 for n in lens))
        self.assertTrue(all(TOK.count(w.text) <= budget for w in ws))
        for a, b in zip(ws, ws[1:]):
            shared = text[b.start:a.end]
            self.assertGreaterEqual(TOK.count(shared), 128, "consecutive windows overlap by at least 128 tokens")
            self.assertGreater(b.start, a.start)
        self.assertEqual(ws[0].start, 0)
        self.assertEqual(ws[-1].end, len(text) - 1)  # everything is covered; the final newline is dropped
        self.assertTrue(all(w.text == text[w.start:w.end] for w in ws))

    def test_windows_cut_at_line_boundaries(self):
        text = lines(300)
        for w in sw.windows(text, TOK, "c", 512):
            self.assertTrue(w.start == 0 or text[w.start - 1] == "\n")
            self.assertTrue(w.end == len(text) or text[w.end] == "\n")

    def test_a_line_longer_than_the_budget_is_cut_at_tokens(self):
        text = " ".join(f"t{i}" for i in range(2000)) + "\n"
        ws = sw.windows(text, TOK, "c", 512)
        self.assertGreater(len(ws), 3)
        sw.fits(ws, TOK, "c", 512)

    def test_short_text_is_one_window(self):
        self.assertEqual(len(sw.windows(lines(10), TOK, "c", 512)), 1)

    def test_a_claim_that_leaves_no_room_is_refused(self):
        with self.assertRaises(ValueError):
            sw.windows(lines(10), TOK, " ".join(["w"] * 400), 512)

    def test_fits_reports_an_over_long_window(self):
        with self.assertRaises(AssertionError):
            sw.fits([sw.Window(0, 1, " ".join(["w"] * 600))], TOK, "c", 512)

    def test_chunks_do_not_overlap(self):
        text = lines(300)
        cs = sw.chunks(text, TOK, 400)
        for a, b in zip(cs, cs[1:]):
            self.assertLessEqual(a.end, b.start)
        self.assertTrue(all(TOK.count(c.text) <= 400 for c in cs))


def sentences(line):
    """A stand-in for nltk's sent_tokenize: split after ". "."""
    out, at = [], 0
    for i in range(len(line)):
        if line[i] == "." and line[i + 1:i + 2] == " ":
            out.append(line[at:i + 1].strip())
            at = i + 2
    tail = line[at:].strip()
    return [s for s in out if s] + ([tail] if tail else [])


class MiniCheckChunks(unittest.TestCase):
    def test_sentences_grouped_within_the_size(self):
        text = "".join(f"Alpha {i} one. Beta {i} two three. Gamma {i}\n" for i in range(60)) + "\nlast line"
        cs = sw.minicheck_chunks(text, TOK.count, sentences, 40)
        self.assertGreater(len(cs), 5)
        for c in cs:
            self.assertLessEqual(TOK.count(c.text), 40)
            self.assertEqual(c.text, c.text.strip())
            # The span runs from the chunk's first sentence to its last (the text
            # differs only in whitespace: MiniCheck joins sentences with spaces
            # and collapses " \n " but not " \n \n ").
            self.assertEqual(text[c.start:c.end].split(), c.text.split())
        for a, b in zip(cs, cs[1:]):
            self.assertLessEqual(a.end, b.start)
        self.assertTrue(cs[-1].text.endswith("last line"))

    def test_minicheck_joins_sentences_with_spaces(self):
        cs = sw.minicheck_chunks("A b. C d.\nE f.\n", TOK.count, sentences, 400)
        self.assertEqual([c.text for c in cs], ["A b. C d.\nE f."])
        self.assertEqual((cs[0].start, cs[0].end), (0, 14))

    def test_a_sentence_over_the_size_stands_alone(self):
        cs = sw.minicheck_chunks("a b c d e f g h\ni j\n", TOK.count, sentences, 4)
        self.assertEqual([c.text for c in cs], ["a b c d e f g h", "i j"])


class Dedupe(unittest.TestCase):
    def test_windows_outside_the_needle_are_shared_by_variants(self):
        base = lines(200)
        s = base + "[call c1] $ demo\nport = 1\n[exit 0]\n"
        c = base + "[call c1] $ demo\nport = 2\n[exit 0]\n"
        claim = "port is 1"
        ks = [sw.key(w.text, claim) for w in sw.windows(s, TOK, claim, 512)]
        kc = [sw.key(w.text, claim) for w in sw.windows(c, TOK, claim, 512)]
        self.assertGreater(len(set(ks) & set(kc)), 0)
        self.assertNotEqual(ks[-1], kc[-1])
        self.assertNotEqual(sw.key("a", "b"), sw.key("a", "c"))

    def test_export_sends_a_shared_window_once(self):
        items = [{"item": "t1:s", "triple": "t1", "variant": "S", "claim": "x", "evidence": "same\n"},
                 {"item": "t1:c", "triple": "t1", "variant": "C", "claim": "x", "evidence": "same\n"}]
        cuts = {"window-512": {i["item"]: [sw.Window(0, 4, "same")] for i in items}}
        reqs = ex.build(items, cuts, {"system": "s", "full": "{claim}{evidence}", "window": "{claim}{evidence}",
                                      "sha256": "p"}, {"max_tokens": 9, "reasoning_effort": "medium"},
                        forms=("window-512",))
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0]["serves"], [{"triple": "t1", "variant": "S", "window": 0},
                                             {"triple": "t1", "variant": "C", "window": 0}])


class Anchors(unittest.TestCase):
    def test_the_fixed_rule(self):
        self.assertEqual(sw.anchors("fact: jj file show picks the revision with -r/--revision."), ["-r", "--revision"])
        self.assertEqual(sw.anchors("fact: package.json pins packageManager to pnpm@9.15.4."),
                         ["package.json", "9.15.4"])
        self.assertEqual(sw.anchors("fact: the repo has a .jj directory and no .git directory."), [".jj", ".git"])
        self.assertEqual(sw.anchors("fact: jj's template language has no `id` keyword"), ["id"])
        self.assertEqual(sw.anchors("learned: abandoning 'tweak dressing' rebased one commit."), [])
        self.assertEqual(sw.anchors("fact: user.email is ana@example.com."), ["user.email"])

    def test_gate_mask(self):
        ws = [sw.Window(0, 1, "uses --force here"), sw.Window(1, 2, "nothing")]
        self.assertEqual(sw.gated(ws, ["--force"]), [True, False])


def synth_triple():
    """Two slots: an early stale record (loser) and a late one (winner)."""
    filler = lines(150)
    segs = [filler, filler, "[exit 0]\n"]
    slots = [{"s": "[call c1] $ cat a\nport = 1\n", "c": "[call c1] $ cat a\nport = 2\n", "nm": "",
              "role": "loser", "block": 0},
             {"s": "[call c2] $ cat a\nport = 2\n", "c": "[call c2] $ cat a\nport = 1\n", "nm": "",
              "role": "winner", "block": 1}]
    return {"segments": segs, "slots": slots}


class DecidingSpans(unittest.TestCase):
    def claim(self, mech="sequence"):
        return {"mechanism": mech, "probes": {"support": [r"^port = 2$"], "contradict": [r"^port = 1$"]}}

    def test_sequence_needs_both_records_in_one_window(self):
        t = synth_triple()
        for v, want in (("s", False), ("c", False), ("nm", True)):
            text = gen.expand(t, v)
            groups = sw.deciding_groups(t, self.claim(), v, text)
            small = sw.windows(text, TOK, "port is 2", 512)
            self.assertEqual(sw.reachable(small, groups), want, v)
            big = sw.windows(text, TOK, "port is 2", 4096)
            self.assertTrue(sw.reachable(big, groups), v)

    def test_plain_needs_one_match_widened_to_its_block(self):
        t = synth_triple()
        text = gen.expand(t, "s")
        groups = sw.deciding_groups(t, self.claim("plain"), "s", text)
        self.assertEqual(len(groups), 1)
        (start, end), = groups[0]
        self.assertTrue(text[start:end].startswith("[call c2] $ cat a"))
        self.assertTrue(text[start:end].endswith("port = 2"))

    def test_gate_ceiling(self):
        t = synth_triple()
        text = gen.expand(t, "s")
        groups = sw.deciding_groups(t, self.claim("plain"), "s", text)
        self.assertTrue(sw.gate_reachable(text, groups, ["port"]))
        self.assertFalse(sw.gate_reachable(text, groups, ["absent"]))
        self.assertFalse(sw.gate_reachable(text, groups, []))
        self.assertTrue(sw.gate_reachable(text, [], []))  # NOT MENTIONED: the gate's answer is right

    def test_an_unmatched_group_is_unreachable(self):
        self.assertFalse(sw.reachable([sw.Window(0, 10, "x")], [[]]))


class Verdicts(unittest.TestCase):
    def p(self, e, c):
        return {"entailment": e, "contradiction": c, "neutral": 1 - e - c}

    def test_three_way(self):
        self.assertEqual(rules.three_way([self.p(.9, .05), self.p(.1, .2)], .5)["verdict"], "S")
        self.assertEqual(rules.three_way([self.p(.6, .1), self.p(.0, .7)], .5)["verdict"], "C")  # C beats E
        self.assertEqual(rules.three_way([self.p(.5, .5)], .5)["verdict"], "S")                  # a tie goes to S
        self.assertEqual(rules.three_way([self.p(.3, .4)], .5)["verdict"], "NM")
        self.assertEqual(rules.three_way([], .5)["verdict"], "NM")

    def test_two_way(self):
        self.assertEqual(rules.two_way([rules.sigmoid(-2), rules.sigmoid(1)], .5)["verdict"], "S")
        self.assertEqual(rules.two_way([rules.sigmoid(-2)], .5)["verdict"], "not-S")
        self.assertAlmostEqual(rules.sigmoid(0), .5)

    def test_gated_with_no_anchor_is_not_mentioned_and_counted(self):
        kept, none = rules.gate([self.p(.9, 0)], [False])
        self.assertTrue(none)
        self.assertEqual(rules.three_way(kept, .5)["verdict"], "NM")

    def test_reader_verdicts_follow_the_same_rule(self):
        ws = [rules.reader_probs("S"), rules.reader_probs("C")]
        self.assertEqual(rules.three_way(ws, .5)["verdict"], "S")
        self.assertEqual(rules.three_way([rules.reader_probs(None)], .5)["verdict"], "NM")

    def test_triple_accuracy_modes(self):
        self.assertTrue(rules.triple_correct({"s": "S", "c": "C", "nm": "NM"}, "three-way"))
        self.assertFalse(rules.triple_correct({"s": "S", "c": "NM", "nm": "NM"}, "three-way"))
        self.assertTrue(rules.triple_correct({"s": "S", "c": "NM", "nm": "not-S"}, "binary"))
        self.assertFalse(rules.triple_correct({"s": "S", "c": "S", "nm": "NM"}, "binary"))
        self.assertFalse(rules.triple_correct({"s": "S", "c": "C"}, "three-way"))  # a missing verdict is wrong


class ControlGate(unittest.TestCase):
    def results(self, wrong=()):
        out = []
        for g, gold in (("positive", "S"), ("contradiction", "C"), ("null", "NM")):
            for form in ("short", "long"):
                for i in range(6):
                    v = ("S" if gold != "S" else "NM") if (g, form, i) in wrong else gold
                    out.append({"group": g, "form": form, "gold": gold, "verdict": v})
        return out

    def test_six_of_six_short_five_of_six_long(self):
        all3 = ("positive", "contradiction", "null")
        self.assertTrue(rules.control_gate(self.results(), all3)["passed"])
        self.assertTrue(rules.control_gate(self.results({("positive", "long", 0)}), all3)["passed"])
        self.assertFalse(rules.control_gate(self.results({("positive", "short", 0)}), all3)["passed"])
        self.assertFalse(rules.control_gate(self.results({("null", "long", 0), ("null", "long", 1)}), all3)["passed"])

    def test_a_two_way_checker_answers_positive_and_null(self):
        res = [r if r["gold"] != "NM" else {**r, "verdict": "not-S"} for r in self.results()]
        res = [r if r["group"] != "contradiction" else {**r, "verdict": "S"} for r in res]
        g = rules.control_gate(res, ("positive", "null"))
        self.assertTrue(g["passed"])
        self.assertNotIn("contradiction/short", g["groups"])

    def test_controls_over_the_context_are_not_run_not_wrong(self):
        all3 = ("positive", "contradiction", "null")
        res = [{**r, "verdict": None, "not_run": True} if r["form"] == "long" else r for r in self.results()]
        g = rules.control_gate(res, all3)
        self.assertTrue(g["passed"])
        self.assertIsNone(g["groups"]["positive/long"]["passed"])
        self.assertEqual(g["groups"]["positive/long"]["not_run"], 6)
        # A wrong short answer still fails; a partly run long group needs its share of those run.
        self.assertFalse(rules.control_gate([{**r, "verdict": "C"} if r["form"] == "short" and r["gold"] == "S"
                                             else r for r in res], all3)["passed"])
        part = [{**r, "verdict": None, "not_run": True} if r["form"] == "long" and n % 2 else r
                for n, r in enumerate(self.results())]
        g = rules.control_gate(part, all3)
        self.assertTrue(g["passed"])
        self.assertEqual(g["groups"]["null/long"]["need"], 3)

    def test_a_two_way_model_through_the_three_way_rule_fails(self):
        res = [r if r["group"] != "contradiction" else {**r, "verdict": "NM"} for r in self.results()]
        self.assertFalse(rules.control_gate(res, ("positive", "contradiction", "null"))["passed"])


class Statistics(unittest.TestCase):
    def setUp(self):
        self.shapes = {f"t{i}": f"shape{i // 2}" for i in range(40)}

    def test_bootstrap_is_seeded_and_bounded(self):
        correct = {t: i % 3 != 0 for i, t in enumerate(self.shapes)}
        a = rules.shape_bootstrap(correct, self.shapes, 500, 133)
        b = rules.shape_bootstrap(correct, self.shapes, 500, 133)
        self.assertEqual(a, b)
        self.assertNotEqual(a, rules.shape_bootstrap(correct, self.shapes, 500, 134))
        self.assertTrue(all(0 <= x <= 1 for x in a))
        s = rules.summary(sum(correct.values()) / 40, a)
        self.assertLess(s["ci95"][0], s["value"])
        self.assertGreater(s["ci95"][1], s["value"])

    def test_resampling_is_by_shape(self):
        # Triples of one shape are all right or all wrong: every draw's accuracy
        # is a multiple of 1/20 (20 shapes of 2), never of 1/40 alone.
        correct = {t: int(t[1:]) // 2 % 2 == 0 for t in self.shapes}
        for x in rules.shape_bootstrap(correct, self.shapes, 200, 133):
            self.assertAlmostEqual(x * 20, round(x * 20))

    def test_mde_is_2_8_times_the_paired_se(self):
        x = {t: True for t in self.shapes}
        y = {t: int(t[1:]) % 4 != 0 for t in self.shapes}
        cmp = rules.compare(x, y, self.shapes, 1000, 133)
        self.assertAlmostEqual(cmp["mde"], 2.8 * cmp["se"])
        self.assertAlmostEqual(cmp["value"], 0.25)
        self.assertEqual(cmp["triples"], 40)

    def test_identical_checkers_differ_by_nothing(self):
        x = {t: int(t[1:]) % 3 == 0 for t in self.shapes}
        cmp = rules.compare(x, dict(x), self.shapes, 200, 133)
        self.assertEqual(cmp["value"], 0)
        self.assertEqual(cmp["se"], 0)

    def test_comparison_verdicts(self):
        big = {"value": .3, "ci95": [.1, .5], "mde": .2, "se": .07}
        self.assertEqual(rules.comparison_verdict(big, True, True, 1, .5)["verdict"], rules.MEASURED)
        small = {"value": 0, "ci95": [-.05, .05], "mde": .2, "se": .07}
        self.assertEqual(rules.comparison_verdict(small, True, True, 1, .5)["verdict"], rules.NEGATIVE)
        wide = {"value": 0, "ci95": [-.3, .3], "mde": .2, "se": .15}
        self.assertEqual(rules.comparison_verdict(wide, True, True, 1, .5)["verdict"], rules.INCONCLUSIVE)
        self.assertEqual(rules.comparison_verdict(big, False, True, 1, .5)["verdict"], rules.LIMITED)
        self.assertEqual(rules.comparison_verdict(small, True, True, .4, .5)["verdict"], rules.LIMITED)
        self.assertEqual(rules.comparison_verdict(small, True, True, 1, .5, reader_fails_share=.8)["verdict"],
                         rules.LIMITED)
        self.assertEqual(rules.single_verdict(False)["verdict"], rules.LIMITED)

    def test_auroc(self):
        self.assertEqual(rules.auroc([.9, .8], [.1, .2]), 1.0)
        self.assertEqual(rules.auroc([.1], [.9]), 0.0)
        self.assertEqual(rules.auroc([.5], [.5]), 0.5)
        self.assertIsNone(rules.auroc([], [.5]))

    def test_confusion(self):
        m = rules.confusion([("S", "S"), ("C", "NM"), ("NM", None)])
        self.assertEqual(m["S"]["S"], 1)
        self.assertEqual(m["C"]["NM"], 1)
        self.assertEqual(m["NM"]["none"], 1)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caps = gen.load_captures(gen.FIXTURES / "captures.jsonl", gen.FIXTURES / "captures.lock.json")
        cls.file = sc.load()
        cls.items = sc.build(cls.file, cls.caps)

    def test_the_control_file_keeps_its_rules(self):
        self.assertEqual(sc.check(self.file, self.caps), [])

    def test_eighteen_controls_in_two_forms(self):
        self.assertEqual(len(self.items), 36)
        for g in sc.GROUPS:
            for f in sc.FORMS:
                self.assertEqual(sum(1 for i in self.items if i["group"] == g and i["form"] == f), 6)

    def test_deterministic(self):
        again = sc.build(self.file, self.caps)
        self.assertEqual([i["evidence_sha256"] for i in again], [i["evidence_sha256"] for i in self.items])

    def test_long_forms(self):
        for it in self.items:
            if it["form"] == "long":
                n = canon.approx_tokens(it["evidence"])
                self.assertTrue(0.9 * 32000 <= n <= 1.05 * 32000 + 500, (it["item"], n))
                at = it["evidence"].find(it["needle"])
                self.assertTrue(0.3 < at / len(it["evidence"]) < 0.7, it["item"])
            else:
                self.assertEqual(it["evidence"], it["needle"] + "\n")

    def test_groups_share_their_filler(self):
        by = {}
        for it in self.items:
            if it["form"] == "long":
                by.setdefault(it["control"], set()).add(it["evidence"].replace(it["needle"], "<block>"))
        for cid, bodies in by.items():
            self.assertEqual(len(bodies), 1, cid)

    def test_filler_says_nothing_about_the_claim(self):
        for ctl in self.file["control"]:
            it = next(i for i in self.items if i["item"] == f"ctl:{ctl['id']}:null:long")
            self.assertFalse(sc.mentions(it["evidence"], ctl), ctl["id"])

    def test_the_refusal_scan_passes(self):
        recipes = gen.load_toml(gen.FIXTURES / "recipes.toml")
        allow = [{**a, "capture": "*"} for a in recipes.get("scan", {}).get("allow", [])]
        clock = tuple(recipes.get("scan", {}).get("clock", ["2026-10-01", "2026-10-01"]))
        for it in self.items:
            found = scrub.unallowed(scrub.scan(it["evidence"], scrub.local_identity(), clock), allow, it["item"])
            self.assertEqual(found, [], it["item"])


class Ceilings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = json.loads((REPO / "docs" / "data" / "verification-ceilings.json").read_text())
        cls.cfg = scoring_set.config()

    def test_the_committed_ceilings_match_the_set_and_settings(self):
        lock = canon.read_json(gen.FIXTURES / f"verification-{self.cfg['set']}.lock.json")
        self.assertEqual(self.c["set_sha256"], lock["set_sha256"])
        self.assertEqual(self.c["controls_sha256"], canon.sha256_file(sc.CONTROLS))
        self.assertEqual(self.c["scoring_sha256"], canon.sha256_file(scoring_set.SCORING))

    def test_every_checker_and_triple(self):
        n = len(canon.read_json(gen.FIXTURES / f"verification-{self.cfg['set']}.lock.json")["triples"])
        for m in self.cfg["nli"]:
            ch = self.c["checkers"][m["id"]]
            self.assertEqual(ch["tokenizer_sha256"], m["sha256_tokenizer"])
            self.assertEqual(len(ch["triples"]), n)
            w, g = ch["window_reachable"]["overall"], ch["gated_window_reachable"]["overall"]
            self.assertLessEqual(g["reachable"], w["reachable"])
            self.assertLessEqual(w["reachable"], ch["binary"]["window_reachable"]["overall"]["reachable"])
            self.assertTrue(self.c["controls"][m["id"]]["short_single_window"])
            self.assertTrue(self.c["controls"][m["id"]]["long_block_reachable"])
        self.assertIn("minicheck-deberta-v3-large", self.c["checkers"])
        self.assertIn("foundation-models", self.c["checkers"])

    def test_label_notes(self):
        self.assertEqual(sorted(n["n"] for n in self.c["label_notes"]), [20, 45, 50])

    def test_context_ceiling_from_a_run(self):
        from tasks import scoring_ceilings
        recs = [{"item": "a.1k:s"}, {"item": "a.1k:c"}, {"item": "a.1k:nm"},
                {"item": "b.32k:s", "not_run": True}, {"item": "b.32k:c"}, {"item": "ctl:x:positive:long"}]
        self.assertEqual(scoring_ceilings.context_ceiling(recs), {"triples": 2, "fit": 1, "share": 0.5})


class JudgeAndExport(unittest.TestCase):
    def both(self, content):
        return judge.parse(content), judge.parse_strict(content)

    def test_bare_object(self):
        self.assertEqual(self.both('{"verdict": "supported"}'), (("S", True), ("S", True)))
        self.assertEqual(self.both(' {"verdict":"not_mentioned"} \n'), (("NM", True), ("NM", True)))

    def test_leading_blank_line(self):
        self.assertEqual(self.both('\n\n{"verdict": "contradicted"}'), (("C", True), ("C", True)))

    def test_prose_then_json_is_lenient_only(self):
        content = 'gh stack sync pushed 4 branches to origin.\n\n{"verdict": "not_mentioned"}'
        self.assertEqual(self.both(content), (("NM", True), (None, False)))
        self.assertEqual(self.both('\nThe excerpt says {nothing} here.\n{"verdict": "supported"}\n'),
                         (("S", True), (None, False)))  # braces that aren't JSON are prose

    def test_prose_only(self):
        self.assertEqual(self.both("The claim is supported."), ((None, False), (None, False)))

    def test_two_objects(self):
        content = '{"verdict": "supported"}\n{"verdict": "contradicted"}'
        self.assertEqual(self.both(content), ((None, False), (None, False)))
        self.assertEqual(judge.parse('I first thought {"verdict": "supported"}, but\n{"verdict": "contradicted"}'),
                         (None, False))

    def test_json_then_trailing_prose(self):
        self.assertEqual(self.both('{"verdict": "supported"}\nThat is my answer.'), ((None, False), (None, False)))
        self.assertEqual(self.both('```json\n{"verdict": "supported"}\n```'), ((None, False), (None, False)))

    def test_fenced_answers_fail_but_are_recognised(self):
        for content in ('```json\n{"verdict": "supported"}\n```', '\n```\n{"verdict": "contradicted"}\n```\n',
                        'Here it is:\n```json\n{"verdict": "not_mentioned"}\n```'):
            self.assertEqual(self.both(content), ((None, False), (None, False)), content)
            self.assertTrue(judge.fenced(content), content)
        for content in ('{"verdict": "supported"}', 'Prose.\n{"verdict": "supported"}',
                        '```json\n{"verdict": "maybe"}\n```', '```json\n{"verdict": "supported"}\n```\nDone.',
                        '```\nsupported\n```', "```", None):
            self.assertFalse(judge.fenced(content), content)

    def test_invalid_json_or_verdict(self):
        for content in ('{"verdict": "supported"', "{'verdict': 'supported'}", '{"verdict": "maybe"}',
                        '{"answer": "supported"}', '["supported"]', "", None):
            self.assertEqual(self.both(content), ((None, False), (None, False)), content)

    def test_nested_object_is_one_object(self):
        self.assertEqual(judge.parse('Done.\n{"verdict": "supported", "why": {"line": 3}}'), ("S", True))

    def test_reasoning_is_never_read(self):
        resp = {"response": {"choices": [{"message": {"content": '\n{"verdict": "supported"}',
                                                      "reasoning_content": '{"verdict": "contradicted"}'}}]}}
        self.assertEqual(judge.parse(judge.final_content(resp)), ("S", True))
        resp["response"]["choices"][0]["message"]["content"] = ""
        self.assertEqual(judge.parse(judge.final_content(resp)), (None, False))

    def test_prompt_is_pinned_and_formats(self):
        p = judge.load()
        self.assertEqual(p["sha256"], canon.sha256_file(judge.PROMPT))
        m = judge.messages(p, "window", "a {claim}", "x {y} z\n")
        self.assertIn("a {claim}", m[1]["content"])
        self.assertIn("x {y} z\n>>>", m[1]["content"])
        self.assertIn("one excerpt", m[1]["content"])
        self.assertIn("whole transcript", judge.messages(p, "full", "c", "e")[1]["content"])

    def batch(self):
        items = [{"item": "t1:s", "triple": "t1", "variant": "S", "claim": "uses --force", "evidence": "a\nb --force\n"},
                 {"item": "ctl:k:null:short", "control": "ctl:k:null:short", "claim": "uses --force",
                  "evidence": "a\n"}]
        cuts = {f: {"t1:s": [sw.Window(0, 1, "a"), sw.Window(2, 11, "b --force")],
                    "ctl:k:null:short": [sw.Window(0, 1, "a")]} for f in ("window-512", "window-2048", "chunk-400")}
        prompt = judge.load()
        return ex.build(items, cuts, prompt, {"max_tokens": 2048, "reasoning_effort": "medium"}), prompt

    def test_request_format(self):
        reqs, prompt = self.batch()
        for r in reqs:
            self.assertEqual(set(r) - {"anchored"}, {"id", "form", "prompt_sha256", "messages", "output_mode",
                                                     "max_tokens", "reasoning_effort", "shadows", "serves"})
            self.assertEqual(r["prompt_sha256"], prompt["sha256"])
            self.assertEqual(r["output_mode"], "final-json")
            self.assertEqual(r["max_tokens"], 2048)
            self.assertTrue(r["id"].startswith(r["form"] + "-"))
            self.assertEqual([m["role"] for m in r["messages"]], ["system", "user"])
            for s in r["serves"]:
                self.assertTrue(("triple" in s and s["variant"] in ("S", "C", "NM")) or "control" in s)
                if r["form"] != "full":
                    self.assertIn("window" if r["form"].startswith("window") else "chunk", s)
                    self.assertIn("anchored", r)
        # The control's "a" window and the triple's are the same request.
        a = [r for r in reqs if r["form"] == "window-512" and len(r["serves"]) == 2]
        self.assertEqual(len(a), 1)
        self.assertEqual(sum(1 for r in reqs if r["form"] == "full"), 2)
        anchored = {r["serves"][0].get("window"): r["anchored"] for r in reqs if r["form"] == "window-512"}
        self.assertEqual(anchored, {0: False, 1: True})
        text = json.dumps(reqs)
        for banned in ("http://", "https://", "api_key", "Authorization"):
            self.assertNotIn(banned, text)

    def test_responses_round_trip(self):
        reqs, _ = self.batch()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "responses.jsonl"
            rows = []
            for n, r in enumerate(reqs):
                content = '{"verdict": "supported"}' if n % 2 == 0 else "no json"
                row = {"id": r["id"], "model": "example/model-1", "content": content,
                       "sha256": hashlib.sha256(content.encode()).hexdigest()}
                if n == 1:
                    row["sha256"] = "0" * 64
                rows.append(row)
            p.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
            resp = ex.read_responses(p, reqs)
        self.assertEqual(resp["model"], "example/model-1")
        self.assertEqual(resp["missing"], [reqs[-1]["id"]])
        self.assertEqual(resp["bad_sha"], [reqs[1]["id"]])
        self.assertEqual(resp["by_id"][reqs[0]["id"]],
                         {"verdict": "S", "parsed": True, "strict": True, "fenced": False})
        self.assertEqual(resp["by_id"][reqs[-1]["id"]],
                         {"verdict": None, "parsed": False, "strict": False, "fenced": False})
        wins = ex.reader_windows(reqs, resp, "window-512")
        self.assertEqual([w["index"] for w in wins["t1:s"]], [0, 1])
        self.assertIn("ctl:k:null:short", wins)

    def test_the_endpoints_response_format(self):
        """Lines as the judge endpoint returns them: reasoning apart, timing,
        usage and an `error` field; the model's public id."""
        reqs, prompt = self.batch()

        def line(r, content, **kw):
            return {"id": r["id"], "request_sha256": "0" * 64, "prompt_sha256": prompt["sha256"],
                    "elapsed_s": 1.0, "error": None, "model": "example/model-1", "content": content,
                    "reasoning_content": '{"verdict": "contradicted"}', "finish_reason": "stop",
                    "usage": {"prompt_tokens": 800, "completion_tokens": 150}, **kw}
        rows = [line(reqs[0], '\n\n{"verdict": "not_mentioned"}'),
                line(reqs[1], 'Use it like:\n```json\n{"x": 1}\n```\nso:\n{"verdict": "supported"}'),
                line(reqs[2], '{"verdict": "supported"}', error="timeout"),
                line(reqs[3], '{"verdict": "supported"}', prompt_sha256="f" * 64)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "responses.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            resp = ex.read_responses(p, reqs[:4])
        self.assertEqual(resp["model"], "example/model-1")
        self.assertEqual(resp["by_id"][reqs[0]["id"]],
                         {"verdict": "NM", "parsed": True, "strict": True, "fenced": False})
        # A fence inside the prose, ending with bare JSON: a lenient parse, and not a fenced answer.
        self.assertEqual(resp["by_id"][reqs[1]["id"]],
                         {"verdict": None, "parsed": False, "strict": False, "fenced": False})
        self.assertEqual(resp["errors"], [reqs[2]["id"]])
        self.assertEqual(resp["by_id"][reqs[2]["id"]]["verdict"], None)
        self.assertEqual(resp["bad_prompt"], [reqs[3]["id"]])
        self.assertEqual(resp["missing"], [])

    def test_a_fence_in_the_prose_then_bare_json(self):
        content = 'The evidence shows:\n```\n$ gh stack sync\n```\n\n{"verdict": "supported"}'
        self.assertEqual(judge.parse(content), ("S", True))
        self.assertFalse(judge.fenced(content))

    def test_parse_rates_per_form(self):
        reqs, _ = self.batch()
        answers = ['{"verdict": "supported"}', 'Because.\n{"verdict": "supported"}', "no json",
                   '```json\n{"verdict": "supported"}\n```']
        by_id = {}
        for n, r in enumerate(reqs):
            c = answers[n % 4]
            by_id[r["id"]] = {"verdict": judge.parse(c)[0], "parsed": judge.parse(c)[1],
                              "strict": judge.parse_strict(c)[1], "fenced": judge.fenced(c)}
        rates = ex.parse_rates(reqs, {"by_id": by_id})
        self.assertEqual(set(rates), {"window-512", "window-2048", "chunk-400", "full"})
        for form, r in rates.items():
            ids = [x["id"] for x in reqs if x["form"] == form]
            self.assertEqual(r["requests"], len(ids))
            self.assertGreaterEqual(r["lenient"], r["strict"])
            self.assertEqual(r["lenient"], round(sum(by_id[i]["parsed"] for i in ids) / len(ids), 4))
            self.assertEqual(r["fenced"], round(sum(by_id[i]["fenced"] for i in ids) / len(ids), 4))
        fenced_ids = [i for i, v in by_id.items() if v["fenced"]]
        self.assertTrue(fenced_ids)
        self.assertTrue(all(not by_id[i]["parsed"] and not by_id[i]["strict"] for i in fenced_ids))

    def test_report_carries_reader_parse_rates(self):
        reqs, _ = self.batch()
        by_id = {r["id"]: {"verdict": "S", "parsed": True, "strict": n % 2 == 0} for n, r in enumerate(reqs)}
        runs = report.reader_runs(reqs, {"by_id": by_id, "model": "example/model-1", "missing": [], "bad_sha": []},
                                  0.5)
        self.assertEqual({r["checker"] for r in runs}, {"R512", "R512g", "R2048", "R2048g", "R400", "J"})
        s = report.build(runs, [], {**scoring_set.config(), "resamples": 50}, None)
        self.assertEqual(set(s["reader_parse_rates"]), {"window-512", "window-2048", "chunk-400", "full"})
        md = report.render_md(s)
        self.assertIn("| reader form | requests | parsed (lenient, scored) | parsed (strict) "
                      "| fenced (scored as failures) |", md)

    def test_reestimate_leaves_requests_untouched(self):
        reqs, _ = self.batch()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            est = ex.estimate(reqs, {"assumed_reasoning_tokens": 400}, 800, 40)
            ex.write_batch(root / "full", reqs, {"batch": "full", "estimate": est})
            ex.sha256sums(root)
            before = (root / "full" / "requests.jsonl").read_bytes()
            seconds = {"window-512": 1.4, "window-2048": 2.5, "chunk-400": 1.3, "full": 10.0}
            out = ex.reestimate(root, seconds, "2026-10-21")
            self.assertEqual((root / "full" / "requests.jsonl").read_bytes(), before)
            m = canon.read_json(root / "full" / "manifest.json")
            readme = (root / "full" / "README.md").read_text()
            sums = (root / "SHA256SUMS").read_text()
        self.assertEqual(m["estimate"]["measured"]["on"], "2026-10-21")
        want = sum(seconds[r["form"]] for r in reqs) / 3600
        self.assertAlmostEqual(out["full"]["total"]["hours"], round(want, 2))
        self.assertIn("measured on 2026-10-21", readme)
        self.assertNotIn("prompt tokens/s", readme)
        self.assertIn(f"{canon.sha256_bytes(before)}  full/requests.jsonl", sums)

    def test_estimate(self):
        reqs, _ = self.batch()
        est = ex.estimate(reqs, {"assumed_reasoning_tokens": 400}, 1000, 50)
        self.assertEqual(est["total"]["requests"], len(reqs))
        self.assertEqual(est["total"]["output_tokens"], len(reqs) * 416)
        self.assertGreater(est["total"]["hours"], 0)
        self.assertEqual(est["assumptions"]["concurrency"], 1)


class Report(unittest.TestCase):
    def triples(self, n=16):
        return [{"id": f"t{i}.1k.start.none", "claim": {"shape": f"sh{i // 2}", "mechanism": "plain", "type": "tool"},
                 "config": {"obstacle": "none", "length": "1k"}} for i in range(n)]

    def run_(self, name, fam, mode, right):
        items = []
        for i, t in enumerate(self.triples()):
            ok = right(i)
            for v, g in (("s", "S"), ("c", "C"), ("nm", "NM")):
                ans = g if ok else "NM"
                if mode == "binary":
                    ans = rules.binary(ans)
                items.append({"item": f"{t['id']}:{v}", "verdict": ans, "E": 0.9 if ans == "S" else 0.1,
                              "C": 0.9 if ans == "C" else 0.1})
        for g, gold in (("positive", "S"), ("contradiction", "C"), ("null", "NM")):
            for f in ("short", "long"):
                for k in range(6):
                    items.append({"item": f"ctl:c{k}:{g}:{f}", "verdict": gold if mode == "three-way" else
                                  rules.binary(gold)})
        return {"checker": name, "family": fam, "mode": mode, "items": items}

    def test_scores_and_section(self):
        cfg = {**scoring_set.config(), "resamples": 300}
        runs = [self.run_("A3:m", "A3", "three-way", lambda i: i % 4 != 0),
                self.run_("A3g:m", "A3g", "three-way", lambda i: i % 2 == 0),
                self.run_("A2:z", "A2", "binary", lambda i: True)]
        s = report.build(runs, self.triples(), cfg, None)
        a3 = s["checkers"]["A3:m"]
        self.assertEqual(a3["triple_accuracy"]["value"], 0.75)
        self.assertTrue(a3["gate"]["passed"])
        self.assertEqual(a3["triple_accuracy"]["verdict"], rules.MEASURED)
        self.assertEqual(s["checkers"]["A2:z"]["three_way"], "not applicable")
        self.assertIn("binary_triple_accuracy", s["checkers"]["A2:z"])
        self.assertEqual(s["checkers"]["A2:z"]["binary_triple_accuracy"]["chance"], 1 / 8)
        cmp = [c for c in s["comparisons"] if c["what"] == "gated against ungated"]
        self.assertEqual(len(cmp), 1)
        self.assertAlmostEqual(cmp[0]["mde"], cfg["mde_factor"] * cmp[0]["se"])
        self.assertTrue(s["lead"].startswith("This round is a pilot"))
        self.assertEqual(s["notes"], report.NOTES)
        self.assertIn("define the labels for every item alike", s["notes"][0])
        self.assertIn("only approximately", s["notes"][1])
        md = report.render_md(s)
        for note in report.NOTES:
            self.assertIn(note, md)
        self.assertTrue(md.startswith(report.MD_START))
        spliced = report.splice_md("# Models\n\n" + md + "\nafter\n", md.replace("pilot", "PILOT"))
        self.assertEqual(spliced.count(report.MD_START), 1)
        self.assertIn("PILOT", spliced)
        self.assertTrue(spliced.endswith("after\n"))


if __name__ == "__main__":
    unittest.main()
