"""Shared task corpora: the verification family (claim-versus-evidence
triples) and, later, the decision family. See tools/tasks/gen_verification.py
for the verification generator and tools/tasks/capture.py for its hermetic
capture stage.

Everything here is stdlib-only Python 3.11+. Tests:

    python3 -m unittest discover -s tools/tasks/tests -t tools
"""
