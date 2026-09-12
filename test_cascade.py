import json
from pathlib import Path
import tempfile
import unittest
from vmax_live_server import needs_deep_model, enqueue_better


class CascadeTests(unittest.TestCase):
    def test_threshold_and_missing_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for scores,expected in [([.799],True),([.8],False),([.9],False),([.7,.9],False),([],False)]:
                (root/'clip.json').write_text(json.dumps({'observations':[{'confidence':s} for s in scores]}))
                self.assertEqual(needs_deep_model(root,'clip'),expected)
                if not expected:self.assertEqual(enqueue_better(root,'clip'),'skipped')
