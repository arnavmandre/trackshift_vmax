import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'spam/vmax'))
import generate_extra as extra
import geometry as g
import tyre_boundaries as tb
from evaluate_extra import decision_summary


class ExpandedSuiteTests(unittest.TestCase):
    def test_reviews_cannot_inflate_accuracy(self):
        result=decision_summary([{'truth':False,'prediction':False},
            {'truth':True,'prediction':None},{'truth':True,'prediction':True},
            {'truth':False,'prediction':True}])
        self.assertEqual(result['accuracy_percent_reviews_count_as_misses'],50)
        self.assertEqual(result['decision_coverage_percent'],75)
        self.assertAlmostEqual(result['accuracy_percent_decided_only'],200/3)
        self.assertEqual(result['violation_recall_percent'],50)

    def test_unknown_truth_not_scored_as_correct(self):
        result=decision_summary([{'truth':None,'prediction':None}])
        self.assertEqual(result['known_truth_frames'],0)
        self.assertIsNone(result['accuracy_percent_reviews_count_as_misses'])

    def test_all_review_is_zero_accuracy(self):
        result=decision_summary([{'truth':True,'prediction':None}])
        self.assertEqual(result['accuracy_percent_reviews_count_as_misses'],0)
        self.assertIsNone(result['accuracy_percent_decided_only'])

    def test_generation_deterministic_and_restores_geometry(self):
        before=g.S.copy()
        a,travel=extra.make_document(extra.CASES[2])
        b,_=extra.make_document(extra.CASES[2])
        self.assertEqual(a,b)
        np.testing.assert_array_equal(g.S,before)
        self.assertEqual(len(a['frames']),96)
        self.assertTrue(np.all(np.diff(travel)>0))

    def test_new_cases_cover_legal_shallow_and_short_excursions(self):
        counts={};positions=[]
        for case in extra.CASES:
            doc,_=extra.make_document(case)
            rows=tb.annotate_document(doc)['frames']
            counts[case['name']]=sum(r['is_violation'] is True for r in rows)
            positions.append(np.array([r['world_position'] for r in rows]))
            for row in rows:
                if row['is_violation'] is None:
                    lo,hi=row['boundary_margin_m_interval']
                    self.assertLessEqual(lo,0)
                    self.assertGreater(hi,0)
        self.assertEqual(counts['legal_close'],0)
        self.assertEqual(counts['legal_edge'],0)
        self.assertGreater(counts['shallow_excursion'],0)
        self.assertLess(counts['short_excursion'],counts['sustained_excursion'])
        for i in range(len(positions)):
            for j in range(i):self.assertFalse(np.allclose(positions[i],positions[j]))


if __name__=='__main__':unittest.main()
