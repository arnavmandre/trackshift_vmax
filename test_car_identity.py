"""Two cars must keep separate, stable identities through the export.

Uses a real two-car clip from final_demo/. Detections within a frame arrive in
confidence order, which can flip between frames, so identity must come from the
tracker and survive that order being shuffled.
"""
import json
import random
import unittest
from pathlib import Path

import numpy as np

import vmax_export as export

DEMO = Path(__file__).parent / 'final_demo'
CLIP = '0345af32f4b56d19_a03'  # Incident 1 Clip 4: two cars in every frame


class CarIdentityTests(unittest.TestCase):
    def setUp(self):
        if not (DEMO / f'{CLIP}.json').is_file():
            self.skipTest('final_demo not present')
        self.prediction = json.loads((DEMO / f'{CLIP}.json').read_text())
        camera = json.loads((DEMO / f'{CLIP}.camera.json').read_text())
        self.entry = {'calibration': {'homography': camera['ground_plane_homography']}}
        self.fps = 12

    def evidence(self, shuffle_seed=None):
        raw = export.raw_from_prediction(self.prediction, self.fps)
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            for frame in raw:
                rng.shuffle(frame['detections'])
        return export.build_evidence(raw, self.entry, 0.5, self.fps)

    @staticmethod
    def centres(observations):
        by_car = {}
        for ob in observations:
            if ob['car_id']:
                pts = np.array(list(ob['points'].values()))
                by_car.setdefault(ob['car_id'], []).append((ob['time'], pts.mean(axis=0)))
        return by_car

    def test_two_cars_are_labelled_separately_in_each_frame(self):
        observations, _ = self.evidence()
        self.assertEqual(sorted({o['car_id'] for o in observations if o['car_id']}), ['Car 1', 'Car 2'])
        per_frame = {}
        for ob in observations:
            if ob['car_id']:
                per_frame.setdefault(ob['time'], []).append(ob['car_id'])
        for time, labels in per_frame.items():
            self.assertEqual(len(labels), len(set(labels)), f'two detections share a car label at t={time}')

    def test_each_car_label_follows_one_car(self):
        # At every frame, each car must be closer to where *it* was in the
        # previous frame than to where the *other* car was. A label that swaps
        # cars fails this even when the two cars are close together.
        observations, _ = self.evidence()
        tracks = {car: dict(samples) for car, samples in self.centres(observations).items()}
        car_a, car_b = 'Car 1', 'Car 2'
        a, b = tracks[car_a], tracks[car_b]
        times = sorted(set(a) & set(b))
        checked = 0
        for prev, now in zip(times, times[1:]):
            for mine, other, name in ((a, b, car_a), (b, a, car_b)):
                self.assertLess(np.linalg.norm(mine[now] - mine[prev]), np.linalg.norm(mine[now] - other[prev]),
                                f'{name} at t={now} sits where the other car was')
                checked += 1
        self.assertGreater(checked, 20)

    def test_identity_does_not_depend_on_detection_order(self):
        baseline = {o['time']: {} for o in self.evidence()[0]}
        for ob in self.evidence()[0]:
            baseline[ob['time']][tuple(ob['points']['FL'])] = ob['car_id']
        for seed in range(3):
            for ob in self.evidence(shuffle_seed=seed)[0]:
                self.assertEqual(ob['car_id'], baseline[ob['time']][tuple(ob['points']['FL'])])

    def test_candidates_name_their_car(self):
        _, candidates = self.evidence()
        for c in candidates:
            self.assertIn('car_id', c)


class CrossCameraNamingTests(unittest.TestCase):
    """The same physical car must have the same name in every camera angle of an
    incident, or a steward comparing angles would see labels disagree."""

    INCIDENTS = {'Incident 1': '0345af32f4b56d19', 'Incident 4': '0ccce32ff96d597e'}

    def world_centres(self, clip):
        prediction = json.loads((DEMO / f'{clip}.json').read_text())
        camera = json.loads((DEMO / f'{clip}.camera.json').read_text())
        entry = {'calibration': {'homography': camera['ground_plane_homography']}}
        observations, _ = export.build_evidence(export.raw_from_prediction(prediction, 12), entry, 0.5, 12)
        h_inv = np.linalg.inv(camera['ground_plane_homography'])
        centres = {}
        for ob in observations:
            if ob['car_id']:
                uv = np.array(list(ob['points'].values()))
                centres[(ob['car_id'], ob['time'])] = export.e.transform(uv, h_inv).mean(axis=0)
        return centres

    def test_car_names_agree_across_camera_angles(self):
        for incident, family in self.INCIDENTS.items():
            clips = [f'{family}_a0{i}' for i in range(4)]
            if not all((DEMO / f'{c}.json').is_file() for c in clips):
                self.skipTest('final_demo not present')
            views = [self.world_centres(c) for c in clips]
            compared = 0
            for a, b in zip(views, views[1:]):
                for (car, t), pos in a.items():
                    same = b.get((car, t))
                    other = b.get(('Car 2' if car == 'Car 1' else 'Car 1', t))
                    if same is None or other is None:
                        continue
                    # Same name must be the same car: nearer than the other car.
                    self.assertLess(np.linalg.norm(pos - same), np.linalg.norm(pos - other),
                                    f'{incident}: {car} at t={t} is a different car in another angle')
                    compared += 1
            self.assertGreater(compared, 10, f'{incident}: too few shared frames to compare')


if __name__ == '__main__':
    unittest.main()
