"""Correctness checks for the kerb elevation model in geometry.py."""
import unittest
import numpy as np
import geometry as g


class KerbHeightProfile(unittest.TestCase):
    def test_flat_inside_track(self):
        self.assertEqual(g.kerb_height(-5.0), 0.0)
        self.assertEqual(g.kerb_height(0.0), 0.0)

    def test_flat_beyond_curb_on_grass(self):
        self.assertEqual(g.kerb_height(g.KERB_WIDTH_M), 0.0)
        self.assertEqual(g.kerb_height(g.KERB_WIDTH_M + 5.0), 0.0)

    def test_reaches_peak_height_on_the_plateau(self):
        mid = g.KERB_WIDTH_M / 2
        self.assertAlmostEqual(g.kerb_height(mid), g.KERB_HEIGHT_M, places=10)

    def test_ramps_up_linearly_from_the_boundary_line(self):
        half_ramp = g.KERB_RAMP_M / 2
        self.assertAlmostEqual(g.kerb_height(half_ramp), g.KERB_HEIGHT_M / 2, places=10)

    def test_ramps_down_linearly_before_the_outer_edge(self):
        point = g.KERB_WIDTH_M - g.KERB_RAMP_M / 2
        self.assertAlmostEqual(g.kerb_height(point), g.KERB_HEIGHT_M / 2, places=10)

    def test_vectorized_over_numpy_array(self):
        e = np.array([-1.0, 0.0, g.KERB_WIDTH_M / 2, g.KERB_WIDTH_M, 5.0])
        expected = np.array([0.0, 0.0, g.KERB_HEIGHT_M, 0.0, 0.0])
        np.testing.assert_allclose(g.kerb_height(e), expected, atol=1e-10)


class SimulatorContactElevation(unittest.TestCase):
    def test_contact_points_are_3d_matching_kerb_height(self):
        from simulator.generate import make_specs, trajectory
        spec = make_specs(3, 99, 12, 2, 480, 270, angles=1)[0]
        rows, _ = trajectory(spec)
        for row in rows:
            contacts = np.asarray(row['contacts_world'])
            self.assertEqual(contacts.shape, (4, 3))
            excess = g.distance(contacts[:, :2]) - g.HALF
            np.testing.assert_allclose(contacts[:, 2], g.kerb_height(excess), atol=1e-10)

    def test_some_frames_actually_reach_the_kerb(self):
        # Sanity check that this spec/seed isn't a vacuous all-flat case.
        from simulator.generate import make_specs, trajectory
        spec = make_specs(3, 99, 12, 2, 480, 270, angles=1)[0]
        rows, _ = trajectory(spec)
        heights = [np.asarray(r['contacts_world'])[:, 2].max() for r in rows]
        self.assertGreater(max(heights), 0.0)

    def test_margins_stay_purely_horizontal_despite_elevation(self):
        from simulator.generate import make_specs, trajectory
        spec = make_specs(3, 99, 12, 2, 480, 270, angles=1)[0]
        rows, _ = trajectory(spec)
        for row in rows:
            xy = np.asarray(row['contacts_world'])[:, :2]
            excess = g.distance(xy) - g.HALF
            self.assertAlmostEqual(row['point_margin_m'], float(excess.min()), places=8)


class CarMeshWheelElevation(unittest.TestCase):
    TIRE_COLOR = (20, 22, 25)

    def _wheel_vertices(self, mesh, cx, cy):
        tris = [v for v, c in mesh if c == self.TIRE_COLOR
                and np.any((np.abs(v[:, 0] - cx) < .4) & (np.abs(v[:, 1] - cy) < .2))]
        return np.concatenate(tris)

    def test_default_call_matches_all_wheels_flat(self):
        default = g.car_mesh((1, 2, 3))
        explicit_flat = g.car_mesh((1, 2, 3), (0., 0., 0., 0.))
        for (v1, c1), (v2, c2) in zip(default, explicit_flat):
            np.testing.assert_allclose(v1, v2)
            self.assertEqual(c1, c2)

    def test_each_wheel_bottom_sits_at_its_own_elevation(self):
        dz = (0.05, 0.0, 0.02, 0.03)
        mesh = g.car_mesh((1, 2, 3), dz)
        for (cx, cy), expected_dz in zip(g.CONTACT, dz):
            verts = self._wheel_vertices(mesh, cx, cy)
            self.assertAlmostEqual(verts[:, 2].min(), expected_dz, places=6)
            self.assertAlmostEqual(verts[:, 2].max(), expected_dz + .72, places=6)

    def test_chassis_geometry_is_unaffected_by_wheel_elevation(self):
        wheel_colors = {self.TIRE_COLOR, (120, 125, 130), (40, 40, 45)}
        flat = g.car_mesh((1, 2, 3))
        raised = g.car_mesh((1, 2, 3), (0.05, 0.05, 0.05, 0.05))
        chassis_flat = [v for v, c in flat if c not in wheel_colors]
        chassis_raised = [v for v, c in raised if c not in wheel_colors]
        self.assertEqual(len(chassis_flat), len(chassis_raised))
        for v1, v2 in zip(chassis_flat, chassis_raised):
            np.testing.assert_allclose(v1, v2)


if __name__ == '__main__':
    unittest.main()
