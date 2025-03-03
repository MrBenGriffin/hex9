from unittest import TestCase
import numpy as np
from ..util import Util


class TestUtil(TestCase):
    def setUp(self):
        self.u = Util()

    def test_ll_xyz(self):
        sp = self.u.sph_rnd(1000)
        ll = self.u.xyz_ll(sp)
        so = self.u.ll_xyz(ll)
        l1 = self.u.xyz_ll(so)
        s2 = self.u.ll_xyz(l1)
        same = np.allclose(s2, sp)
        self.assertTrue(same)


class TestSphRnd(TestUtil):
    def setUp(self):
        TestUtil.setUp(self)
        np.random.seed(42)  # reset for each test.

    def test_empty(self):
        v = self.u.sph_rnd(0)
        self.assertEqual(len(v), 0)

    def test_one(self):
        v = self.u.sph_rnd(1)
        q = np.array([[-0.30524033,  0.30700944,  0.90142861]])
        same = np.allclose(v, q)
        self.assertTrue(same)

    def test_valid(self):
        v = self.u.sph_rnd(1000)
        q = np.linalg.norm(v, axis=-1) - 1.0
        same = np.allclose(q, np.zeros_like(q))
        self.assertTrue(same)


class TestOctRnd(TestUtil):
    def setUp(self):
        TestUtil.setUp(self)
        np.random.seed(42)  # reset for each test.

    def test_empty(self):
        v = self.u.oct_rnd(0)
        self.assertEqual(len(v), 0)

    def test_one(self):
        v = self.u.oct_rnd(1)
        q = np.array([[0.62545988,  -0.04928569, 0.32525443]])
        same = np.allclose(v, q)
        self.assertTrue(same)

    def test_valid(self):
        u = self.u
        v = self.u.oct_rnd(1000)
        av = np.abs(v)
        q = np.sum(av, axis=1) - 1.0
        same = np.allclose(q, np.zeros_like(q))
        self.assertTrue(same)
