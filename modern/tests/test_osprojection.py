from unittest import TestCase
import numpy as np
from ..osprojection import OSProjection
from ..util import Util


class TestOSProjection(TestCase):
    def setUp(self):
        self.os = OSProjection()
        self.u = Util()  # Validate these first!

    def s_valid(self):
        sp = self.u.sph_rnd(1000)
        self.assertTrue(self.os.s_valid(sp))

    def o_valid(self):
        sp = self.u.oct_rnd(1000)
        self.assertTrue(self.os.o_valid(sp))

