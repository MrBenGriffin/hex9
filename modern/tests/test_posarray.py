import unittest
import numpy as np
from examples.addressing import Points


class MockCoordinateSet:
    """Mock Coordinate Set"""
    def __init__(self, name):
        self.name = name
        self.address_formats = {
            'simple': self
        }

    def format(self, arr):
        return f"<Formatted: {arr.tolist()}>"


class TestPosArray(unittest.TestCase):
    """Test PosArray"""
    def test_creation_and_cs_assignment(self):
        arr = Points([1, 2, 3])
        self.assertIsInstance(arr, Points)
        self.assertTrue(np.array_equal(arr, [1, 2, 3]))

        cs = MockCoordinateSet("mock_sys")
        arr.set_domain(cs)
        self.assertEqual(arr._cs, cs)
        self.assertEqual(arr.domain(), "mock_sys")

    def test_formatting_with_valid_spec(self):
        cs = MockCoordinateSet("mock_sys")
        arr = Points([10, 20]).cs(cs)
        formatted = format(arr, "simple")
        self.assertEqual(formatted, "<Formatted: [10, 20]>")

    def test_formatting_with_invalid_spec(self):
        cs = MockCoordinateSet("mock_sys")
        arr = Points([10, 20]).cs(cs)
        with self.assertRaises(ValueError):
            format(arr, "invalid")

    def test_len_and_bool(self):
        empty_arr = Points([])
        non_empty_arr = Points([1])

        self.assertEqual(len(non_empty_arr), 1)
        self.assertEqual(len(empty_arr), 0)

        self.assertTrue(non_empty_arr)
        self.assertFalse(empty_arr)

    def test_repr_contains_cs(self):
        cs = MockCoordinateSet("mock_sys")
        arr = Points([4, 5]).cs(cs)
        r = repr(arr)
        self.assertIn("mock_sys", r)
        self.assertIn("PosArray", r)  # base repr includes 'array'


if __name__ == "__main__":
    unittest.main()
