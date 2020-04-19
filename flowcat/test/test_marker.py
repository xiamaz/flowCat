import unittest

from flowcat.types import marker as fcm

class TestMarker(unittest.TestCase):

    def test_simple(self):
        """Test marker creation."""
        cases = [
            ("CD53", "CD53"),
            ("CD103 Boron", "CD103-Boron"),
            ("SS INT LIN", "SS"),
            ("FSC", "FS"),
        ]
        for input_data, expected in cases:
            result = fcm.Marker.name_to_marker(input_data)
            self.assertEqual(str(result), expected)

    def test_equality(self):
        """Test equality functions."""
        self.assertEqual(
            fcm.Marker("FS", None, None), fcm.Marker("FS", "Norm", None)
        )
        self.assertIn(
            fcm.Marker("FS", None, None), [fcm.Marker("FS", "Norm", None)]
        )
        self.assertIn(
            fcm.Marker("FS", None, None), ["FS", "SS"]
        )
