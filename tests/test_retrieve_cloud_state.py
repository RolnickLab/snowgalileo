import unittest

from scripts.cloud_analysis import get_cloud_state_modis


# TODO: catch fill value (= 0)
class TestRetrieveCloudState(unittest.TestCase):
    def test_get_cloud_state_modis(self):
        # Test cases: (input_integer_state, qa_bit_state, expected_output)
        # Output mapping: 0 -> clear, 1 -> cloudy
        test_cases = [
            (45068, "1011000000001100", 1),
            (1033, "0000010000001001", 0),
            (1034, "0000010000001010", 0),
            (8392, "0010000011001000", 0),
        ]

        for integer, expected_bit, expected_state in test_cases:
            with self.subTest(state=integer):
                bit, state = get_cloud_state_modis(integer)
                self.assertEqual(state, expected_state)
                self.assertEqual(bit, expected_bit)  # just to verify bit string matches
