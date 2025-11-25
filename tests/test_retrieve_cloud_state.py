import unittest

from src.eval.cloud_eval import CloudMetaDataset


# TODO: catch fill value (= 0)
class TestRetrieveCloudState(unittest.TestCase):
    def test_map_int_to_cloud_states(self):
        # TODO: add a bunch of sample inputs and expected outputs

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
                bit, state = CloudMetaDataset.map_int_to_cloud_states(integer)
                self.assertEqual(state, expected_state)
                self.assertEqual(bit, expected_bit)  # just to verify bit string matches

    def test_location(self):
        # TODO: add a test that checks if location is as expected
        raise NotImplementedError("Test not implemented yet")

    def test_end_to_end(self):
        # TODO: checks the number of cloud/clear counts in a sample tif file
        raise NotImplementedError("Test not implemented yet")
