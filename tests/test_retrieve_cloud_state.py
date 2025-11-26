import os
import unittest
from pathlib import Path

from src.data.config import NUM_TIMESTEPS
from src.eval.cloud_eval import CloudMetaDataset

DATA_FOLDER = Path(__file__).parents[1] / "data/eval_tifs"


class TestRetrieveCloudState(unittest.TestCase):
    def test_map_int_to_cloud_states(self):
        # Test cases: (input_integer_state, qa_bit_state, expected_output)
        # Output mapping: 0 -> clear, 1 -> cloudy
        # Using https://blog.ronnyale.com/posts/2023-12-25-modis-bitstring/ for validation

        # TODO: add more test cases
        test_cases = [(1131675649, "0000000000000001", (0, 0, 0))]

        for integer, expected_bit, expected_state in test_cases:
            with self.subTest(state=integer):
                bit, cloud_state, shadow_state, cirrus_state = (
                    CloudMetaDataset.map_int_to_cloud_states(integer)
                )
                self.assertEqual((cloud_state, shadow_state, cirrus_state), expected_state)
                self.assertEqual(bit, expected_bit)  # just to verify bit string matches

    def test_end_to_end(self):
        # checks that the number of clear days matches the inverse of cloud + shadow + cirrus days
        cloud_dataset = CloudMetaDataset(data_folder=DATA_FOLDER)
        filenames = [f for f in os.listdir(DATA_FOLDER)]

        for filename in filenames:
            cloud_state_dict = cloud_dataset.return_cloud_state_from_filename(filename)
            total_cloudy = cloud_state_dict["total_cloudy_days"]
            total_shadow = cloud_state_dict["total_cloud_shadow_days"]
            total_cirrus = cloud_state_dict["total_cirrus_days"]
            total_clear = cloud_state_dict["total_clear_days"]
            total_days = cloud_state_dict["total_days"]

            self.assertEqual(total_days, total_clear + total_cloudy + total_shadow + total_cirrus)
            self.assertEqual(
                total_clear,
                total_days - (total_cloudy + total_shadow + total_cirrus),
            )
            self.assertEqual(total_days, NUM_TIMESTEPS-1)  # last timestep excluded
