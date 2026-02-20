import os
import unittest
from pathlib import Path

from src.data.config import NUM_TIMESTEPS
from src.fsc.add_eval.cloud_eval import CloudMetaDataset

DATA_FOLDER = Path(__file__).parents[1] / "data/eval_tifs"


class TestRetrieveCloudState(unittest.TestCase):
    def test_map_int_to_cloud_states(self):
        # test cases without expected bit string (derived with
        # https://gis.stackexchange.com/questions/349371/creating-cloud-free-images-out-of-a-mod09a1-modis-image-in-gee/349401#349401)
        test_cases_without_bit = [
            (1048, (1, 0, 0)),
            (8208, (0, 0, 0)),
            (8210, (1, 0, 0)),
            (1041, (1, 0, 0)),
            (40981, (1, 1, 0)),
            (1049, (1, 0, 0)),
        ]
        for integer, expected_state in test_cases_without_bit:
            with self.subTest(state=integer):
                cloud_state, shadow_state, cirrus_state = CloudMetaDataset.map_int_to_cloud_states(
                    integer
                )
                self.assertEqual((cloud_state, shadow_state, cirrus_state), expected_state)

        # NOTE: the binary test cases are in reversed order (LSB is bit 0, MSB is bit 15)
        # so have to be read from right to left when deriving the expected cloud states
        test_cases_with_bit = [
            (200, "0000000011001000",(0,0,0)),
            (8, "0000000000001000",(0,0,0)),
            (1288, "0000010100001000",(0,0,1)),
            (141, "0000000010001101",(1,1,0)),
            (204, "0000000011001100",(0,1,0)),
            (5384, "0001010100001000",(0,0,1)),
            (40970, "1010000000001010",(1,0,0)),
            (1034, "0000010000001010",(1,0,0)),
            (8392, "0010000011001000",(0,1,0)),
            (40969, "1010000000001001",(1,0,0)),
            (1033, "0000010000001001",(1,0,0)),
        ]
        for integer, bit_string, expected_state in test_cases_with_bit:
            with self.subTest(state=integer):
                cloud_state, shadow_state, cirrus_state = CloudMetaDataset.map_int_to_cloud_states(
                    integer
                )
                self.assertEqual(format(integer, "016b"), bit_string)
                self.assertEqual((cloud_state, shadow_state, cirrus_state), expected_state)

    def test_end_to_end(self):
        # checks that the number of clear days matches the inverse of cloud + shadow + cirrus days
        cloud_dataset = CloudMetaDataset(data_folder=DATA_FOLDER)
        filenames = [f for f in os.listdir(DATA_FOLDER)]

        for filename in filenames:
            cloud_state_dict, _ = cloud_dataset.return_cloud_state_from_filename(filename)
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
            self.assertEqual(total_days, NUM_TIMESTEPS - 1)  # last timestep excluded


if __name__ == "__main__":
    unittest.main()
