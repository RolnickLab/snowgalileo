import unittest
from src.data.earthengine.utils import (
    sample_season_year,
    sample_time_window,
)
from src.data.config import START_YEAR, END_YEAR, NUM_TIMESTEPS, NORTH_HEM_SEASONS


class TestPretrainTemporalSampling(unittest.TestCase):
    def test_temporal_sampling_north_hem(self):
        # Sample each point for each season
        for season in NORTH_HEM_SEASONS.items():
            # randomly choose year to sample from
            season = sample_season_year(season, START_YEAR, END_YEAR)

            SEASON_START_DATE = season[0]
            SEASON_END_DATE = season[1]

            WINDOW_START_DATE, WINDOW_END_DATE = sample_time_window(
                SEASON_START_DATE, SEASON_END_DATE, NUM_TIMESTEPS
            )

            # Check if the sampled time window is within the range
            self.assertTrue(
                SEASON_START_DATE <= WINDOW_START_DATE <= SEASON_END_DATE,
                f"Start date {WINDOW_START_DATE} is out of range {SEASON_START_DATE} to {SEASON_END_DATE}",
            )
            self.assertTrue(
                SEASON_START_DATE <= WINDOW_END_DATE <= SEASON_END_DATE,
                f"End date {WINDOW_END_DATE} is out of range {SEASON_START_DATE} to {SEASON_END_DATE}",
            )
            # Check if the sampled season is within the range
            # except mid season, which can be sampled from the previous year
            if SEASON_START_DATE.startswith("12"):
                self.assertTrue(
                    START_YEAR - 1 <= int(SEASON_START_DATE[:4]) <= END_YEAR + 1,
                    f"Start year {SEASON_START_DATE[:4]} is out of range {START_YEAR} to {END_YEAR + 1}",
                )
            else:
                self.assertTrue(
                    START_YEAR <= int(WINDOW_START_DATE[:4]) <= END_YEAR,
                    f"Start year {WINDOW_START_DATE[:4]} is out of range {START_YEAR} to {END_YEAR}",
                )
                self.assertTrue(
                    START_YEAR <= int(WINDOW_END_DATE[:4]) <= END_YEAR,
                    f"End year {WINDOW_END_DATE[:4]} is out of range {START_YEAR} to {END_YEAR}",
                )

            # test if window size is == NUM_TIMESTEPS
            self.assertEqual(
                (WINDOW_END_DATE - WINDOW_START_DATE).days + 1,
                NUM_TIMESTEPS,
                f"Window size {WINDOW_END_DATE - WINDOW_START_DATE} is not equal to {NUM_TIMESTEPS}",
            )

            # test if year change is handled correctly
            # if the start date is in the end of december, the end date will be in the next year
            if WINDOW_START_DATE.split("-")[1] == 12 and WINDOW_START_DATE.split("-")[2] >= 31 - NUM_TIMESTEPS:
                self.assertTrue(
                    WINDOW_END_DATE[:4] == SEASON_END_DATE.year + 1,
                    f"End year {WINDOW_END_DATE.year} is not equal to {SEASON_END_DATE.year + 1}",
                )