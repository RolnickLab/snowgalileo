import unittest
from datetime import datetime

from src.data.config import END_YEAR, NORTH_HEM_SEASONS, NUM_TIMESTEPS, START_YEAR
from src.data.earthengine.utils import (
    sample_season_year,
    sample_time_window,
)


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

            # Example: assuming your date strings are in ISO format 'YYYY-MM-DD'
            SEASON_START_DATE = (
                datetime.strptime(SEASON_START_DATE, "%Y-%m-%d").date()
                if isinstance(SEASON_START_DATE, str)
                else SEASON_START_DATE
            )
            WINDOW_START_DATE = (
                datetime.strptime(WINDOW_START_DATE, "%Y-%m-%d").date()
                if isinstance(WINDOW_START_DATE, str)
                else WINDOW_START_DATE
            )
            WINDOW_END_DATE = (
                datetime.strptime(WINDOW_END_DATE, "%Y-%m-%d").date()
                if isinstance(WINDOW_END_DATE, str)
                else WINDOW_END_DATE
            )
            SEASON_END_DATE = (
                datetime.strptime(SEASON_END_DATE, "%Y-%m-%d").date()
                if isinstance(SEASON_END_DATE, str)
                else SEASON_END_DATE
            )

            # Check if the sampled time window is within the range
            self.assertTrue(
                SEASON_START_DATE <= WINDOW_START_DATE <= SEASON_END_DATE + 1,
                f"Start date {WINDOW_START_DATE} is out of range {SEASON_START_DATE} to {SEASON_END_DATE}",
            )
            self.assertTrue(
                SEASON_START_DATE <= WINDOW_END_DATE <= SEASON_END_DATE + 1,
                f"End date {WINDOW_END_DATE} is out of range {SEASON_START_DATE} to {SEASON_END_DATE}",
            )
            # Check if the sampled season is within the range
            # except mid season, which can be sampled from the previous year
            if SEASON_START_DATE.month == 12:
                self.assertTrue(
                    START_YEAR - 1 <= SEASON_START_DATE.year <= END_YEAR + 1,
                    f"Start year {SEASON_START_DATE.year} is out of range {START_YEAR} to {END_YEAR + 1}",
                )
            else:
                self.assertTrue(
                    START_YEAR <= WINDOW_START_DATE.year <= END_YEAR,
                    f"Start year {WINDOW_START_DATE.year} is out of range {START_YEAR} to {END_YEAR}",
                )
                self.assertTrue(
                    START_YEAR <= WINDOW_END_DATE.year <= END_YEAR,
                    f"End year {WINDOW_END_DATE.year} is out of range {START_YEAR} to {END_YEAR}",
                )

            # test if window size is == NUM_TIMESTEPS
            self.assertEqual(
                (WINDOW_END_DATE - WINDOW_START_DATE).days + 1,
                NUM_TIMESTEPS,
                f"Window size {WINDOW_END_DATE - WINDOW_START_DATE} is not equal to {NUM_TIMESTEPS}",
            )

            # test if year change is handled correctly
            # if the start date is in the end of december, the end date will be in the next year
            if WINDOW_START_DATE.month == 12 and WINDOW_START_DATE.day > 31 - NUM_TIMESTEPS:
                self.assertTrue(
                    WINDOW_END_DATE.year == SEASON_END_DATE.year + 1,
                    f"End year {WINDOW_END_DATE.year} is not equal to {SEASON_END_DATE.year + 1}",
                )
