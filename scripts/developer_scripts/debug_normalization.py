from pathlib import Path

import numpy as np

from src.data.config import DATA_FOLDER, NORMALIZATION_DICT_FILENAME
from src.data.dataset import Dataset, Normalizer
from src.utils import config_dir

if __name__ == "__main__":
    dataset = Dataset(
        data_folder=DATA_FOLDER / "tifs_all_bands",
        download=False,
        h5py_folder=Path("data/h5pys_pretrain"),
        h5pys_only=True,
    )

    normalizing_dict = dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    print(normalizing_dict, flush=True)
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
    dataset.normalizer = normalizer

    stats = []

    # create a csv that stores the min and max values for each channel
    for i in range(len(dataset)):
        (
            s_t_h_x,
            s_t_m_x,
            s_t_l_x,
            sp_x,
            t_x,
            st_x,
            months,
            valid_data_mask_s_t_h,
            valid_data_mask_s_t_m,
            valid_data_mask_s_t_l,
            valid_data_mask_sp,
            valid_data_mask_t,
            valid_data_mask_st,
        ) = dataset[i]

        s_t_h_x_c0_valid = s_t_h_x[..., 0][valid_data_mask_s_t_h[..., 0].astype(bool)]
        s_t_h_x_c1_valid = s_t_h_x[..., 1][valid_data_mask_s_t_h[..., 1].astype(bool)]
        s_t_h_x_c2_valid = s_t_h_x[..., 2][valid_data_mask_s_t_h[..., 2].astype(bool)]
        s_t_h_x_c3_valid = s_t_h_x[..., 3][valid_data_mask_s_t_h[..., 3].astype(bool)]
        s_t_h_x_c4_valid = s_t_h_x[..., 4][valid_data_mask_s_t_h[..., 4].astype(bool)]
        s_t_h_x_c5_valid = s_t_h_x[..., 5][valid_data_mask_s_t_h[..., 5].astype(bool)]
        s_t_h_x_c6_valid = s_t_h_x[..., 6][valid_data_mask_s_t_h[..., 6].astype(bool)]
        s_t_h_x_c7_valid = s_t_h_x[..., 7][valid_data_mask_s_t_h[..., 7].astype(bool)]
        s_t_h_x_c8_valid = s_t_h_x[..., 8][valid_data_mask_s_t_h[..., 8].astype(bool)]
        s_t_h_x_c9_valid = s_t_h_x[..., 9][valid_data_mask_s_t_h[..., 9].astype(bool)]
        s_t_h_x_c10_valid = s_t_h_x[..., 10][valid_data_mask_s_t_h[..., 10].astype(bool)]
        s_t_h_x_c11_valid = s_t_h_x[..., 11][valid_data_mask_s_t_h[..., 11].astype(bool)]
        s_t_h_x_c12_valid = s_t_h_x[..., 12][valid_data_mask_s_t_h[..., 12].astype(bool)]
        s_t_h_x_c13_valid = s_t_h_x[..., 13][valid_data_mask_s_t_h[..., 13].astype(bool)]
        s_t_h_x_c14_valid = s_t_h_x[..., 14][valid_data_mask_s_t_h[..., 14].astype(bool)]

        s_t_m_x_c0_valid = s_t_m_x[..., 0][valid_data_mask_s_t_m[..., 0].astype(bool)]
        s_t_m_x_c1_valid = s_t_m_x[..., 1][valid_data_mask_s_t_m[..., 1].astype(bool)]

        s_t_l_x_c0_valid = s_t_l_x[..., 0][valid_data_mask_s_t_l[..., 0].astype(bool)]
        s_t_l_x_c1_valid = s_t_l_x[..., 1][valid_data_mask_s_t_l[..., 1].astype(bool)]
        s_t_l_x_c2_valid = s_t_l_x[..., 2][valid_data_mask_s_t_l[..., 2].astype(bool)]
        s_t_l_x_c3_valid = s_t_l_x[..., 3][valid_data_mask_s_t_l[..., 3].astype(bool)]
        s_t_l_x_c4_valid = s_t_l_x[..., 4][valid_data_mask_s_t_l[..., 4].astype(bool)]
        s_t_l_x_c5_valid = s_t_l_x[..., 5][valid_data_mask_s_t_l[..., 5].astype(bool)]
        s_t_l_x_c6_valid = s_t_l_x[..., 6][valid_data_mask_s_t_l[..., 6].astype(bool)]
        s_t_l_x_c7_valid = s_t_l_x[..., 7][valid_data_mask_s_t_l[..., 7].astype(bool)]
        s_t_l_x_c8_valid = s_t_l_x[..., 8][valid_data_mask_s_t_l[..., 8].astype(bool)]
        s_t_l_x_c9_valid = s_t_l_x[..., 9][valid_data_mask_s_t_l[..., 9].astype(bool)]
        s_t_l_x_c10_valid = s_t_l_x[..., 10][valid_data_mask_s_t_l[..., 10].astype(bool)]

        sp_x_c0_valid = sp_x[..., 0][valid_data_mask_sp[..., 0].astype(bool)]
        sp_x_c1_valid = sp_x[..., 1][valid_data_mask_sp[..., 1].astype(bool)]
        sp_x_c2_valid = sp_x[..., 2][valid_data_mask_sp[..., 2].astype(bool)]
        sp_x_c3_valid = sp_x[..., 3][valid_data_mask_sp[..., 3].astype(bool)]

        t_x_c0_valid = t_x[..., 0][valid_data_mask_t[..., 0].astype(bool)]
        t_x_c1_valid = t_x[..., 1][valid_data_mask_t[..., 1].astype(bool)]
        t_x_c2_valid = t_x[..., 2][valid_data_mask_t[..., 2].astype(bool)]
        t_x_c3_valid = t_x[..., 3][valid_data_mask_t[..., 3].astype(bool)]
        t_x_c4_valid = t_x[..., 4][valid_data_mask_t[..., 4].astype(bool)]
        t_x_c5_valid = t_x[..., 5][valid_data_mask_t[..., 5].astype(bool)]
        t_x_c6_valid = t_x[..., 6][valid_data_mask_t[..., 6].astype(bool)]
        t_x_c7_valid = t_x[..., 7][valid_data_mask_t[..., 7].astype(bool)]
        t_x_c8_valid = t_x[..., 8][valid_data_mask_t[..., 8].astype(bool)]

        st_x_c0_valid = st_x[..., 0][valid_data_mask_st[..., 0].astype(bool)]
        st_x_c1_valid = st_x[..., 1][valid_data_mask_st[..., 1].astype(bool)]
        st_x_c2_valid = st_x[..., 2][valid_data_mask_st[..., 2].astype(bool)]

        # collect per-channel values to plot distributions later
        stats.append(
            {
                "tif": i,
                "s_t_h_x_c1_mean": s_t_h_x_c1_valid.mean()
                if len(s_t_h_x_c1_valid) > 0
                else np.nan,
                "s_t_h_x_c2_mean": s_t_h_x_c2_valid.mean()
                if len(s_t_h_x_c2_valid) > 0
                else np.nan,
                "s_t_h_x_c3_mean": s_t_h_x_c3_valid.mean()
                if len(s_t_h_x_c3_valid) > 0
                else np.nan,
                "s_t_h_x_c4_mean": s_t_h_x_c4_valid.mean()
                if len(s_t_h_x_c4_valid) > 0
                else np.nan,
                "s_t_h_x_c5_mean": s_t_h_x_c5_valid.mean()
                if len(s_t_h_x_c5_valid) > 0
                else np.nan,
                "s_t_h_x_c6_mean": s_t_h_x_c6_valid.mean()
                if len(s_t_h_x_c6_valid) > 0
                else np.nan,
                "s_t_h_x_c7_mean": s_t_h_x_c7_valid.mean()
                if len(s_t_h_x_c7_valid) > 0
                else np.nan,
                "s_t_h_x_c8_mean": s_t_h_x_c8_valid.mean()
                if len(s_t_h_x_c8_valid) > 0
                else np.nan,
                "s_t_h_x_c9_mean": s_t_h_x_c9_valid.mean()
                if len(s_t_h_x_c9_valid) > 0
                else np.nan,
                "s_t_h_x_c10_mean": s_t_h_x_c10_valid.mean()
                if len(s_t_h_x_c10_valid) > 0
                else np.nan,
                "s_t_h_x_c11_mean": s_t_h_x_c11_valid.mean()
                if len(s_t_h_x_c11_valid) > 0
                else np.nan,
                "s_t_h_x_c12_mean": s_t_h_x_c12_valid.mean()
                if len(s_t_h_x_c12_valid) > 0
                else np.nan,
                "s_t_h_x_c13_mean": s_t_h_x_c13_valid.mean()
                if len(s_t_h_x_c13_valid) > 0
                else np.nan,
                "s_t_h_x_c14_mean": s_t_h_x_c14_valid.mean()
                if len(s_t_h_x_c14_valid) > 0
                else np.nan,
                "s_t_m_x_c0_mean": s_t_m_x_c0_valid.mean()
                if len(s_t_m_x_c0_valid) > 0
                else np.nan,
                "s_t_m_x_c1_mean": s_t_m_x_c1_valid.mean()
                if len(s_t_m_x_c1_valid) > 0
                else np.nan,
                "s_t_l_x_c0_mean": s_t_l_x_c0_valid.mean()
                if len(s_t_l_x_c0_valid) > 0
                else np.nan,
                "s_t_l_x_c1_mean": s_t_l_x_c1_valid.mean()
                if len(s_t_l_x_c1_valid) > 0
                else np.nan,
                "s_t_l_x_c2_mean": s_t_l_x_c2_valid.mean()
                if len(s_t_l_x_c2_valid) > 0
                else np.nan,
                "s_t_l_x_c3_mean": s_t_l_x_c3_valid.mean()
                if len(s_t_l_x_c3_valid) > 0
                else np.nan,
                "s_t_l_x_c4_mean": s_t_l_x_c4_valid.mean()
                if len(s_t_l_x_c4_valid) > 0
                else np.nan,
                "s_t_l_x_c5_mean": s_t_l_x_c5_valid.mean()
                if len(s_t_l_x_c5_valid) > 0
                else np.nan,
                "s_t_l_x_c6_mean": s_t_l_x_c6_valid.mean()
                if len(s_t_l_x_c6_valid) > 0
                else np.nan,
                "s_t_l_x_c7_mean": s_t_l_x_c7_valid.mean()
                if len(s_t_l_x_c7_valid) > 0
                else np.nan,
                "s_t_l_x_c8_mean": s_t_l_x_c8_valid.mean()
                if len(s_t_l_x_c8_valid) > 0
                else np.nan,
                "s_t_l_x_c9_mean": s_t_l_x_c9_valid.mean()
                if len(s_t_l_x_c9_valid) > 0
                else np.nan,
                "s_t_l_x_c10_mean": s_t_l_x_c10_valid.mean()
                if len(s_t_l_x_c10_valid) > 0
                else np.nan,
                "sp_x_c0_mean": sp_x_c0_valid.mean() if len(sp_x_c0_valid) > 0 else np.nan,
                "sp_x_c1_mean": sp_x_c1_valid.mean() if len(sp_x_c1_valid) > 0 else np.nan,
                "sp_x_c2_mean": sp_x_c2_valid.mean() if len(sp_x_c2_valid) > 0 else np.nan,
                "sp_x_c3_mean": sp_x_c3_valid.mean() if len(sp_x_c3_valid) > 0 else np.nan,
                "t_x_c0_mean": t_x_c0_valid.mean() if len(t_x_c0_valid) > 0 else np.nan,
                "t_x_c1_mean": t_x_c1_valid.mean() if len(t_x_c1_valid) > 0 else np.nan,
                "t_x_c2_mean": t_x_c2_valid.mean() if len(t_x_c2_valid) > 0 else np.nan,
                "t_x_c3_mean": t_x_c3_valid.mean() if len(t_x_c3_valid) > 0 else np.nan,
                "t_x_c4_mean": t_x_c4_valid.mean() if len(t_x_c4_valid) > 0 else np.nan,
                "t_x_c5_mean": t_x_c5_valid.mean() if len(t_x_c5_valid) > 0 else np.nan,
                "t_x_c6_mean": t_x_c6_valid.mean() if len(t_x_c6_valid) > 0 else np.nan,
                "t_x_c7_mean": t_x_c7_valid.mean() if len(t_x_c7_valid) > 0 else np.nan,
                "t_x_c8_mean": t_x_c8_valid.mean() if len(t_x_c8_valid) > 0 else np.nan,
                "st_x_c0_mean": st_x_c0_valid.mean() if len(st_x_c0_valid) > 0 else np.nan,
                "st_x_c1_mean": st_x_c1_valid.mean() if len(st_x_c1_valid) > 0 else np.nan,
                "st_x_c2_mean": st_x_c2_valid.mean() if len(st_x_c2_valid) > 0 else np.nan,
                "s_t_h_x_c1_std": s_t_h_x_c1_valid.std() if len(s_t_h_x_c1_valid) > 0 else np.nan,
                "s_t_h_x_c2_std": s_t_h_x_c2_valid.std() if len(s_t_h_x_c2_valid) > 0 else np.nan,
                "s_t_h_x_c3_std": s_t_h_x_c3_valid.std() if len(s_t_h_x_c3_valid) > 0 else np.nan,
                "s_t_h_x_c4_std": s_t_h_x_c4_valid.std() if len(s_t_h_x_c4_valid) > 0 else np.nan,
                "s_t_h_x_c5_std": s_t_h_x_c5_valid.std() if len(s_t_h_x_c5_valid) > 0 else np.nan,
                "s_t_h_x_c6_std": s_t_h_x_c6_valid.std() if len(s_t_h_x_c6_valid) > 0 else np.nan,
                "s_t_h_x_c7_std": s_t_h_x_c7_valid.std() if len(s_t_h_x_c7_valid) > 0 else np.nan,
                "s_t_h_x_c8_std": s_t_h_x_c8_valid.std() if len(s_t_h_x_c8_valid) > 0 else np.nan,
                "s_t_h_x_c9_std": s_t_h_x_c9_valid.std() if len(s_t_h_x_c9_valid) > 0 else np.nan,
                "s_t_h_x_c10_std": s_t_h_x_c10_valid.std()
                if len(s_t_h_x_c10_valid) > 0
                else np.nan,
                "s_t_h_x_c11_std": s_t_h_x_c11_valid.std()
                if len(s_t_h_x_c11_valid) > 0
                else np.nan,
                "s_t_h_x_c12_std": s_t_h_x_c12_valid.std()
                if len(s_t_h_x_c12_valid) > 0
                else np.nan,
                "s_t_h_x_c13_std": s_t_h_x_c13_valid.std()
                if len(s_t_h_x_c13_valid) > 0
                else np.nan,
                "s_t_h_x_c14_std": s_t_h_x_c14_valid.std()
                if len(s_t_h_x_c14_valid) > 0
                else np.nan,
                "s_t_m_x_c0_std": s_t_m_x_c0_valid.std() if len(s_t_m_x_c0_valid) > 0 else np.nan,
                "s_t_m_x_c1_std": s_t_m_x_c1_valid.std() if len(s_t_m_x_c1_valid) > 0 else np.nan,
                "s_t_l_x_c0_std": s_t_l_x_c0_valid.std() if len(s_t_l_x_c0_valid) > 0 else np.nan,
                "s_t_l_x_c1_std": s_t_l_x_c1_valid.std() if len(s_t_l_x_c1_valid) > 0 else np.nan,
                "s_t_l_x_c2_std": s_t_l_x_c2_valid.std() if len(s_t_l_x_c2_valid) > 0 else np.nan,
                "s_t_l_x_c3_std": s_t_l_x_c3_valid.std() if len(s_t_l_x_c3_valid) > 0 else np.nan,
                "s_t_l_x_c4_std": s_t_l_x_c4_valid.std() if len(s_t_l_x_c4_valid) > 0 else np.nan,
                "s_t_l_x_c5_std": s_t_l_x_c5_valid.std() if len(s_t_l_x_c5_valid) > 0 else np.nan,
                "s_t_l_x_c6_std": s_t_l_x_c6_valid.std() if len(s_t_l_x_c6_valid) > 0 else np.nan,
                "s_t_l_x_c7_std": s_t_l_x_c7_valid.std() if len(s_t_l_x_c7_valid) > 0 else np.nan,
                "s_t_l_x_c8_std": s_t_l_x_c8_valid.std() if len(s_t_l_x_c8_valid) > 0 else np.nan,
                "s_t_l_x_c9_std": s_t_l_x_c9_valid.std() if len(s_t_l_x_c9_valid) > 0 else np.nan,
                "s_t_l_x_c10_std": s_t_l_x_c10_valid.std()
                if len(s_t_l_x_c10_valid) > 0
                else np.nan,
                "sp_x_c0_std": sp_x_c0_valid.std() if len(sp_x_c0_valid) > 0 else np.nan,
                "sp_x_c1_std": sp_x_c1_valid.std() if len(sp_x_c1_valid) > 0 else np.nan,
                "sp_x_c2_std": sp_x_c2_valid.std() if len(sp_x_c2_valid) > 0 else np.nan,
                "sp_x_c3_std": sp_x_c3_valid.std() if len(sp_x_c3_valid) > 0 else np.nan,
                "t_x_c0_std": t_x_c0_valid.std() if len(t_x_c0_valid) > 0 else np.nan,
                "t_x_c1_std": t_x_c1_valid.std() if len(t_x_c1_valid) > 0 else np.nan,
                "t_x_c2_std": t_x_c2_valid.std() if len(t_x_c2_valid) > 0 else np.nan,
                "t_x_c3_std": t_x_c3_valid.std() if len(t_x_c3_valid) > 0 else np.nan,
                "t_x_c4_std": t_x_c4_valid.std() if len(t_x_c4_valid) > 0 else np.nan,
                "t_x_c5_std": t_x_c5_valid.std() if len(t_x_c5_valid) > 0 else np.nan,
                "t_x_c6_std": t_x_c6_valid.std() if len(t_x_c6_valid) > 0 else np.nan,
                "t_x_c7_std": t_x_c7_valid.std() if len(t_x_c7_valid) > 0 else np.nan,
                "t_x_c8_std": t_x_c8_valid.std() if len(t_x_c8_valid) > 0 else np.nan,
                "st_x_c0_std": st_x_c0_valid.std() if len(st_x_c0_valid) > 0 else np.nan,
                "st_x_c1_std": st_x_c1_valid.std() if len(st_x_c1_valid) > 0 else np.nan,
                "st_x_c2_std": st_x_c2_valid.std() if len(st_x_c2_valid) > 0 else np.nan,
            }
        )
        """
        stats.append(
            {
                "tif": i,
                "s_t_h_x_c1_min": s_t_h_x_c1_valid.min() if len(s_t_h_x_c1_valid) > 0 else np.nan,
                "s_t_h_x_c2_min": s_t_h_x_c2_valid.min() if len(s_t_h_x_c2_valid) > 0 else np.nan,
                "s_t_h_x_c3_min": s_t_h_x_c3_valid.min() if len(s_t_h_x_c3_valid) > 0 else np.nan,
                "s_t_h_x_c4_min": s_t_h_x_c4_valid.min() if len(s_t_h_x_c4_valid) > 0 else np.nan,
                "s_t_h_x_c5_min": s_t_h_x_c5_valid.min() if len(s_t_h_x_c5_valid) > 0 else np.nan,
                "s_t_h_x_c6_min": s_t_h_x_c6_valid.min() if len(s_t_h_x_c6_valid) > 0 else np.nan,
                "s_t_h_x_c7_min": s_t_h_x_c7_valid.min() if len(s_t_h_x_c7_valid) > 0 else np.nan,
                "s_t_h_x_c8_min": s_t_h_x_c8_valid.min() if len(s_t_h_x_c8_valid) > 0 else np.nan,
                "s_t_h_x_c9_min": s_t_h_x_c9_valid.min() if len(s_t_h_x_c9_valid) > 0 else np.nan,
                "s_t_h_x_c10_min": s_t_h_x_c10_valid.min() if len(s_t_h_x_c10_valid) > 0 else np.nan,
                "s_t_h_x_c11_min": s_t_h_x_c11_valid.min() if len(s_t_h_x_c11_valid) > 0 else np.nan,
                "s_t_h_x_c12_min": s_t_h_x_c12_valid.min() if len(s_t_h_x_c12_valid) > 0 else np.nan,
                "s_t_h_x_c13_min": s_t_h_x_c13_valid.min() if len(s_t_h_x_c13_valid) > 0 else np.nan,
                "s_t_h_x_c14_min": s_t_h_x_c14_valid.min() if len(s_t_h_x_c14_valid) > 0 else np.nan,
                "s_t_m_x_c0_min": s_t_m_x_c0_valid.min() if len(s_t_m_x_c0_valid) > 0 else np.nan,
                "s_t_m_x_c1_min": s_t_m_x_c1_valid.min() if len(s_t_m_x_c1_valid) > 0 else np.nan,
                "s_t_l_x_c0_min": s_t_l_x_c0_valid.min() if len(s_t_l_x_c0_valid) > 0 else np.nan,
                "s_t_l_x_c1_min": s_t_l_x_c1_valid.min() if len(s_t_l_x_c1_valid) > 0 else np.nan,
                "s_t_l_x_c2_min": s_t_l_x_c2_valid.min() if len(s_t_l_x_c2_valid) > 0 else np.nan,
                "s_t_l_x_c3_min": s_t_l_x_c3_valid.min() if len(s_t_l_x_c3_valid) > 0 else np.nan,
                "s_t_l_x_c4_min": s_t_l_x_c4_valid.min() if len(s_t_l_x_c4_valid) > 0 else np.nan,
                "s_t_l_x_c5_min": s_t_l_x_c5_valid.min() if len(s_t_l_x_c5_valid) > 0 else np.nan,
                "s_t_l_x_c6_min": s_t_l_x_c6_valid.min() if len(s_t_l_x_c6_valid) > 0 else np.nan,
                "s_t_l_x_c7_min": s_t_l_x_c7_valid.min() if len(s_t_l_x_c7_valid) > 0 else np.nan,
                "s_t_l_x_c8_min": s_t_l_x_c8_valid.min() if len(s_t_l_x_c8_valid) > 0 else np.nan,
                "s_t_l_x_c9_min": s_t_l_x_c9_valid.min() if len(s_t_l_x_c9_valid) > 0 else np.nan,
                "s_t_l_x_c10_min": s_t_l_x_c10_valid.min() if len(s_t_l_x_c10_valid) > 0 else np.nan,
                "sp_x_c0_min": sp_x_c0_valid.min() if len(sp_x_c0_valid) > 0 else np.nan,
                "sp_x_c1_min": sp_x_c1_valid.min() if len(sp_x_c1_valid) > 0 else np.nan,
                "sp_x_c2_min": sp_x_c2_valid.min() if len(sp_x_c2_valid) > 0 else np.nan,
                "sp_x_c3_min": sp_x_c3_valid.min() if len(sp_x_c3_valid) > 0 else np.nan,
                "t_x_c0_min": t_x_c0_valid.min() if len(t_x_c0_valid) > 0 else np.nan,
                "t_x_c1_min": t_x_c1_valid.min() if len(t_x_c1_valid) > 0 else np.nan,
                "t_x_c2_min": t_x_c2_valid.min() if len(t_x_c2_valid) > 0 else np.nan,
                "t_x_c3_min": t_x_c3_valid.min() if len(t_x_c3_valid) > 0 else np.nan,
                "t_x_c4_min": t_x_c4_valid.min() if len(t_x_c4_valid) > 0 else np.nan,
                "t_x_c5_min": t_x_c5_valid.min() if len(t_x_c5_valid) > 0 else np.nan,
                "t_x_c6_min": t_x_c6_valid.min() if len(t_x_c6_valid) > 0 else np.nan,
                "t_x_c7_min": t_x_c7_valid.min() if len(t_x_c7_valid) > 0 else np.nan,
                "t_x_c8_min": t_x_c8_valid.min() if len(t_x_c8_valid) > 0 else np.nan,
                "st_x_c0_min": st_x_c0_valid.min() if len(st_x_c0_valid) > 0 else np.nan,
                "st_x_c1_min": st_x_c1_valid.min() if len(st_x_c1_valid) > 0 else np.nan,
                "st_x_c2_min": st_x_c2_valid.min() if len(st_x_c2_valid) > 0 else np.nan,
                "s_t_h_x_c1_max": s_t_h_x_c1_valid.max() if len(s_t_h_x_c1_valid) > 0 else np.nan,
                "s_t_h_x_c2_max": s_t_h_x_c2_valid.max() if len(s_t_h_x_c2_valid) > 0 else np.nan,
                "s_t_h_x_c3_max": s_t_h_x_c3_valid.max() if len(s_t_h_x_c3_valid) > 0 else np.nan,
                "s_t_h_x_c4_max": s_t_h_x_c4_valid.max() if len(s_t_h_x_c4_valid) > 0 else np.nan,
                "s_t_h_x_c5_max": s_t_h_x_c5_valid.max() if len(s_t_h_x_c5_valid) > 0 else np.nan,
                "s_t_h_x_c6_max": s_t_h_x_c6_valid.max() if len(s_t_h_x_c6_valid) > 0 else np.nan,
                "s_t_h_x_c7_max": s_t_h_x_c7_valid.max() if len(s_t_h_x_c7_valid) > 0 else np.nan,
                "s_t_h_x_c8_max": s_t_h_x_c8_valid.max() if len(s_t_h_x_c8_valid) > 0 else np.nan,
                "s_t_h_x_c9_max": s_t_h_x_c9_valid.max() if len(s_t_h_x_c9_valid) > 0 else np.nan,
                "s_t_h_x_c10_max": s_t_h_x_c10_valid.max() if len(s_t_h_x_c10_valid) > 0 else np.nan,
                "s_t_h_x_c11_max": s_t_h_x_c11_valid.max() if len(s_t_h_x_c11_valid) > 0 else np.nan,
                "s_t_h_x_c12_max": s_t_h_x_c12_valid.max() if len(s_t_h_x_c12_valid) > 0 else np.nan,
                "s_t_h_x_c13_max": s_t_h_x_c13_valid.max() if len(s_t_h_x_c13_valid) > 0 else np.nan,
                "s_t_h_x_c14_max": s_t_h_x_c14_valid.max() if len(s_t_h_x_c14_valid) > 0 else np.nan,
                "s_t_m_x_c0_max": s_t_m_x_c0_valid.max() if len(s_t_m_x_c0_valid) > 0 else np.nan,
                "s_t_m_x_c1_max": s_t_m_x_c1_valid.max() if len(s_t_m_x_c1_valid) > 0 else np.nan,
                "s_t_l_x_c0_max": s_t_l_x_c0_valid.max() if len(s_t_l_x_c0_valid) > 0 else np.nan,
                "s_t_l_x_c1_max": s_t_l_x_c1_valid.max() if len(s_t_l_x_c1_valid) > 0 else np.nan,
                "s_t_l_x_c2_max": s_t_l_x_c2_valid.max() if len(s_t_l_x_c2_valid) > 0 else np.nan,
                "s_t_l_x_c3_max": s_t_l_x_c3_valid.max() if len(s_t_l_x_c3_valid) > 0 else np.nan,
                "s_t_l_x_c4_max": s_t_l_x_c4_valid.max() if len(s_t_l_x_c4_valid) > 0 else np.nan,
                "s_t_l_x_c5_max": s_t_l_x_c5_valid.max() if len(s_t_l_x_c5_valid) > 0 else np.nan,
                "s_t_l_x_c6_max": s_t_l_x_c6_valid.max() if len(s_t_l_x_c6_valid) > 0 else np.nan,
                "s_t_l_x_c7_max": s_t_l_x_c7_valid.max() if len(s_t_l_x_c7_valid) > 0 else np.nan,
                "s_t_l_x_c8_max": s_t_l_x_c8_valid.max() if len(s_t_l_x_c8_valid) > 0 else np.nan,
                "s_t_l_x_c9_max": s_t_l_x_c9_valid.max() if len(s_t_l_x_c9_valid) > 0 else np.nan,
                "s_t_l_x_c10_max": s_t_l_x_c10_valid.max() if len(s_t_l_x_c10_valid) > 0 else np.nan,
                "sp_x_c0_max": sp_x_c0_valid.max() if len(sp_x_c0_valid) > 0 else np.nan,
                "sp_x_c1_max": sp_x_c1_valid.max() if len(sp_x_c1_valid) > 0 else np.nan,
                "sp_x_c2_max": sp_x_c2_valid.max() if len(sp_x_c2_valid) > 0 else np.nan,
                "sp_x_c3_max": sp_x_c3_valid.max() if len(sp_x_c3_valid) > 0 else np.nan,
                "t_x_c0_max": t_x_c0_valid.max() if len(t_x_c0_valid) > 0 else np.nan,
                "t_x_c1_max": t_x_c1_valid.max() if len(t_x_c1_valid) > 0 else np.nan,
                "t_x_c2_max": t_x_c2_valid.max() if len(t_x_c2_valid) > 0 else np.nan,
                "t_x_c3_max": t_x_c3_valid.max() if len(t_x_c3_valid) > 0 else np.nan,
                "t_x_c4_max": t_x_c4_valid.max() if len(t_x_c4_valid) > 0 else np.nan,
                "t_x_c5_max": t_x_c5_valid.max() if len(t_x_c5_valid) > 0 else np.nan,
                "t_x_c6_max": t_x_c6_valid.max() if len(t_x_c6_valid) > 0 else np.nan,
                "t_x_c7_max": t_x_c7_valid.max() if len(t_x_c7_valid) > 0 else np.nan,
                "t_x_c8_max": t_x_c8_valid.max() if len(t_x_c8_valid) > 0 else np.nan,
                "st_x_c0_max": st_x_c0_valid.max() if len(st_x_c0_valid) > 0 else np.nan,
                "st_x_c1_max": st_x_c1_valid.max() if len(st_x_c1_valid) > 0 else np.nan,
                "st_x_c2_max": st_x_c2_valid.max() if len(st_x_c2_valid) > 0 else np.nan,
            }
        )
        try:
            assert len(s_t_h_x_c2_valid) > 0
            assert len(s_t_l_x_c9_valid) > 0
            assert len(s_t_l_x_c10_valid) > 0
            assert len(t_x_c7_valid) > 0
            assert len(t_x_c8_valid) > 0
        
            if len(s_t_h_x_c2_valid) > 0 and s_t_h_x_c2_valid.min() < -1:
                print(f"Found value less than -1 in tif {i}: {s_t_h_x_c2_valid.min()}", flush=True)
                import pdb; pdb.set_trace()
            if len(s_t_l_x_c9_valid) > 0 and s_t_l_x_c9_valid.min() < -1:
                print(f"Found value less than -1 in tif {i}: {s_t_l_x_c9_valid.min()}", flush=True)
                import pdb; pdb.set_trace()
            if len(s_t_l_x_c10_valid) > 0 and s_t_l_x_c10_valid.min() < -1:
                print("min:", s_t_l_x_c10_valid.min(), "dtype:", s_t_l_x_c10_valid.dtype, "id:", id(s_t_l_x_c10_valid))
                import pdb; pdb.set_trace()
            if len(t_x_c7_valid) > 0 and t_x_c7_valid.min() < -1:
                print(f"Found value less than -1 in tif {i}: {t_x_c7_valid.min()}", flush=True)
                import pdb; pdb.set_trace()
            if len(t_x_c8_valid) > 0 and t_x_c8_valid.min() < -1:
                print(f"Found value less than -1 in tif {i}: {t_x_c8_valid.min()}", flush=True)
                import pdb; pdb.set_trace()
            if len(t_x_c8_valid) > 0 and t_x_c8_valid.min() < -1:
                print(f"Found value less than -1 in tif {i}: {t_x_c8_valid.min()}", flush=True)
                import pdb; pdb.set_trace()
            if len(s_t_h_x_c2_valid) > 0 and s_t_h_x_c2_valid.max() > 2:
                print(f"Found value greater than 2 in tif {i}: {s_t_h_x_c2_valid.max()}", flush=True)
                import pdb; pdb.set_trace()
            if len(s_t_l_x_c9_valid) > 0 and s_t_l_x_c9_valid.max() > 2:
                print(f"Found value greater than 2 in tif {i}: {s_t_l_x_c9_valid.max()}", flush=True)
                import pdb; pdb.set_trace()
            if len(s_t_l_x_c10_valid) > 0 and s_t_l_x_c10_valid.max() > 2:
                print(f"Found value greater than 2 in tif {i}: {s_t_l_x_c10_valid.max()}", flush=True)
                import pdb; pdb.set_trace()
        except AssertionError:
            print(f"No valid data found for some channels in tif {i}", flush=True)
    """

    import pandas as pd

    df = pd.DataFrame(stats)
    df.to_csv("data_stats_mean_std_new.csv", index=False)
