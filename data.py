import os
import zipfile
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED_DIR = BASE_DIR / "eeg-dataset/derivatives/preprocessed_eeg"


def load_subject_data(sub_id: int, preprocessed_dir=None, return_times: bool = False):
    """
    Load preprocessed test EEG array for THINGS-EEG2:
      Shape: (200 concepts, 80 reps, 63 channels, 100 times)
    Channel 64 ('stim') is removed.
    """
    p_dir = Path(preprocessed_dir) if preprocessed_dir else DEFAULT_PREPROCESSED_DIR
    zf_name = f"sub-{sub_id:02d}__63_channels.zip"
    zf_path = p_dir / zf_name
    if not zf_path.exists():
        raise FileNotFoundError(f"Missing subject zip: {zf_path}")

    npy_name = f"sub-{sub_id:02d}__63_channels/preprocessed_eeg_test.npy"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open(npy_name) as fp:
            raw = np.load(fp, allow_pickle=True).item()

    data = raw["preprocessed_eeg_data"][:, :, :63, :]
    ch_names = list(raw["ch_names"][:63])

    if return_times:
        times = raw.get("times", None)
        return data, ch_names, times
    return data, ch_names

