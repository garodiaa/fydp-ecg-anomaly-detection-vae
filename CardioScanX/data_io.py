import numpy as np
import wfdb
from pathlib import Path
import zipfile
import io


SUPPORTED_EXT = ['.npy', '.dat', '.hea']


def load_ecg_file(fp: str):
    """
    Load ECG data from .npy, .dat/.hea (WFDB-compatible) or raise on failure.
    Returns a 12xT numpy array.
    """
    p = Path(fp)
    ext = p.suffix.lower()
    if ext == '.npy':
        arr = np.load(fp)
        # ensure shape [12, T]
        if arr.ndim != 2:
            raise ValueError('Expected 2D array saved in .npy file')
        if arr.shape[0] == 12:
            return arr
        elif arr.shape[1] == 12:
            return arr.T
        elif arr.shape[0] > 12:
            return arr[:12, :]
        else:
            raise ValueError('Unexpected numpy shape for ECG file')

    elif ext in ['.dat', '.hea']:
        # WFDB usually requires reading by record name (without .dat). Attempt to read using rdsamp
        # If user provided .dat/.hea file path explicitly we will attempt to rdsamp using parent dir
        path = str(p)
        try:
            # wfdb rdsamp expects record path without extension
            base = str(p.with_suffix(''))
            record = wfdb.rdsamp(base)
        except Exception:
            # fallback: try to read using the file path as record (some wfdb versions allow full path)
            record = wfdb.rdsamp(path)
        sig = record.p_signal
        # convert to numpy and ensure 12 leads
        arr = np.array(sig).T
        if arr.shape[0] >= 12:
            return arr[:12, :]
        else:
            raise ValueError('WFDB file has less than 12 leads')

    else:
        raise ValueError('Unsupported extension: ' + ext)


def extract_from_zip(zpath: str):
    z = zipfile.ZipFile(zpath)
    files = []
    for fn in z.namelist():
        if Path(fn).suffix.lower() in SUPPORTED_EXT:
            # return a tuple (filename, bytes) if needed. For now just extract to tmp
            files.append(fn)
    return files
