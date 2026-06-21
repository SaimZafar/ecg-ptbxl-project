import os
import ast
import numpy as np
import pandas as pd
import wfdb

from config import DATA_DIR, CLASSES, SAMPLE_RATE


def load_metadata():
    """Load the PTB-XL metadata CSV and the diagnostic code mapping."""
    meta_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
    scp_path = os.path.join(DATA_DIR, "scp_statements.csv")

    df = pd.read_csv(meta_path, index_col="ecg_id")
    # scp_codes is stored as a string like "{'NORM': 100.0, 'LVH': 80.0}"
    # ast.literal_eval safely turns that string back into a real dict
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    # only keep rows that are actual diagnostic codes (not rhythm/form codes)
    scp_df = scp_df[scp_df["diagnostic"] == 1]

    return df, scp_df


def code_to_superclass_map(scp_df):
    """Build a dict like {'IMI': 'MI', 'NORM': 'NORM', 'LVH': 'HYP', ...}"""
    return scp_df["diagnostic_class"].to_dict()


def build_labels(df, code_map):
    """
    For each patient, turn their scp_codes dict into a multi-hot vector
    over the 5 superclasses, e.g. [1, 0, 0, 0, 0] for NORM-only.
    """
    labels = np.zeros((len(df), len(CLASSES)), dtype=np.float32)

    for row_idx, codes_dict in enumerate(df["scp_codes"]):
        for code in codes_dict.keys():
            superclass = code_map.get(code)
            if superclass in CLASSES:
                class_idx = CLASSES.index(superclass)
                labels[row_idx, class_idx] = 1.0

    return labels


def load_signal(df_row):
    """Load one ECG recording given its metadata row."""
    if SAMPLE_RATE == 100:
        path = os.path.join(DATA_DIR, df_row["filename_lr"])
    else:
        path = os.path.join(DATA_DIR, df_row["filename_hr"])

    signal, _ = wfdb.rdsamp(path)
    return signal  # shape: (1000, 12) for 100Hz — 10 seconds, 12 leads


def get_splits(df):
    """PTB-XL's official split: folds 1-8 train, 9 val, 10 test."""
    train_df = df[df["strat_fold"] <= 8]
    val_df = df[df["strat_fold"] == 9]
    test_df = df[df["strat_fold"] == 10]
    return train_df, val_df, test_df