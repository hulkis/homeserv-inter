import lightgbm as lgb
import pandas as pd
from sklearn import preprocessing

from homeserv_inter import DATA_DIR, label_cols, nlp_cols, timestamp_cols
from wax_toolbox.profiling import Timer

SEED = 42


def datetime_features(df, col):
    idx = df[col].dt
    df[col + "_year"] = idx.year
    df[col + "_month"] = idx.month
    df[col + "_day"] = idx.day
    df[col + "_dayofyear"] = idx.dayofyear
    df[col + "_hour"] = idx.hour
    df[col + "_minute"] = idx.minute

    df[col + "_week"] = idx.week
    df[col + "_weekday"] = idx.weekday
    return df


def _encode_labels(df, label_cols):
    """Encode labels."""
    label_encoders = []
    for col in label_cols:
        le_tmp = preprocessing.LabelEncoder()
        df[col] = le_tmp.fit_transform(df[col])
        label_encoders.append(le_tmp)

    return df, label_encoders


def _decode_labels(df, label_cols, label_encoders):
    if not len(label_cols) == len(label_encoders):
        raise ValueError("Not same len for label_cols and label_encoders.")
    for i, col in enumerate(label_cols):
        df[col] = label_cols[i].inverse_transform(df[col])
    return df


def build_features(df):
    """Build features."""

    with Timer("Building timestamp features", report_func=print):
        columns = df.columns.tolist()
        for col in timestamp_cols:
            if col in columns:
                df = datetime_features(df, col)
                df = df.drop(columns=[col])

    with Timer("Encoding labels", report_func=print):
        df, label_encoders = _encode_labels(df, label_cols)

    # Still to do, nlp on nlp_
    df = df.drop(columns=nlp_cols)

    return df, label_encoders


class HomeServiceDataHandle:
    train_parquetpath = DATA_DIR / "train.parquet.gzip"
    test_parquetpath = DATA_DIR / "test.parquet.gzip"

    dftrain = None
    dftest = None

    def __init__(self, debug=True, mode="validation"):
        self.debug = debug
        self.mode = mode

    def _get_formatted_datas(self, split_perc=0.2, as_lgb_dataset=False):
        with Timer("Reading train set", report_func=print):
            if self.debug:
                dftrain = pd.read_parquet(DATA_DIR / "train.parquet.gzip").sample(
                    n=100000, random_state=SEED,
                )
                dftrain = dftrain.dropna(axis=1, how="all")
            else:
                dftrain = pd.read_parquet(DATA_DIR / "train.parquet.gzip")

        # if self.mode == "predict" and not self.debug:
        #     with Timer("Reading test set", report_func=print):
        #         dftest = pd.read_parquet(DATA_DIR / "test.parquet.gzip")

        df, label_encoders = build_features(dftrain)

        if as_lgb_dataset:
            return lgb.Dataset(data=df), label_encoders

        else:
            return df, label_encoders
