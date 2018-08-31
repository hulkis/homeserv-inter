import pickle

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing

from homeserv_inter.constants import (
    CLEANED_DATA_DIR,
    DATA_DIR,
    DROPCOLS,
    ISNA_COLS,
    SEED,
    LABEL_COLS,
    NLP_COLS,
    TIMESTAMP_COLS,
)
from wax_toolbox.profiling import Timer


def _encode_labels(df, label_cols):
    """Encode labels."""
    label_encoders = []
    for col in label_cols:
        le_tmp = preprocessing.LabelEncoder()
        df[col] = pd.to_numeric(le_tmp.fit_transform(df[col]), downcast="unsigned")
        label_encoders.append(le_tmp)

    return df, label_encoders


def _decode_labels(df, label_cols, label_encoders):
    if not len(label_cols) == len(label_encoders):
        raise ValueError("Not same len for label_cols and label_encoders.")
    for i, col in enumerate(label_cols):
        df[col] = label_cols[i].inverse_transform(df[col])
    return df


# Feature ing.:
def datetime_features(df, col):
    idx = df[col].dt
    df[col + "_year"] = pd.to_numeric(idx.year.fillna(0), downcast="unsigned")
    df[col + "_month"] = pd.to_numeric(idx.month.fillna(0), downcast="unsigned")
    df[col + "_day"] = pd.to_numeric(idx.day.fillna(0), downcast="unsigned")
    df[col + "_dayofyear"] = pd.to_numeric(idx.dayofyear.fillna(0), downcast="unsigned")
    # df[col + "_hour"] = pd.to_numeric(idx.hour.fillna(0), downcast='unsigned')
    # df[col + "_minute"] = pd.to_numeric(idx.minute.fillna(0), downcast='unsigned')
    df[col + "_week"] = pd.to_numeric(idx.week.fillna(0), downcast="unsigned")
    df[col + "_weekday"] = pd.to_numeric(idx.weekday.fillna(0), downcast="unsigned")
    return df


def build_features(df):
    """Build features."""

    # Intervention duration
    serie = (
        df["SCHEDULED_END_DATE"] - df["SCHEDULED_START_DATE"]
    ).dt.total_seconds() / 3600
    df["INTERVENTION_DURATION_HOUR"] = pd.to_numeric(serie, downcast="unsigned")
    df = df.drop(columns=["SCHEDULED_END_DATE", "SCHEDULED_START_DATE"])

    # Is Not a Number columns:
    for col in ISNA_COLS:
        df[col + "_isna"] = pd.to_numeric(
            df[col].isna().astype(int), downcast="unsigned"
        )
    df = df.drop(columns=ISNA_COLS)

    with Timer("Building timestamp features", report_func=print):
        columns = df.columns.tolist()
        for col in TIMESTAMP_COLS:
            if col in columns:
                df = datetime_features(df, col)
                df = df.drop(columns=[col])

    with Timer("Encoding labels", report_func=print):
        label_cols = list(set(df.columns).intersection(set(LABEL_COLS)))
        df, label_encoders = _encode_labels(df, label_cols)

    # Still to do, nlp on nlp_cols
    nlp_cols = list(set(df.columns).intersection(set(NLP_COLS)))
    df = df.drop(columns=nlp_cols)

    return df, label_encoders


# Methods to generate cleaned datas:
def _generate_cleaned_single_set(dataset, drop_cols=None):
    """Generate one cleaned set amon ['train', 'test']"""
    with Timer("Reading {} set".format(dataset), report_func=print):
        df = pd.read_parquet(DATA_DIR / "{}.parquet.gzip".format(dataset))

    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    df, label_encoders = build_features(df)

    pathpickle = CLEANED_DATA_DIR / "{}_labelencoders.pickle".format(dataset)
    with open(pathpickle.as_posix(), "wb") as f:
        pickle.dump(label_encoders, f)

    savepath = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format(dataset)
    with Timer("Saving into {}".format(savepath)):
        df.to_parquet(savepath, compression="gzip")


def generate_cleaned_sets(drop_cols=DROPCOLS):
    """Generate cleaned sets."""
    with Timer("Gen clean trainset", report_at_enter=True, report_func=print):
        _generate_cleaned_single_set(dataset="train", drop_cols=drop_cols)

    with Timer("Gen clean testset", report_at_enter=True, report_func=print):
        _generate_cleaned_single_set(dataset="test", drop_cols=drop_cols)


class HomeServiceDataHandle:
    train_parquetpath = DATA_DIR / "train.parquet.gzip"
    test_parquetpath = DATA_DIR / "test.parquet.gzip"

    dftrain = None
    dftest = None

    def __init__(self, debug=True, mode="validation"):
        self.debug = debug
        self.mode = mode

    # Methods to get cleaned datas:
    def _get_cleaned_single_set(self, dataset="train"):
        with Timer("Reading train set", report_func=print):
            pathdata = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format(dataset)
            df = pd.read_parquet(pathdata)
            if self.debug:
                df = df.sample(n=10000, random_state=SEED).dropna(axis=1, how="all")

        pathpickle = CLEANED_DATA_DIR / "{}_labelencoders.pickle".format(dataset)
        with open(pathpickle.as_posix(), "rb") as f:
            label_encoders = pickle.load(f)
        return df, label_encoders

    def get_test_set(self):
        df, self.test_label_encoder = self._get_cleaned_single_set(dataset="test")
        return df

    def get_train_set(self, as_xgb_dmatrix=False, as_lgb_dataset=False):
        df, self.train_label_encoder = self._get_cleaned_single_set(dataset="train")
        train_cols = df.columns.tolist()
        train_cols.remove("target")
        if as_xgb_dmatrix:
            pass
        elif as_lgb_dataset:
            return lgb.Dataset(df[train_cols], df[["target"]].values.ravel())
        else:
            return df[train_cols], df[["target"]]

    def get_train_valid_set(
        self, split_perc=0.2, as_xgb_dmatrix=False, as_lgb_dataset=False
    ):
        df, self.train_label_encoder = self._get_cleaned_single_set(dataset="train")
        Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(
            df.drop(columns=["target"]),
            df[["target"]],
            test_size=split_perc,
            random_state=42,
        )
        if as_xgb_dmatrix:
            return xgb.DMatrix(Xtrain, ytrain), xgb.DMatrix(Xtest, ytest)
        elif as_lgb_dataset:
            return (
                lgb.Dataset(Xtrain, ytrain.values.ravel()),
                lgb.Dataset(Xtest, ytest.values.ravel()),
            )
        else:
            return Xtrain, Xtest, ytrain, ytest
