import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from homeserv_inter.constants import (
    CLEANED_DATA_DIR,
    DATA_DIR,
    DROPCOLS,
    LABEL_COLS,
    NLP_COLS,
    SEED,
    TIMESTAMP_COLS,
)
from wax_toolbox.profiling import Timer


class LabelEncoderByCol(BaseEstimator, TransformerMixin):
    x_cleaned = None

    def __init__(self, columns):
        # List of column names in the DataFrame that should be encoded
        self.columns = columns
        # Dictionary storing a LabelEncoder for each column
        self.label_encoders = {}
        for el in self.columns:
            self.label_encoders[el] = LabelEncoder()

    def handle_nans(self, x):
        if self.x_cleaned is None:
            with Timer("Cleaning NaNs for label encoding"):
                # Fill missing values with the string 'NaN'
                x[self.columns] = x[self.columns].fillna("NaN")

                # str to replace interpreted as NaN:
                lst_str_nans = ["nan", "", "-"]
                for s in lst_str_nans:
                    for col in self.columns:
                        x[col] = x[col].replace(s, "NaN")

            self.x_cleaned = x.copy()

        return self.x_cleaned

    def fit(self, x):
        x = self.handle_nans(x)

        for el in self.columns:
            # Only use the values that are not 'NaN' to fit the Encoder
            a = x[el][x[el] != "NaN"]
            self.label_encoders[el].fit(a)
        return self

    def transform(self, x):
        x = self.handle_nans(x)

        for el in self.columns:
            # Only use the values that are not 'NaN' to fit the Encoder
            a = x[el][x[el] != "NaN"]
            # Store an ndarray of the current column
            b = x[el].get_values()
            # Replace the elements in the ndarray that are not 'NaN'
            # using the transformer
            b[b != "NaN"] = self.label_encoders[el].transform(a)
            # Overwrite the column in the DataFrame
            x[el] = b

            # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst#categorical_feature
            # Replace string "NaN" by -1 as lgb: all negative values will be treated as missing values
            # SHIT ! seg fault when -1 at init booster time, go for np.nan
            # conversion at the moment we read the parquet file...
            x[el] = x[el].replace("NaN", -1)

        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def _decode_labels(df, label_cols, label_encoders):
    if not len(label_cols) == len(label_encoders):
        raise ValueError("Not same len for label_cols and label_encoders.")
    for i, col in enumerate(label_cols):
        df[col] = label_cols[i].inverse_transform(df[col])
    return df


# Feature ing.:
def datetime_features(df, col):
    # It is needed to reconvert to datetime, as parquet fail and get sometimes
    # int64 which are epoch, but pandas totally handle it like a boss
    serie = pd.to_datetime(df[col]).dropna()
    df.loc[serie.index, col + "_year"] = serie.dt.year
    df.loc[serie.index, col + "_month"] = serie.dt.month
    df.loc[serie.index, col + "_day"] = serie.dt.day
    df.loc[serie.index, col + "_dayofyear"] = serie.dt.dayofyear
    df.loc[serie.index, col + "_week"] = serie.dt.week
    df.loc[serie.index, col + "_weekday"] = serie.dt.weekday
    df = df.drop(columns=[col])

    lst_cols = df.filter(regex=col + "_*").columns.tolist()
    for col in lst_cols:
        df[col] = df[col].fillna(-1)
        df[col] = pd.to_numeric(df[col], downcast="signed")

    return df


def build_features(df):
    """Build features."""

    with Timer("Building timestamp features"):
        columns = df.columns.tolist()
        for col in TIMESTAMP_COLS:
            if col in columns:
                df = datetime_features(df, col)

    with Timer("Encoding labels"):
        label_cols = list(set(df.columns).intersection(set(LABEL_COLS)))
        label_encoder = LabelEncoderByCol(columns=label_cols)
        df = label_encoder.fit_transform(df)

    # Still to do, nlp on nlp_cols
    nlp_cols = list(set(df.columns).intersection(set(NLP_COLS)))
    df = df.drop(columns=nlp_cols)

    # # Convert all in integers as there is no float
    # with Timer("Converting all dataset into integers"):
    #     for col in df.columns:
    #         df[col] = pd.to_numeric(df[col], downcast="signed")

    return df, label_encoder.label_encoders


# Methods to generate cleaned datas:
def _generate_cleaned_single_set(dataset, drop_cols=None):
    """Generate one cleaned set amon ['train', 'test']"""
    with Timer("Reading {} set".format(dataset)):
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
    with Timer("Gen clean trainset", True):
        _generate_cleaned_single_set(dataset="train", drop_cols=drop_cols)

    with Timer("Gen clean testset", True):
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
        with Timer("Reading train set"):
            pathdata = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format(dataset)
            df = pd.read_parquet(pathdata)
            if self.debug:
                df = df.sample(n=10000, random_state=SEED).dropna(axis=1, how="all")

        with Timer("Replacing -1 categorical by np.nan"):
            lst_cols = list(set(df.columns.tolist()).intersection(LABEL_COLS))
            for col in lst_cols:
                df[col] = df[col].replace(-1, np.nan)

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
            return xgb.DMatrix(df[train_cols], df[["target"]]),
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
            return (
                xgb.DMatrix(Xtrain, ytrain),
                xgb.DMatrix(Xtest, ytest),
            )
        elif as_lgb_dataset:
            return (
                lgb.Dataset(Xtrain, ytrain.values.ravel()),
                lgb.Dataset(Xtest, ytest.values.ravel()),
            )
        else:
            return Xtrain, Xtest, ytrain, ytest
