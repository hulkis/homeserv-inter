import pickle
import re

import catboost as cgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from homeserv_inter.constants import (
    CATEGORICAL_FEATURES, CLEANED_DATA_DIR, DATA_DIR, DROPCOLS, LABEL_COLS,
    LOW_IMPORTANCE_FEATURES, NLP_COLS, SEED, STR_COLS, TIMESTAMP_COLS)
from homeserv_inter.sklearn_missval import LabelEncoderByColMissVal
from sklearn import model_selection, preprocessing
from wax_toolbox.profiling import Timer


# Feature ing.:
def datetime_features_single(df, col):
    # It is needed to reconvert to datetime, as parquet fail and get sometimes
    # int64 which are epoch, but pandas totally handle it like a boss
    serie = pd.to_datetime(df[col]).dropna()
    df.loc[serie.index, col + "_year"] = serie.dt.year
    df.loc[serie.index, col + "_month"] = serie.dt.month
    df.loc[serie.index, col + "_day"] = serie.dt.day
    df.loc[serie.index, col + "_dayofyear"] = serie.dt.dayofyear
    df.loc[serie.index, col + "_week"] = serie.dt.week
    df.loc[serie.index, col + "_weekday"] = serie.dt.weekday

    if col in ['CRE_DATE', 'UPD_DATE']:  # only those got hour, else is day
        df.loc[serie.index, col + "_hour"] = serie.dt.hour

    # Fill na with -1 for integer columns & convert it to signed
    regex = col + "_(year)|(month)|(day)|(dayofyear)|(week)|(weekday)|(hour)"
    lst_cols = df.filter(regex=regex).columns.tolist()
    for col in lst_cols:
        df[col] = df[col].fillna(-1)
        df[col] = pd.to_numeric(df[col], downcast="signed")

    return df


def build_features_datetime(df):
    columns = df.columns.tolist()
    timestamp_cols = set(TIMESTAMP_COLS).intersection(set(df.columns.tolist()))
    for col in timestamp_cols:
        if col in columns:
            df = datetime_features_single(df, col)

    # Relative features with ref date as CRE_DATE_GZL
    # (The main date at which the intervention bulletin is created):
    for col in [dt for dt in timestamp_cols if dt != 'CRE_DATE_GZL']:
        td_col_name = "bulletin_creation_TD_" + col + "_days"
        df[td_col_name] = (df['CRE_DATE_GZL'] - df[col]).dt.days

    # Some additionnals datetime features
    df['nbdays_duration_of_intervention'] = (
        df['SCHEDULED_END_DATE'] - df['SCHEDULED_START_DATE']).dt.days
    df['nbdays_duration_of_contract'] = (
        df['DATE_FIN'] - df['DATE_DEBUT']).dt.days
    df['nbdays_delta_intervention_contract_start'] = (
        df['CRE_DATE_GZL'] - df['DATE_DEBUT']).dt.days

    # Ratios
    df['ratio_duration_contract_duration_first_interv'] = (
        df['nbdays_duration_of_contract'] /
        df['nbdays_delta_intervention_contract_start'])

    df['ratio_duration_contract_td_install_days'] = (
        df['nbdays_duration_of_contract'] /
        df['bulletin_creation_TD_INSTALL_DATE_days'])

    df['ratio_intervention_contract_start_td_install_days'] = (
        df['nbdays_delta_intervention_contract_start'] /
        df['bulletin_creation_TD_INSTALL_DATE_days'])

    return df


def build_features_str(df):
    # Some Str cleaning:

    # # --> FORMULE:
    # # treat 'SECURITE*' & 'SECURITE* 2V' & 'ESSENTIAL CLIENT' as the same
    # r = re.compile('SECURITE.*')
    # df['FORMULE'] = df['FORMULE'].str.replace(r, 'SECURITE')

    # # treat 'ESSENTIEL P2' & 'ESSENTIEL CLIENT' as the same
    # r = re.compile('ESSENTIEL.*')
    # df['FORMULE'] = df['FORMULE'].str.replace(r, 'ESSENTIEL')

    # # treat 'Sécurité Pack *' as the same
    # r = re.compile('Sécurité Pack.*')
    # df['FORMULE'] = df['FORMULE'].str.replace(r, 'Sécurité Pack')

    # # treat 'TRANQUILITE PRO .*' as nan due to only one register in test set
    # r = re.compile('TRANQUILLITE.*')
    # df['FORMULE'] = df['FORMULE'].replace(
    #     r, np.nan)  # no str so that can be np.nan

    # # --> ORIGINE_INCIDENT:
    # # treat 'Fax' as nan due to only one register in test set
    # df['ORIGINE_INCIDENT'] = df['ORIGINE_INCIDENT'].replace('Fax', np.nan)

    # # treat 'Répondeur', 'Mail', 'Internet' as one label: 'indirect_contact'
    # # but still keep 'Courrier' as it is soooo mainstream, those people are old & odd.
    # r = re.compile('(Répondeur)|(Mail)|(Internet)')
    # df['ORIGINE_INCIDENT'] = df['ORIGINE_INCIDENT'].replace(
    #     r, 'indirect_contact')

    # --> INCIDENT_TYPE_NAME
    # Multi Label Binarize & one hot encoder INCIDENT_TYPE_NAME:
    # i.e. from :            to:
    # Dépannage                 1   0
    # Entretien                 0   1
    # Dépannage+Entretien       1   1

    # df['INCIDENT_TYPE_NAME'] = df['INCIDENT_TYPE_NAME'].str.split('+')
    # mlb = preprocessing.MultiLabelBinarizer()
    # df['INCIDENT_TYPE_NAME'] = list(
    #     mlb.fit_transform(df['INCIDENT_TYPE_NAME']))
    # dftmp = pd.DataFrame(
    #     index=df['INCIDENT_TYPE_NAME'].index,
    #     data=df['INCIDENT_TYPE_NAME'].values.tolist()).add_prefix(
    #         'INCIDENT_TYPE_NAME_label')
    # df = pd.concat([df.drop(columns=['INCIDENT_TYPE_NAME']), dftmp], axis=1)

    # # Categorical features LabelBinarizer (equivalent to onehotencoding):
    # msg = 'One Hot Encoding for CATEGORICAL_FEATURES with pd.get_dummies'
    # with Timer(msg):
    #     for col in CATEGORICAL_FEATURES:
    #         df = pd.concat(
    #             [pd.get_dummies(df[col], prefix=col),
    #              df.drop(columns=[col])],
    #             axis=1,
    #         )

    # Still to do, nlp on nlp_cols, but for the moment take the len of the
    # commentary
    nlp_cols = list(set(df.columns).intersection(set(NLP_COLS)))
    for col in nlp_cols:
        newcol = '{}_len'.format(col)
        df[newcol] = df[col].apply(len)
        df[newcol].replace(1, -1)  # considered as NaN

    df = df.drop(columns=nlp_cols)

    return df


def build_features(df):
    """Build features."""

    with Timer("Building timestamp features"):
        df = build_features_datetime(df)

    timestamp_cols = set(TIMESTAMP_COLS).intersection(set(df.columns.tolist()))
    df = df.drop(columns=timestamp_cols)

    with Timer("Building str features"):
        df = build_features_str(df)

    # Just encore the rest
    with Timer("Encoding labels"):
        label_cols = list(set(df.columns).intersection(set(LABEL_COLS)))
        label_encoder = LabelEncoderByColMissVal(columns=label_cols)
        df = label_encoder.fit_transform(df)

    return df


# Methods to generate cleaned datas:
def _generate_cleaned_single_set(dataset,
                                 drop_cols=None,
                                 use_full_history=False):
    """Generate one cleaned set amon ['train', 'test']"""
    if dataset == "train" and use_full_history:
        pathfile = DATA_DIR / "fullhist_{}.parquet.gzip".format(dataset)
        savepath = CLEANED_DATA_DIR / "fullhist_{}_cleaned.parquet.gzip".format(
            dataset)
    else:
        pathfile = DATA_DIR / "{}.parquet.gzip".format(dataset)
        savepath = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format(dataset)

    with Timer("Reading {} set".format(dataset)):
        df = pd.read_parquet(pathfile)

    if drop_cols is not None:
        drop_cols = list(set(drop_cols).intersection(set(df.columns.tolist())))
        df = df.drop(columns=drop_cols)

    df = build_features(df)

    with Timer("Saving into {}".format(savepath)):
        df.to_parquet(savepath, compression="gzip")


def generate_cleaned_sets(drop_cols=DROPCOLS, use_full_history=False):
    """Generate cleaned sets."""
    with Timer("Gen clean trainset", True):
        _generate_cleaned_single_set(
            dataset="train",
            drop_cols=drop_cols,
            use_full_history=use_full_history)

    with Timer("Gen clean testset", True):
        _generate_cleaned_single_set(
            dataset="test",
            drop_cols=drop_cols,
            use_full_history=use_full_history)


class HomeServiceDataHandle:
    train_parquetpath = DATA_DIR / "train.parquet.gzip"
    test_parquetpath = DATA_DIR / "test.parquet.gzip"

    dftrain = None
    dftest = None

    def __init__(self,
                 debug=True,
                 drop_lowimp_features=False,
                 use_full_history=False):
        self.debug = debug
        self.drop_lowimp_features = drop_lowimp_features
        self.use_full_history = use_full_history

    def _fillna_labelbin(self, df, cols):
        # Fillna with new category for catboost:
        for col in cols:

            # Drop if only NaN
            if df.loc[:, col].isna().all():
                df = df.drop(columns=[col])
                continue

            # df[col] = pd.to_numeric(df[col], downcast='integer')
            df.loc[:, col] = df.loc[:, col].fillna(
                df.loc[:, col].max() + 1).astype(int)
        return df

    def _get_list_categorical_features(self, df):
        str_cols = set(STR_COLS).intersection(set(df.columns.tolist()))
        str_cols = list(str_cols)
        ids_cols = pd.DataFrame(columns=df.columns.tolist()).filter(
            regex='_ID').columns.tolist()
        lst = str_cols + ids_cols

        lst_idx = [df.columns.get_loc(col) for col in str_cols + ids_cols]
        return lst, lst_idx

    def _gen_catboost_set(self, df):
        # Get categorical columns present in df
        cat_cols, _ = self._get_list_categorical_features(df)

        # Fillna & ensure integer type:
        df_cleaned = self._fillna_labelbin(df, cat_cols)

        # Get categorical columns present in df_cleaned
        cat_cols, _ = self._get_list_categorical_features(df_cleaned)

        # Reorder
        new_order_cols = cat_cols + list(
            set(df_cleaned.columns.tolist()) - set(cat_cols))
        df_cleaned = df_cleaned[new_order_cols]

        # Get indices
        cat_cols, cat_indices = self._get_list_categorical_features(df_cleaned)

        return df_cleaned, cat_indices

    # Methods to get cleaned datas:
    def _get_cleaned_single_set(self, dataset="train"):
        if self.use_full_history and dataset == "train":
            pathdata = CLEANED_DATA_DIR / "fullhist_{}_cleaned.parquet.gzip".format(
                dataset)
        else:
            pathdata = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format(
                dataset)

        with Timer("Reading {} set".format(dataset)):
            df = pd.read_parquet(pathdata)
            if self.debug:
                df = df.sample(
                    n=10000, random_state=SEED).dropna(
                        axis=1, how="all")

        with Timer("Replacing -1 categorical by np.nan"):
            lst_cols = list(set(df.columns.tolist()).intersection(LABEL_COLS))
            for col in lst_cols:
                df[col] = df[col].replace(-1, np.nan)

        return df

    def get_test_set(self, as_cgb_pool=False, order_cols=None):
        df = self._get_cleaned_single_set(dataset="test")

        if self.drop_lowimp_features:
            print('Dropping low importance features !')
            dropcols = set(df.columns.tolist()).intersection(
                set(LOW_IMPORTANCE_FEATURES))
            df = df.drop(columns=list(dropcols))

        if not order_cols:
            order_cols = df.columns.tolist()

        if as_cgb_pool:
            df_cat, indices = self._gen_catboost_set(df[order_cols])
            with Timer('Creating Test Pool'):
                pool = cgb.Pool(df_cat, label=None, cat_features=indices)

            return pool

        return df

    def get_train_set(self,
                      as_xgb_dmatrix=False,
                      as_lgb_dataset=False,
                      as_cgb_pool=False):
        df = self._get_cleaned_single_set(dataset="train")
        train_cols = df.columns.tolist()
        train_cols.remove("target")

        if self.drop_lowimp_features:
            print('Dropping low importance features !')
            dropcols = set(df.columns.tolist()).intersection(
                set(LOW_IMPORTANCE_FEATURES))
            df = df.drop(columns=list(dropcols))

        if as_xgb_dmatrix:
            return xgb.DMatrix(data=df[train_cols], label=df[["target"]])
        elif as_lgb_dataset:
            return lgb.Dataset(df[train_cols], df[["target"]].values.ravel())
        elif as_cgb_pool:
            df, indices = self._gen_catboost_set(df)

            with Timer('Creating Train Pool'):
                Ptrain = cgb.Pool(
                    df.drop(columns=["target"]), df[["target"]], indices)

            return Ptrain, df.drop(columns=["target"]).columns.tolist()
        else:

            return df[train_cols], df[["target"]]

    def get_train_valid_set(self,
                            split_perc=0.2,
                            as_xgb_dmatrix=False,
                            as_lgb_dataset=False,
                            as_cgb_pool=False):
        df = self._get_cleaned_single_set(dataset="train")

        if self.drop_lowimp_features:
            print('Dropping low importance features !')
            dropcols = set(df.columns.tolist()).intersection(
                set(LOW_IMPORTANCE_FEATURES))
            df = df.drop(columns=list(dropcols))

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
        elif as_cgb_pool:
            df_train, indices_train = self._gen_catboost_set(Xtrain)
            df_test, indices_test = self._gen_catboost_set(Xtest)

            with Timer('Creating Train Pool'):
                Ptrain = cgb.Pool(df_train, ytrain, indices_train)
            with Timer('Creating Test Pool'):
                Ptest = cgb.Pool(df_test, ytest, indices_test)
            return (Ptrain, Ptest)
        else:
            return Xtrain, Xtest, ytrain, ytest


# > homeserv cgb validate --debug=False --num-boost-round=10000 --early-stopping-rounds=200
# 373:    test: 0.9051169 best: 0.9051169 (373)   test1: 0.8295167        total: 1h 9m 36s        remaining: 1d 5h 51m 42s
# 374:    test: 0.9053387 best: 0.9053387 (374)   test1: 0.8295473        total: 1h 9m 48s        remaining: 1d 5h 51m 35s
# 375:    test: 0.9055101 best: 0.9055101 (375)   test1: 0.8296845        total: 1h 9m 59s        remaining: 1d 5h 51m 30s
# 376:    test: 0.9055589 best: 0.9055589 (376)   test1: 0.8300121        total: 1h 10m 11s       remaining: 1d 5h 51m 27s
# 377:    test: 0.9057872 best: 0.9057872 (377)   test1: 0.8300845        total: 1h 10m 21s       remaining: 1d 5h 51m 5s
# 378:    test: 0.9058749 best: 0.9058749 (378)   test1: 0.8304580        total: 1h 10m 33s       remaining: 1d 5h 51m 5s
# 379:    test: 0.9060632 best: 0.9060632 (379)   test1: 0.8306174        total: 1h 10m 46s       remaining: 1d 5h 51m 47s
# 380:    test: 0.9060809 best: 0.9060809 (380)   test1: 0.8309316        total: 1h 10m 58s       remaining: 1d 5h 51m 44s
# 381:    test: 0.9062347 best: 0.9062347 (381)   test1: 0.8312468        total: 1h 11m 9s        remaining: 1d 5h 51m 29s
# 382:    test: 0.9063261 best: 0.9063261 (382)   test1: 0.8313696        total: 1h 11m 21s       remaining: 1d 5h 51m 40s
# 383:    test: 0.9064714 best: 0.9064714 (383)   test1: 0.8320675        total: 1h 11m 34s       remaining: 1d 5h 52m 10s
# 384:    test: 0.9065794 best: 0.9065794 (384)   test1: 0.8325532        total: 1h 11m 46s       remaining: 1d 5h 52m 23s
# 385:    test: 0.9068535 best: 0.9068535 (385)   test1: 0.8186662        total: 1h 11m 59s       remaining: 1d 5h 53m
# 386:    test: 0.9069289 best: 0.9069289 (386)   test1: 0.8190755        total: 1h 12m 10s       remaining: 1d 5h 52m 53s
# 387:    test: 0.9069990 best: 0.9069990 (387)   test1: 0.8194594        total: 1h 12m 21s       remaining: 1d 5h 52m 40s
# 388:    test: 0.9071230 best: 0.9071230 (388)   test1: 0.8197652        total: 1h 12m 32s       remaining: 1d 5h 52m 26s
# 389:    test: 0.9072616 best: 0.9072616 (389)   test1: 0.8203275        total: 1h 12m 45s       remaining: 1d 5h 52m 52s
# 390:    test: 0.9073305 best: 0.9073305 (390)   test1: 0.8206990        total: 1h 12m 56s       remaining: 1d 5h 52m 43s
# 391:    test: 0.9074454 best: 0.9074454 (391)   test1: 0.8207700        total: 1h 13m 10s       remaining: 1d 5h 53m 32s
