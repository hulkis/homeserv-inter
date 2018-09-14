import re

import catboost as cgb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing

from homeserv_inter.constants import (
    CATBOOST_FEATURES, CATEGORICAL_FEATURES, CLEANED_DATA_DIR, DATA_DIR,
    DROPCOLS, LABEL_COLS, LOW_IMPORTANCE_FEATURES, NLP_COLS, RAW_DATA_DIR,
    SEED, TIMESTAMP_COLS)
from homeserv_inter.sklearn_missval import LabelEncoderByColMissVal
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


DEFAULT_SUBSET_HIST = [
    'SCHEDULED_START_DATE',
    'SCHEDULED_END_DATE',
    'ACTUAL_START_DATE',
    'ACTUAL_END_DATE',
    "INSTANCE_ID",
    "RESOURCE_ID",
]


def read_intervention_history(subset=DEFAULT_SUBSET_HIST):
    with Timer("Reading intervention_history.csv"):
        df = pd.read_csv(
            RAW_DATA_DIR / "intervention_history.csv",
            sep="|",
            encoding="Latin-1")
        dt_cols = [
            "DATE_SAISIE_RETOUR",
            "SCHEDULED_START_DATE",
            "SCHEDULED_END_DATE",
            "ACTUAL_START_DATE",
            "ACTUAL_END_DATE",
            "CRE_DATE_GZL",
        ]

    for col in dt_cols:
        df[col] = pd.to_datetime(df[col])

    if subset is not None:
        assert all([v in df.columns for v in subset])
        return df[subset]
    else:
        return df


def build_features_datetime(df):
    columns = df.columns.tolist()
    for col in TIMESTAMP_COLS:
        if col in columns:
            df = datetime_features_single(df, col)

    # Relative features with ref date as CRE_DATE_GZL
    # (The main date at which the intervention bulletin is created):
    for col in [dt for dt in TIMESTAMP_COLS if dt != 'CRE_DATE_GZL']:
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

    # --> FORMULE:
    # treat 'SECURITE*' & 'SECURITE* 2V' & 'ESSENTIAL CLIENT' as the same
    r = re.compile('SECURITE.*')
    df['FORMULE'] = df['FORMULE'].str.replace(r, 'SECURITE')

    # treat 'ESSENTIEL P2' & 'ESSENTIEL CLIENT' as the same
    r = re.compile('ESSENTIEL.*')
    df['FORMULE'] = df['FORMULE'].str.replace(r, 'ESSENTIEL')

    # treat 'Sécurité Pack *' as the same
    r = re.compile('Sécurité Pack.*')
    df['FORMULE'] = df['FORMULE'].str.replace(r, 'Sécurité Pack')

    # treat 'TRANQUILITE PRO .*' as nan due to only one register in test set
    r = re.compile('TRANQUILLITE.*')
    df['FORMULE'] = df['FORMULE'].replace(
        r, np.nan)  # no str so that can be np.nan

    # --> ORIGINE_INCIDENT:
    # treat 'Fax' as nan due to only one register in test set
    df['ORIGINE_INCIDENT'] = df['ORIGINE_INCIDENT'].replace('Fax', np.nan)

    # treat 'Répondeur', 'Mail', 'Internet' as one label: 'indirect_contact'
    # but still keep 'Courrier' as it is soooo mainstream, those people are old & odd.
    r = re.compile('(Répondeur)|(Mail)|(Internet)')
    df['ORIGINE_INCIDENT'] = df['ORIGINE_INCIDENT'].replace(
        r, 'indirect_contact')

    # --> INCIDENT_TYPE_NAME
    # Multi Label Binarize & one hot encoder INCIDENT_TYPE_NAME:
    # i.e. from :            to:
    # Dépannage                 1   0
    # Entretien                 0   1
    # Dépannage+Entretien       1   1

    df['INCIDENT_TYPE_NAME'] = df['INCIDENT_TYPE_NAME'].str.split('+')
    mlb = preprocessing.MultiLabelBinarizer()
    df['INCIDENT_TYPE_NAME'] = list(
        mlb.fit_transform(df['INCIDENT_TYPE_NAME']))
    dftmp = pd.DataFrame(
        index=df['INCIDENT_TYPE_NAME'].index,
        data=df['INCIDENT_TYPE_NAME'].values.tolist()).add_prefix(
            'INCIDENT_TYPE_NAME_label')
    df = pd.concat([df.drop(columns=['INCIDENT_TYPE_NAME']), dftmp], axis=1)

    # Categorical features LabelBinarizer (equivalent to onehotencoding):
    msg = 'One Hot Encoding for CATEGORICAL_FEATURES with pd.get_dummies'
    with Timer(msg):
        for col in CATEGORICAL_FEATURES:
            df = pd.concat(
                [pd.get_dummies(df[col], prefix=col),
                 df.drop(columns=[col])],
                axis=1,
            )

    # Still to do, nlp on nlp_cols, but for the moment take the len of the
    # commentary
    nlp_cols = list(set(df.columns).intersection(set(NLP_COLS)))
    for col in nlp_cols:
        newcol = '{}_len'.format(col)
        df[newcol] = df[col].apply(len)
        df[newcol].replace(1, -1)  # considered as NaN

    df = df.drop(columns=nlp_cols)

    return df


CAT_FEATURES_LIST = [['INSTANCE_ID'], ['RESOURCE_ID'],
                     ['INSTANCE_ID', 'RESOURCE_ID']]


def build_features(df, include_hist=False):
    """Build features."""

    if include_hist:
        with Timer("Add history features"):
            dfhist = read_intervention_history()
            for cl in CAT_FEATURES_LIST:
                print('Engineering new categorical features for {}'.format(cl))
                dfhist = build_features_cat(dfhist, cl)

        df = df.merge(dfhist, how="left")

    with Timer("Building timestamp features"):
        df = build_features_datetime(df)

    with Timer("Building str features"):
        df = build_features_str(df)

    # Just encore the rest
    with Timer("Encoding labels"):
        label_cols = list(set(df.columns).intersection(set(LABEL_COLS)))
        label_encoder = LabelEncoderByColMissVal(columns=label_cols)
        df = label_encoder.fit_transform(df)

    to_drop = list(set(TIMESTAMP_COLS).intersection(set(df.columns)))
    df = df.drop(columns=to_drop)

    return df


def build_features_cat(df, filter, dfhist):
    def _mean_delta_interv(df):
        _df = df.dropna(
            subset=[
                'SCHEDULED_START_DATE', 'SCHEDULED_END_DATE',
                'ACTUAL_START_DATE', 'ACTUAL_END_DATE'
            ],
            how='all')

        d_min = pd.concat(
            [_df['ACTUAL_START_DATE'], _df['SCHEDULED_START_DATE']]).min()
        d_max = pd.concat([_df['ACTUAL_END_DATE'],
                           _df['SCHEDULED_END_DATE']]).max()

        n = len(_df)
        if n > 1:
            delta = int(((d_max - d_min) / n).days / 30)
        else:
            delta = 0
        return delta

    new_feature1 = df.groupby(filter).apply(_mean_delta_interv).reset_index()\
        .rename({0: 'DELTA_INTERV_' + '_'.join(filter)})
    new_feature2 = df.groupby(filter).apply('count').reset_index() \
        .rename({0: 'COUNT_' + '_'.join(filter)})
    df = df.merge(new_feature1, on=filter).merge(new_feature2, on=filter)

    return df


# Methods to generate cleaned datas:
def _generate_cleaned_single_set(dataset, drop_cols=None, include_hist=False):
    """Generate one cleaned set amon ['train', 'test']"""
    with Timer("Reading {} set".format(dataset)):
        df = pd.read_parquet(DATA_DIR / "{}.parquet.gzip".format(dataset))

    df = build_features(df, include_hist=include_hist)

    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    return df


def generate_cleaned_sets(drop_cols=DROPCOLS, include_hist=False):
    """Generate cleaned sets."""
    with Timer("Gen clean trainset", True):
        df_train = _generate_cleaned_single_set(
            dataset="train", drop_cols=drop_cols, include_hist=include_hist)
        savepath = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format('train')
        with Timer("Saving into {}".format(savepath)):
            df_train.to_parquet(savepath, compression="gzip")

    with Timer("Gen clean testset", True):
        df_test = _generate_cleaned_single_set(
            dataset="test", drop_cols=drop_cols, include_hist=False)

    if include_hist:
        for cat in CAT_FEATURES_LIST:
            name1 = 'DELTA_INTERV_' + '_'.join(cat)
            name2 = 'COUNT_' + '_'.join(cat)
            df_test = df_test.merge(df_train[[name1, name2]], on=cat)

    savepath = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format('test')
    with Timer("Saving into {}".format(savepath)):
        df_train.to_parquet(savepath, compression="gzip")


class HomeServiceDataHandle:
    train_parquetpath = DATA_DIR / "train.parquet.gzip"
    test_parquetpath = DATA_DIR / "test.parquet.gzip"

    dftrain = None
    dftest = None

    def __init__(self,
                 debug=True,
                 drop_lowimp_features=False,
                 mode="validation"):
        self.debug = debug
        self.mode = mode
        self.drop_lowimp_features = drop_lowimp_features

    @staticmethod
    def _generate_catboost_df(df):
        cols = df.columns.tolist()

        if 'target' in cols:
            cols.remove('target')

        catboost_features = list(
            set(cols).intersection(set(CATBOOST_FEATURES)))
        other_cols = list(set(df.columns.tolist()) - set(catboost_features))

        # Reorder:
        df = df.loc[:, catboost_features + other_cols].copy()

        # Convert into integers without nans:
        for col in catboost_features:
            maxcol = df[col].max()
            maxcol = maxcol if maxcol is not np.nan else 0  # case only nan
            df.loc[:, col] = df.loc[:, col].fillna(maxcol)
            df.loc[:, col] = df[col].astype(int)

        return df, catboost_features

    # Methods to get cleaned datas:
    def _get_cleaned_single_set(self, dataset="train"):
        with Timer("Reading train set"):
            pathdata = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format(
                dataset)
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

    def get_test_set(self, as_cgb_pool=False):
        df = self._get_cleaned_single_set(dataset="test")

        if self.drop_lowimp_features:
            print('Dropping low importance features !')
            dropcols = set(df.columns.tolist()).intersection(
                set(LOW_IMPORTANCE_FEATURES))
            df = df.drop(columns=list(dropcols))

        if as_cgb_pool:
            with Timer('Creating Pool for Test set CatBoost'):
                df, catboost_features = self._generate_catboost_df(df)
                idx_cat_features = list(range(len(catboost_features)))
                pool = cgb.Pool(
                    data=df, label=None, cat_features=idx_cat_features)
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
            train_cols = [
                col for col in train_cols if col not in LOW_IMPORTANCE_FEATURES
            ]

        if as_xgb_dmatrix:
            return xgb.DMatrix(data=df[train_cols], label=df[["target"]])
        elif as_lgb_dataset:
            return lgb.Dataset(df[train_cols], df[["target"]].values.ravel())
        elif as_cgb_pool:
            with Timer('Creating Pool for Train set CatBoost'):
                df, catboost_features = self._generate_catboost_df(df)
                idx_cat_features = list(range(len(catboost_features)))
                pool = cgb.Pool(
                    df.drop(columns=["target"]),
                    df["target"],
                    idx_cat_features)
            return pool
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
            with Timer('Creating Pool for Train&Test set CatBoost'):
                Xtrain, catboost_features = self._generate_catboost_df(Xtrain)
                Xtest, catboost_features_bis = self._generate_catboost_df(
                    Xtest)
                assert catboost_features == catboost_features_bis
                idx_cat_features = list(range(len(catboost_features)))
                pool_train = cgb.Pool(Xtrain, ytrain, idx_cat_features)
                pool_test = cgb.Pool(Xtest, ytest, idx_cat_features)
            return (pool_train, pool_test)
        else:
            return Xtrain, Xtest, ytrain, ytest
