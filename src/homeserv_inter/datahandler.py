import re

import catboost as cgb
import category_encoders as ce
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing

from homeserv_inter.constants import (
    CATBOOST_FEATURES, CLEANED_DATA_DIR, DATA_DIR, DROPCOLS, HIGH_NUM_CAT,
    LABEL_COLS, LOW_IMPORTANCE_FEATURES, MEDIUM_NUM_CAT, NLP_COLS,
    RAW_DATA_DIR, SEED, TIMESTAMP_COLS)
from wax_toolbox.profiling import Timer

# Feature ing.:

DEFAULT_SUBSET_HIST = [
    'SCHEDULED_START_DATE',
    'SCHEDULED_END_DATE',
    'ACTUAL_START_DATE',
    'ACTUAL_END_DATE',
    "INSTANCE_ID",
    "RESOURCE_ID",
]


def read_intervention_history(debug=False, subset=DEFAULT_SUBSET_HIST):
    with Timer("Reading intervention_history.csv"):
        df = pd.read_csv(
            RAW_DATA_DIR / "intervention_history.csv",
            sep="|",
            encoding="Latin-1",
            nrows=20000 if debug else None,
        )
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


CAT_FEATURES_LIST = [['INSTANCE_ID'], ['RESOURCE_ID'],
                     ['INSTANCE_ID', 'RESOURCE_ID']]


class HomeServiceCleanedData:
    def __init__(self,
                 debug=False,
                 label_encode=True,
                 hash_encode=False,
                 include_hist=False):
        self.debug = debug
        self.label_encode = label_encode
        self.hash_encode = hash_encode
        self.include_hist = include_hist

    @staticmethod
    def _build_features_datetime(df):
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

    @staticmethod
    def _build_features_str(
            df,
            modify_FORMULE=True,
            modify_INCIDENT_TYPE_NAME=True,
    ):
        # Some Str cleaning:

        if modify_FORMULE:
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
            df['ORIGINE_INCIDENT'] = df['ORIGINE_INCIDENT'].replace(
                'Fax', np.nan)

            # treat 'Répondeur', 'Mail', 'Internet' as one label: 'indirect_contact'
            # but still keep 'Courrier' as it is soooo mainstream, those people are old & odd.
            r = re.compile('(Répondeur)|(Mail)|(Internet)')
            df['ORIGINE_INCIDENT'] = df['ORIGINE_INCIDENT'].replace(
                r, 'indirect_contact')

        if modify_INCIDENT_TYPE_NAME:
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
            df = pd.concat(
                [df.drop(columns=['INCIDENT_TYPE_NAME']), dftmp], axis=1)

        # Some value counts and rewrite those too rare:
        with Timer('Replacing rare RUE'):
            to_replace = (df['RUE'].value_counts() <= 10).index
            df['RUE'] = df['RUE'].replace(to_replace, 'KitKatIsGood')

        with Timer('Replacing rare RESOURCE_ID'):
            to_replace = (df['RESOURCE_ID'].value_counts() <= 10).index
            df['RESOURCE_ID'] = df['RESOURCE_ID'].replace(to_replace, -1)

        with Timer('Replacing rare VILLE'):
            to_replace = (df['VILLE'].value_counts() <= 10).index
            df['VILLE'] = df['VILLE'].replace(to_replace, 'WonderLand')

        with Timer('Replacing rare MARQUE_LIB'):
            to_replace = (df['MARQUE_LIB'].value_counts() <= 25).index
            df['MARQUE_LIB'] = df['MARQUE_LIB'].replace(to_replace, 'BrandOfChocolat')

        with Timer('Replacing rare TYPE_VOIE'):
            to_replace = (df['TYPE_VOIE'].value_counts() <= 25).index
            df['TYPE_VOIE'] = df['TYPE_VOIE'].replace(to_replace, 'Stairway to hell')

        with Timer('Replacing rare OPTION'):
            to_replace = (df['OPTION'].value_counts() <= 5).index
            df['OPTION'] = df['OPTION'].replace(to_replace, 'NoNeedForIt')

        with Timer('Replacing rare CODE_GEN_EQUIPEMENT'):
            to_replace = (df['CODE_GEN_EQUIPEMENT'].value_counts() <= 5).index
            df['CODE_GEN_EQUIPEMENT'] = df['CODE_GEN_EQUIPEMENT'].replace(
                to_replace, 'SuperCode')

        # Still to do, nlp on nlp_cols, but for the moment take the len of the
        # commentary
        nlp_cols = list(set(df.columns).intersection(set(NLP_COLS)))
        for col in nlp_cols:
            newcol = '{}_len'.format(col)
            df[newcol] = df[col].apply(len)
            df[newcol].replace(1, -1)  # considered as NaN

        df = df.drop(columns=nlp_cols)

        return df

    @staticmethod
    def _build_features_cat(df, filter):
        def _mean_delta_interv(df):
            _df = df.dropna(
                subset=[
                    'SCHEDULED_START_DATE', 'SCHEDULED_END_DATE',
                    'ACTUAL_START_DATE', 'ACTUAL_END_DATE'
                ],
                how='all')

            d_min = pd.concat(
                [_df['ACTUAL_START_DATE'], _df['SCHEDULED_START_DATE']]).min()
            d_max = pd.concat(
                [_df['ACTUAL_END_DATE'], _df['SCHEDULED_END_DATE']]).max()

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

    def _build_features(self, df):
        """Build features."""

        with Timer("Building timestamp features"):
            df = self._build_features_datetime(df)

        with Timer("Building str features"):
            df = self._build_features_str(df)

        label_cols = list(set(df.columns).intersection(set(LABEL_COLS)))
        df.loc[:, label_cols] = df.loc[:, label_cols].astype(str)

        # # Gives memory error
        # with Timer("Encoding with BackwardDifferenceEncoder"):
        #     backward_diff_cols = list(
        #         set(label_cols).intersection(
        #             set(HIGH_NUM_CAT + MEDIUM_NUM_CAT)))
        #     bd_encoder = ce.backward_difference.BackwardDifferenceEncoder(
        #         cols=backward_diff_cols, verbose=1)
        #     dftmp = bd_encoder.fit_transform(df)

        if self.hash_encode:
            with Timer("Encoding with HashingEncoder"):
                for col in ['RESOURCE_ID', 'RUE', 'PARTY_ID_OCC', 'VILLE']:
                    hash_cols = list(set(label_cols).intersection(set([col])))
                    hash_encoder = ce.HashingEncoder(
                        cols=hash_cols, n_components=15, verbose=1)
                    dftmp = hash_encoder.fit_transform(df)
                    newcols = dftmp.columns.difference(df.columns)
                    dftmp = dftmp[newcols]
                    dftmp.columns = 'hash_{}_'.format(col) + dftmp.columns
                    df = pd.concat([df, dftmp], axis=1)

        if self.label_encode:
            # Forgotten columns at the end, simple Binary Encoding:
            with Timer("Encoding remaning ones in as LabelEncoder"):
                other_cols = df.columns.difference(
                    df._get_numeric_data().columns).tolist()
                le = preprocessing.LabelEncoder()
                for col in other_cols:
                    df.loc[:, col] = le.fit_transform(df[col])

        to_drop = list(set(df.columns).intersection(set(TIMESTAMP_COLS)))
        df = df.drop(columns=to_drop)

        return df

    # Methods to generate cleaned datas:
    def generate_single_set(self, dataset, drop_cols=DROPCOLS):
        """Generate one cleaned set amon ['train', 'test']"""
        with Timer("Reading {} set".format(dataset)):
            df = pd.read_parquet(DATA_DIR / "{}.parquet.gzip".format(dataset))

        if self.debug:
            df = df.sample(n=30000, random_state=SEED)

        df = self._build_features(df)

        if drop_cols is not None:
            to_drop = list(set(df.columns).intersection(set(drop_cols)))
            df = df.drop(columns=to_drop)

        savepath = CLEANED_DATA_DIR / "{}{}_cleaned.parquet.gzip".format(
            'debug_' if self.debug else '', dataset)

        with Timer("Saving into {}".format(savepath)):
            df.to_parquet(savepath, compression="gzip")

    def generate_sets(self, drop_cols=DROPCOLS):
        """Generate cleaned sets."""
        with Timer("Gen clean trainset", True):
            self.generate_single_set(dataset="train", drop_cols=drop_cols)

        with Timer("Gen clean testset", True):
            self.generate_single_set(dataset="test", drop_cols=drop_cols)

    def add_history_to_cleaned(self):
        # Test Cleaned Set:
        dataset = 'test'
        with Timer("Reading train set"):
            pathdata = CLEANED_DATA_DIR / "{}{}_cleaned.parquet.gzip".format(
                'debug_' if self.debug else '', dataset)
            df = pd.read_parquet(pathdata)

        with Timer("Add history features"):
            dfhist = read_intervention_history(debug=self.debug)
            for cl in CAT_FEATURES_LIST:
                print('Engineering new categorical features for {}'.format(cl))
                dfhist = self._build_features_cat(dfhist, cl)

        for cat in CAT_FEATURES_LIST:
            name1 = 'DELTA_INTERV_' + '_'.join(cat)
            name2 = 'COUNT_' + '_'.join(cat)
            df = df.merge(dfhist[[name1, name2]], on=cat)
        __import__('IPython').embed()  # Enter Ipython


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
        catboost_features = sorted(
            catboost_features)  # sort them for next run consistency

        other_cols = list(set(cols) - set(catboost_features))

        # Add back target if needed
        if 'target' in df.columns:
            other_cols = other_cols + ['target']

        # Reorder:
        df = df.loc[:, catboost_features + other_cols].copy()

        # Convert into integers without nans:
        for col in catboost_features:
            if (df[col].dtype <= np.integer) or (df[col].dtype <= np.float):
                maxcol = df[col].max()
                maxcol = maxcol if maxcol is not np.nan else 0  # case only nan
                df.loc[:, col] = df.loc[:, col].fillna(maxcol)
                df.loc[:, col] = df[col].astype(int)

        return df, catboost_features

    # Methods to get cleaned datas:
    def _get_cleaned_single_set(self, dataset="train", replace_minus1=False):
        with Timer("Reading train set"):
            pathdata = CLEANED_DATA_DIR / "{}_cleaned.parquet.gzip".format(
                dataset)
            df = pd.read_parquet(pathdata)
            if self.debug:
                df = df.sample(
                    n=10000, random_state=SEED).dropna(
                        axis=1, how="all")

        if replace_minus1:
            with Timer("Replacing -1 categorical by np.nan"):
                lst_cols = list(
                    set(df.columns.tolist()).intersection(LABEL_COLS))
                for col in lst_cols:
                    df[col] = df[col].replace(-1, np.nan)

        # Consistency, always sort columns with target at the end:
        cols = df.columns.tolist()
        if 'target' in df.columns:
            cols.remove('target')
        sorted_cols = sorted(cols)
        if 'target' in df.columns:
            sorted_cols += ['target']
            df = df.loc[:, sorted_cols]

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
            dropcols = set(df.columns.tolist()).intersection(
                set(LOW_IMPORTANCE_FEATURES))
            df = df.drop(columns=list(dropcols))

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
