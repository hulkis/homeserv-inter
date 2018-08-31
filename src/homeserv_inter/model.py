import pickle
import warnings
from copy import deepcopy

import hyperopt
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn import model_selection

from homeserv_inter.constants import LABEL_COLS, MODEL_DIR, RESULT_DIR, TUNING_DIR
from homeserv_inter.datahandler import HomeServiceDataHandle
from homeserv_inter.tuning import HyperParamsTuning
from wax_toolbox import Timer

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


class LgbHomeService(HomeServiceDataHandle, HyperParamsTuning):

    dtrain = None

    # Common params for LightGBM
    common_params = {
        "verbose": -1,
        "nthreads": 4,
        "is_unbalance": True,  # unbalanced classes
        "objective": "xentropy",  # better optimize on cross-entropy loss for auc
        "metric": {"auc"},  # alias for roc_auc_score
    }

    # Best fit params
    params_best_fit = {
        # "task": "train",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 160,
        "min_data_in_leaf": 10,
        "max_depth": -1,
        "bagging_fraction": 0.926,
        "feature_fraction": 0.936,
        "bagging_freq": 9,
        # 'max_bin': 511,
        # 'min_data_in_leaf': 20,
        **common_params,
    }

    # Tuning attributes in relation to HyperParamsTuning
    int_params = ("num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq")
    float_params = ("learning_rate", "feature_fraction", "bagging_fraction")
    hypertuning_space = {
        "boosting": hyperopt.hp.choice("boosting", ["gbdt", "rf"]),  # , "dart"]),
        "num_leaves": hyperopt.hp.quniform("num_leaves", 30, 300, 20),
        "min_data_in_leaf": hyperopt.hp.quniform("min_data_in_leaf", 10, 200, 10),
        # "learning_rate": hyperopt.hp.uniform("learning_rate", 0.001, 0.1),
        "feature_fraction": hyperopt.hp.uniform("feature_fraction", 0.7, 0.99),
        "bagging_fraction": hyperopt.hp.uniform("bagging_fraction", 0.7, 0.99),
        "bagging_freq": hyperopt.hp.quniform("bagging_freq", 6, 10, 1),
    }

    @property
    def now(self):
        return pd.Timestamp.now(tz="CET")

    @property
    def nowstr(self):
        return self.now.strftime("%d-%Hh-%mm")

    @classmethod
    def save_model(self, booster):
        f = MODEL_DIR / "lgb_model_{}.txt".format(self.nowstr)
        booster.save_model(f.as_posix())

    def validate(self, save_model=True, **kwargs):
        dtrain, dtest = self.get_train_valid_set(as_lgb_dataset=True)

        cols = dtrain.data.columns.tolist()
        categorical_feature = list(set(cols).intersection(LABEL_COLS))

        watchlist = [dtrain, dtest]

        booster = lgb.train(
            params=self.params_best_fit,
            categorical_feature=categorical_feature,
            # feval=my_lgb_roc_auc_score,
            # fobj=my_lgb_loglikelood,
            train_set=dtrain,
            valid_sets=watchlist,
            # so that at 3000th iteration, learning_rate=0.025
            # learning_rates=lambda iter: 0.5 * (0.999 ** iter),
            **kwargs,
        )

        if save_model:
            self.save_model(booster)

        return booster

    def cv(
        self,
        params_model=None,
        nfolds=5,
        num_boost_round=10000,
        early_stopping_rounds=100,
        **kwargs,
    ):

        if self.dtrain is None:
            self.dtrain = self.get_train_set(as_lgb_dataset=True)

        cols = self.dtrain.data.columns.tolist()
        categorical_feature = list(set(cols).intersection(LABEL_COLS))

        # If no params_model is given, take self.params_best_fit
        if params_model is None:
            params_model = self.params_best_fit

        eval_hist = lgb.cv(
            params=params_model,
            categorical_feature=categorical_feature,
            # feval=my_lgb_roc_auc_score,
            train_set=self.dtrain,
            verbose_eval=True,  # display the progress
            show_stdv=True,  # display the standard deviation in progress, results are not affected
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )

        return eval_hist

    def params_tuning_sklearn(self, **kwargs):
        Xtrain, ytrain = self.get_train_set(as_lgb_dataset=False)  # False here !

        # Used by RandomizedSearchCV
        params_discovery = {
            "num_leaves": [50, 70, 90, 110, 150, 200, 250, 300],
            "max_bin": [200, 255, 300],
            "min_data_in_leaf": [20, 30, 40, 100],
            "lambda": [0.3, 0.6, 1],  # regularization
        }

        # Get classical parameters for model
        params_fit = deepcopy(self.params_best_fit)
        for k in params_discovery.keys():
            if k in params_fit:
                params_fit.pop(k)

        model = lgb.LGBMClassifier(**params_fit, learning_rate=0.04, **kwargs)

        # Define the grid
        rand_grid_search = model_selection.RandomizedSearchCV(
            model,
            param_distributions=self.params_discovery,
            n_iter=100,  # trade off quality of result / speed
            scoring=[
                "accuracy",
                "roc_auc",
            ],  # evaluate those predictions on the test set
            refit="roc_auc",  # refit on roc_auc metric
            n_jobs=1,  # already running in parallel for lgb
            cv=5,  # cv of 5 folds
            verbose=2,  # make it verbose
        )

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        with Timer("Randomized Search Cross Validation", at_enter=True):
            rand_grid_search.fit(X=Xtrain, y=ytrain.values.ravel(), groups=None)

        print("Best Params: {}".format(rand_grid_search.best_params_))
        return rand_grid_search

    def hypertuning_objective(self, params):
        params = self._ensure_type_params(params)
        msg = "-- HyperOpt -- CV with {}\n".format(params)
        params = {**self.common_params, **params}  # recombine with common params

        # Fix learning rate:
        params["learning_rate"] = 0.04

        with Timer(msg, at_enter=True):
            eval_hist = self.cv(params_model=params, nfolds=5)

        score = max(eval_hist["auc-mean"])

        result = {
            "loss": score,
            "status": hyperopt.STATUS_OK,
            # -- store other results like this
            # "eval_time": time.time(),
            # 'other_stuff': {'type': None, 'value': [0, 1, 2]},
            # -- attachments are handled differently
            "attachments": {"eval_hist": eval_hist},
        }

        return result

    def generate_submit(self, from_model_saved=False):

        if not from_model_saved:
            dtrain = self.get_train_set(as_lgb_dataset=True)

            booster = lgb.train(
                params=self.params_best_fit, train_set=dtrain, num_boost_round=4650
            )

            self.save_model(booster)

        else:
            booster = lgb.Booster(model_file=from_model_saved)

        df = self.get_test_set()
        with Timer("Predicting"):
            pred = booster.predict(df)

        df = pd.DataFrame({"target": pred})
        df.to_csv(RESULT_DIR / "submit_{}.csv".format(self.nowstr), index=False)

    def get_df_importance(self, booster):
        idx = booster.feature_name()
        arr = booster.feature_importance()
        dfimp = pd.DataFrame(index=idx, data=arr, columns=["importance"])
        return dfimp.sort_values("importance", ascending=False)


class XgbHomeService(HomeServiceDataHandle):
    # https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    params_best_fit = {
        "nthread": 4,
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "error",
        "max_depth": 10,
        "eta": 0.1,  # analogous to learning_rate
        # "gamma": 0.015,
        # "subsample": max(min(subsample, 1), 0),
        # "colsample_bytree": max(min(colsample_bytree, 1), 0),
        # "min_child_weight": min_child_weight,
        # "max_delta_step": int(max_delta_step),
        "silent": True,
    }

    def validate(self, **kwargs):
        dtrain, dtest = self.get_train_valid_set(as_xgb_dmatrix=True)
        watchlist = [(dtrain, "train"), (dtest, "eval")]

        booster = xgb.train(
            params=self.params_best_fit,
            # feval=my_xgb_roc_auc_score,
            maximize=True,  # whether to maximize feval
            dtrain=dtrain,
            evals=watchlist,
            **kwargs,
        )
        return

    def cv(self, nfolds=5, **kwargs):
        dtrain = self.get_train_set(as_xgb_dataset=True)
        return xgb.cv(params=self.params_best_fit, train_set=dtrain, **kwargs)
