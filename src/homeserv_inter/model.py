import warnings

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn import metrics, model_selection

from homeserv_inter.datahandler import HomeServiceDataHandle
from wax_toolbox import Timer

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


# Notes:
# 3 times more of class 0 than class 1
# Should look up https://arxiv.org/abs/1608.04802 for trying to optimize AUC


# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def my_lgb_loglikelood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


def my_lgb_roc_auc_score(y_pred, y_true):
    # Example custom objective & metric:
    # https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py

    # self-defined eval metric
    # f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool

    res = metrics.roc_auc_score(y_true=y_true.label, y_score=y_pred, average="macro")
    return "roc_auc_score", res, True


def my_xgb_roc_auc_score(y_pred, y_true):
    # https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py
    res = metrics.roc_auc_score(
        y_true=y_true.get_label(), y_score=y_pred, average="macro"
    )
    return "roc-auc-score", res


class LgbHomeService(HomeServiceDataHandle):
    params_best_fit = {
        # "task": "train",
        "boosting_type": "gbdt",
        "objective": "xentropy",
        "is_unbalance": True,
        # "metric": {"binary_error"},  # same as accuracy (% time predicted was wrong)
        "num_leaves": 100,
        "learning_rate": 0.04,
        "bagging_fraction": 0.95,
        "feature_fraction": 0.98,
        "bagging_freq": 6,
        "max_depth": -1,
        # 'max_bin': 511,
        # 'min_data_in_leaf': 20,
        "verbose": -1,
        "nthreads": 4,
    }

    hyperopt_space = {
        "n_estimators": hp.quniform("n_estimators", 25, 500, 25),
        "max_depth": hp.quniform("max_depth", 1, 10, 1),
    }

    # Used by RandomizedSearchCV
    params_discovery = {
        "learning_rate": [0.001, 0.04, 0.08, 0.1, 0.15],
        "max_depth": [5, 10, 15, 20, 25, 30],
        "num_leaves": [200, 400, 500, 700, 900, 1200, 1400],
        # 'min_data_in_leaf': [800],
    }

    def validate(self, **kwargs):
        dtrain, dtest = self.get_train_valid_set(as_lgb_dataset=True)
        watchlist = [dtrain, dtest]

        booster = lgb.train(
            params=self.params_best_fit,
            feval=my_lgb_roc_auc_score,
            # fobj=my_lgb_loglikelood,
            train_set=dtrain,
            valid_sets=watchlist,
            # learning_rates=lambda iter: 0.05 * (0.99 ** iter),
            **kwargs,
        )
        return

    def cv(self, nfolds=5, **kwargs):
        dtrain = self.get_train_set(as_lgb_dataset=True)
        return lgb.cv(
            params=self.params_best_fit,
            train_set=dtrain,
            **kwargs,
        )

    def params_tuning_sklearn(self):
        Xtrain, ytrain = self.get_train_set(as_lgb_dataset=False)  # False here !

        model = lgb.LGBMClassifier(
            silent=False, objective="binary", n_jobs=4, random_state=42, verbose=-1
        )

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        with Timer("Randomized Search Cross Validation", True):
            rand_grid_search = model_selection.RandomizedSearchCV(
                model,
                param_distributions=self.params_discovery,
                n_iter=50,
                scoring=["roc_auc", "accuracy"],
                refit="roc_auc",
                cv=5,
                verbose=2,
            )
            rand_grid_search.fit(Xtrain, ytrain.values.ravel())

        print("Best Params: {}".format(rand_grid_search.best_params_))
        return rand_grid_search

    def params_tuning_hyperopt(self):
        pass


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
            feval=my_xgb_roc_auc_score,
            maximize=True,  # whether to maximize feval
            dtrain=dtrain,
            evals=watchlist,
            **kwargs,
        )
        return

    def cv(self, nfolds=5, **kwargs):
        dtrain = self.get_train_set(as_xgb_dataset=True)
        return xgb.cv(
            params=self.params_best_fit,
            train_set=dtrain,
            **kwargs,
        )
