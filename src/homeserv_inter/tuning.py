import logging

import numpy as np
import pandas as pd

import hyperopt
import lightgbm as lgb
from sklearn import metrics, model_selection
from wax_toolbox import Timer

logger = logging.getLogger(__name__)


class HyperParamsTuning:
    """Base class for hyper parameters tuning using hyperopt."""

    int_params = None
    float_params = None

    def __init__(self, df, target_colname, max_evals):
        self.df = df
        self.target_colname = target_colname
        self.max_evals = max_evals

        raise NotImplementedError

    def _sanitize_params(self, params):
        """Sanitize params according to their type."""
        for k in self.int_params:
            if k in params:
                params[k] = int(params[k])

        for k in self.float_params:
            if k in params:
                params[k] = round(params[k], 3)

        return params

    def fit(self):
        try:
            return hyperopt.fmin(
                fn=self.objective,
                space=self.space,
                algo=hyperopt.tpe.suggest,
                max_evals=self.max_evals,
            )
        except Exception as e:
            logger.error(e)
            return

    def objective(self, params):
        raise NotImplementedError


class LGBHyperParamsTuning(HyperParamsTuning):
    """For LightGBM optim only."""

    int_params = ("num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq")
    float_params = ("learning_rate", "feature_fraction", "bagging_fraction")

    space = {
        "boosting": hyperopt.hp.choice("boosting", ["gbdt", "rf", "dart"]),
        "num_leaves": hyperopt.hp.quniform("num_leaves", 10, 800, 20),
        "min_data_in_leaf": hyperopt.hp.quniform("min_data_in_leaf", 10, 200, 10),
        "learning_rate": hyperopt.hp.uniform("learning_rate", 0.001, 0.1),
        "feature_fraction": hyperopt.hp.uniform("feature_fraction", 0.7, 0.99),
        "bagging_fraction": hyperopt.hp.uniform("bagging_fraction", 0.7, 0.99),
        "bagging_freq": hyperopt.hp.quniform("bagging_freq", 6, 10, 1),
    }

    def objective(self, params):
        params = self._sanitize_params(params)

        # Adding some fixed params from models config file:
        lgb_params = self.cfg_models["lgb_params"]
        params.update(
            {
                "verbose": -1,
                "num_threads": lgb_params["num_threads"],
                "objective": lgb_params["objective"],
                "metric": ["l1", "l2_root"],
                "n_estimators": 2000,
            }
        )

        msg = "--------- Cross-Validation ---------\n" "params: {}".format(params)
        logger.info(msg)

        scores = []
        generator = self.tscv.split_df(self.df, self.target_colname)
        for X_train, X_valid, y_train, y_valid in generator:
            lgb_train = lgb.Dataset(X_train.values, y_train.values.ravel())
            lgb_valid = lgb.Dataset(
                X_valid.values, y_valid.values.ravel(), reference=lgb_train
            )

            watchlist = [lgb_train, lgb_valid]
            msg = (
                "Validating on train shape {} | valid shape {} with "
                "params\n{}".format(X_train.shape, X_valid.shape, params)
            )
            with Timer(msg):
                gbm = lgb.train(
                    params,
                    lgb_train,
                    valid_sets=watchlist,
                    valid_names=["train", "valid"],
                    early_stopping_rounds=100,
                    verbose_eval=50,
                )

            scores.append(gbm.best_score["valid"]["rmse"])

        score = np.mean(scores)
        logger.info("rmse score using params={0}: {1:.3f}".format(params, score))

        return {"loss": score, "status": hyperopt.STATUS_OK}


class XGBHyperParamsTuning(HyperParamsTuning):
    """For Xgboost optim only."""

    try:
        import xgboost as xgb
    except ImportError as e:
        logger.warning("Can't import xgboost. Be aware some tools won't works.")

    int_params = ("max_depth", "n_estimators")
    float_params = ("learning_rate",)

    space = {
        "max_depth": hyperopt.hp.quniform("max_depth", 2, 20, 1),
        "n_estimators": hyperopt.hp.quniform("n_estimators", 100, 500, 20),
        "learning_rate": hyperopt.hp.uniform("learning_rate", 0.001, 0.1),
    }

    def objective(self, params):
        params = self._sanitize_params(params)
        model = xgb.XGBRegressor(**params)
        score_func = metrics.make_scorer(
            score_func=self.metric_func, greater_is_better=False
        )

        score = model_selection.cross_val_score(
            model, self.X, self.y, scoring=score_func, cv=self.cv.split(self.X)
        ).mean()

        logger.info(
            "{0} score using params={1}: {2:.3f}".format(
                self.metric_func.__name__, params, score
            )
        )

        return score


def gridsearch_tuning(
    model, X, y, param_dist, randomized_search=True, cv=3, save_aspickle=False
):
    """Params Tuning using sklearn gridsearch.

    Example of param_dist used for LightGBM:

        >>> param_dist = {
            'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'num_leaves': [30, 50, 80, 100, 120],
            'n_estimators': [150, 200, 250, 300],
            'min_data_in_leaf': [800],
            }

    # {'learning_rate': 0.0005, 'n_estimators': 150, 'num_leaves': 3}
    # -4.7981848951598725

    # {'learning_rate': 0.0005, 'n_estimators': 200, 'num_leaves': 3}
    # -4.7975942673274155
    """

    tscv = model_selection.TimeSeriesSplit(n_splits=cv)

    if randomized_search:
        grid_search = model_selection.RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            scoring="neg_mean_absolute_error",  # negative for standarization purpose (we want to achieve the max(metric) in sklearn)
            cv=tscv,
            n_iter=10,
            n_jobs=1,
            verbose=3,
        )
    else:
        grid_search = model_selection.GridSearchCV(
            model,
            param_grid=param_dist,
            scoring="neg_mean_absolute_error",  # negative for standarization purpose (we want to achieve the max(metric) in sklearn)
            cv=tscv,
            n_jobs=1,
            verbose=3,
        )

    with Timer("Randomized Search Cross Validation", True):
        grid_search.fit(X, y.values.ravel())

    if save_aspickle:
        import pickle

        fname = "grid_search_{}_{}.pickle".format(
            model.__class__.__name__, pd.Timestamp.now().strftime("%Y.%m.%d-%Hh%M")
        )
        pickle.dump(grid_search, open(fname, "wb"))

    return grid_search
