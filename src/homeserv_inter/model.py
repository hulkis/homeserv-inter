import warnings

import lightgbm as lgb
from sklearn import metrics, model_selection

from homeserv_inter.datahandler import HomeServiceDataHandle
from wax_toolbox import Timer

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


class LgbHomeService(HomeServiceDataHandle):
    params_best_fit = {
        # "task": "train",
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": {"binary_logloss", "auc"},
        # "learning_rate": 0.08,
        # "max_depth": 30,
        # "n_estimators": 300,
        # "num_leaves": 1400,
        # "feature_fraction": 0.9,
        # "bagging_fraction": 0.8,
        # "bagging_freq": 5,
        "verbose": -1,
        "nthreads": 4,
    }

    def validate(self, early_stopping_rounds=20):
        dtrain, dtest = self.get_train_valid_set(as_lgb_dataset=True)
        return lgb.train(
            params=self.params_best_fit,
            train_set=dtrain,
            valid_sets=[dtest],
            early_stopping_rounds=early_stopping_rounds,
        )

    def cv(self, nfolds=5, early_stopping_rounds=10):
        dtrain = self.get_train_set(as_lgb_dataset=True)
        return lgb.cv(
            params=self.params_best_fit,
            train_set=dtrain,
            early_stopping_rounds=early_stopping_rounds,
        )


class HomeService(HomeServiceDataHandle):
    def best_params_discovery(self):
        df, label_encoders = self._get_formatted_datas()

        param_dist = {
            "learning_rate": [0.08, 0.1, 0.15],
            "max_depth": [20, 25, 30],
            "num_leaves": [1000, 1200, 1400],
            "n_estimators": [100, 200, 300],
            # 'min_data_in_leaf': [800],
        }

        model = lgb.LGBMClassifier(
            silent=False, objective="binary", n_jobs=4, random_state=42, verbose=-1
        )

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        with Timer("Randomized Search Cross Validation", True):
            rand_grid_search = model_selection.RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=30,
                scoring=["roc_auc", "accuracy"],
                refit="roc_auc",
                cv=5,
                verbose=2,
            )
            rand_grid_search.fit(
                df.drop(columns=["target"]), df[["target"]].values.ravel()
            )

        return rand_grid_search
