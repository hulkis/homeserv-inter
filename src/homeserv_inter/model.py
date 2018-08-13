import lightgbm as lgb
from homeserv_inter.datahandler import HomeServiceDataHandle
from sklearn import metrics, model_selection
from wax_toolbox import Timer

params_best_fit = {
    "task": "train",
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
}

class HomeService(HomeServiceDataHandle):

    def _best_params_discovery(self):
        df, label_encoders = self._get_formatted_datas()

        param_dist = {
            "learning_rate": [0.08, 0.1, 0.15],
            "max_depth": [20, 25, 30],
            "num_leaves": [1000, 1200, 1400],
            "n_estimators": [100, 200, 300],
            # 'min_data_in_leaf': [800],
        }

        model = lgb.LGBMClassifier(
            silent=False, objective="binary", n_jobs=4, random_state=42,
            verbose=-1,
        )

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        with Timer("Randomized Search Cross Validation", True):
            rand_grid_search = model_selection.RandomizedSearchCV(
                model, param_distributions=param_dist, n_iter=30
            )
            rand_grid_search.fit(
                df.drop(columns=["target"]), df[["target"]].values.ravel()
            )

        return rand_grid_search

        # self.Xtrain, self.Xtest, self.ytrain, self.ytest = model_selection.train_test_split(
        #     dftrain.drop(columns=["target"]),
        #     dftrain[["target"]],
        #     test_size=split_perc,
        #     random_state=42,
        # )