import pickle
import json
import logging

import hyperopt

from homeserv_inter.constants import TUNING_DIR

logger = logging.getLogger(__name__)


class HyperParamsTuning:
    """Base class for hyper parameters tuning using hyperopt."""

    int_params = ()
    float_params = ()

    @property
    def hypertuning_space(self):
        raise NotImplementedError

    def _ensure_type_params(self, params):
        """Sanitize params according to their type."""
        for k in self.int_params:
            if k in params:
                params[k] = int(params[k])

        for k in self.float_params:
            if k in params:
                params[k] = round(params[k], 3)

        return params

    def _hypertuning_save_results(self, best_params, trials):
        # Store eval_hist for best score:
        fpath = TUNING_DIR / "eval_hist_best_score.json"
        print("Saving {}".format(fpath))

        # Best score idx:
        best_score = 0
        for i, d in enumerate(trials.results):
            if best_score < d["loss"]:
                idx = i
            best_score = max(best_score, d["loss"])

        with open(fpath, "w") as file:
            eval_hist = trials.trial_attachments(trials.trials[idx])["eval_hist"]
            file.write(json.dumps(eval_hist))

        fpath = TUNING_DIR / "best_params.json"
        print("Saving {}".format(fpath))
        with open(fpath.as_posix(), "wb") as f:
            json.dump(best_params, f)

        fpath = TUNING_DIR / "all_eval_hist.pkl"
        print("Saving all results in {}".format(fpath))
        with open(fpath.as_posix(), "wb") as f:
            pickle.dump(trials, f)

    def tuning(self, max_evals=3):
        trials = hyperopt.Trials()
        try:
            # https://github.com/hyperopt/hyperopt/wiki/FMin
            best_params = hyperopt.fmin(
                fn=self.hypertuning_objective,
                space=self.hypertuning_space,
                algo=hyperopt.tpe.suggest,
                max_evals=max_evals,
                trials=trials,  # store results
            )
        except Exception as e:
            logger.error(e)
            return

        # Save some results:
        self._hypertuning_save_results(best_params, trials)

        return best_params

    def hypertuning_objective(self, params):
        raise NotImplementedError
