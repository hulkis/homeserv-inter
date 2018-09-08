===============================
homeserv-inter
===============================

.. image:: https://travis-ci.org/gjeusel/homeserv-inter.svg?branch=master
    :target: https://travis-ci.org/gjeusel/homeserv-inter
.. image:: https://readthedocs.org/projects/homeserv-inter/badge/?version=latest
   :target: http://homeserv-inter.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://coveralls.io/repos/github/gjeusel/homeserv-inter/badge.svg
   :target: https://coveralls.io/github/gjeusel/homeserv-inter
.. image:: https://badge.fury.io/py/homeserv-inter.svg
   :target: https://pypi.python.org/pypi/homeserv-inter/
   :alt: Pypi package


Engie datascience challenge for Home Service Intervention Prediction

:License: MIT license
:Documentation: http://homeserv-inter.readthedocs/en/latest
:Source: https://github.com/gjeusel/homeserv-inter


Installation
------------

.. code:: bash

    pip install homeserv-inter


Command Line
------------

.. code:: bash

    # HyperOpt tuning
    homeserv lgb tuning --debug=False --max-evals=100  # 100 CV of 5 folds

    # sklearn random gridsearch
    homeserv lgb params-tuning-sklearn --debug=False --num-boost-round=10000 --early-stopping-rounds=100

    # Validate 80% train 20% test on best params (& store model)
    homeserv lgb validate --debug=False --num-boost-round=10000 --early-stopping-rounds=100



Results
-------

- On tag v0.0.1:

  .. code:: bash

    homeserv lgb validate --debug=False --num-boost-round=10000 --early-stopping-rounds=100
    #[4751]  training's xentropy: 0.33594    training's roc_auc_score: 0.938581      valid_1's xentropy: 0.479222    valid_1's roc_auc_score: 0.751526


- On tag v0.0.2:

  .. code:: bash

    > homeserv lgb validate --debug=False --num-boost-round=10000 --early-stopping-rounds=100
    #[5218]  training's auc: 0.981714        valid_1's auc: 0.752357
    #Public Score: 0.7181597753  # using 80% datas for train
    #Public Score: 0.7205135095  # using 100% datas for train

- On tag v0.0.4:

  .. code:: bash

    > homeserv lgb validate --debug=False --num-boost-round=10000 --early-stopping-rounds=100
    #[3204]  training's auc: 0.913384        valid_1's auc: 0.753298

    > homeserv lgb cv --debug=False --num-boost-round=10000 --early-stopping-rounds=100 --drop-lowimp-features=True
    # [5175]  cv_agg's auc: 0.753022 + 0.00132367

    > homeserv lgb generate-submit --debug=False --num-boost-round=4000 --drop-lowimp-features=True
    # Public score: 0.7348420163  # using 100% datas for train



Refs
----

- `Solution of winner classification kaggle <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557>`_

- `Hot Encoder for one column to nclass columns with 0 - 1 <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer>`_

- `Hot Encoder with only pandas methods <http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html>`_

- `HyperOpt doc on FMin <https://github.com/hyperopt/hyperopt/wiki/FMin>`_

- `HyperOpt simple example usage <https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt>`_

- `Convert 2 Address into distances in python <https://www.woosmap.com/blog/bulk-geocoding-google-api-geopy/>`_

- `Multi class string <https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f>`_

Remarks
-------
Hard deps on networkx (dependency of hyperopt) due to https://github.com/hyperopt/hyperopt/issues/325
