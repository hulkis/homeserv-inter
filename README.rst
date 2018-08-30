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


Todo
----

hyperopt: https://www.kaggle.com/eikedehling/tune-and-compare-xgb-lightgbm-rf-with-hyperopt


> homeserv lgb validate --debug=False --num-boost-round=10000 --early-stopping-rounds=100

[4751]  training's xentropy: 0.33594    training's roc_auc_score: 0.938581      valid_1's xentropy: 0.479222    valid_1's roc_auc_score: 0.751526


Refs
----

- Solution of winner classification kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557

- Hot Encoder for one column to nclass columns with 0 - 1 :
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer
