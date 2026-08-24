"""
RETIRED -- do not run. Kept only so the filename is not silently reused.

This trained the two original scikit-learn artifacts:

    policy_fit.joblib          replaced by Model 1, the guideline reasoning
                               service. Nothing loads this file any more.
    appeal_propensity.joblib   replaced by the regressor bundle written by
                               ml/train_appeal.py -- SAME FILENAME.

That last point is why this script is disabled rather than deleted. It wrote its
classifier to ml/models/appeal_propensity.joblib, which is now the regressor
bundle Model 2 loads. Running the old script would overwrite a working model with
one the serving layer rejects, and the failure would not appear until the next
denial reached Model 2.

To train Model 2:

    python ml/train_appeal.py --csv data/appeals_prediction_transformed.csv
"""
import sys

MESSAGE = """
ml/train.py is retired.

  policy_fit.joblib        -> Model 1 (the guideline service) replaced it.
  appeal_propensity.joblib -> now written by ml/train_appeal.py.

Running this script would overwrite the appeal regressor with the old
classifier and break Model 2. To train the appeal model:

  python ml/train_appeal.py --csv data/appeals_prediction_transformed.csv
"""

if __name__ == "__main__":
    print(MESSAGE)
    sys.exit(1)
