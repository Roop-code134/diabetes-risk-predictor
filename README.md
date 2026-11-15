# Diabetes Risk Predictor

A machine-learning web app that estimates diabetes risk from a few health measurements and explains which factors drove each score.

**Live demo:** https://diabetes-risk-predictor-1-jnin.onrender.com
**API docs:** https://diabetes-risk-predictor-tcuc.onrender.com/docs

> The API runs on a free server that sleeps when idle, so the first request can take 30 to 60 seconds.

**Educational demo only. This is not a medical diagnosis.**

## How it works

1. A logistic regression model is trained on the Pima Indians Diabetes dataset (`train.py`).
2. A FastAPI backend (`backend/main.py`) loads the saved model and exposes `/predict`, `/health` and `/model-info`.
3. A JavaScript frontend (`frontend/index.html`) sends patient values to the API and shows the risk score, a gauge, and the top factors behind the score.

## Model and results

Three models were compared with 5-fold cross-validation (logistic regression, random forest, gradient boosting). Logistic regression scored best on ROC-AUC, and it is also the easiest to explain.

In a medical setting, missing a sick patient is worse than a false alarm. So instead of the default 0.5 cut-off, the decision threshold was lowered to **0.4**, chosen on training data only so that at least 80% of diabetic patients are caught.

Results on a held-out test set of 154 patients:

| Metric | Value |
|---|---|
| ROC-AUC | 0.813 |
| Recall (diabetic patients caught) | 0.833 (45 of 54) |
| Precision | 0.57 |
| Accuracy | 0.721 |
| False alarms | 34 of 100 healthy patients |

The test set is small, so these numbers are rough estimates.

## Data cleaning

In this dataset, zeros in glucose, blood pressure, skin thickness, insulin and BMI mean "not measured." For example, 374 of 768 insulin values are zero. These zeros were treated as missing values and filled with the training-set median inside the model pipeline, so no test data leaks into training.

## Explainability

For logistic regression, each feature's effect on the score can be computed exactly (scaled value times coefficient). The API returns the top four factors for every prediction. Overall, glucose is the strongest driver, followed by BMI, pregnancies, family history, and age.

## API validation and tests

Inputs are validated with Pydantic, so impossible values (for example age 5 or glucose 5000) are rejected with a 422 error. Optional measurements can be left blank. `test_api.py` has 10 tests covering valid requests, the threshold, medical sanity checks (higher glucose raises risk), and rejected inputs. Run them with `python -m pytest -v`.

## Limitations

- Small dataset: 768 patients.
- Only women aged 21 and over of Pima Indian heritage, so the model may not generalize to other groups.
- Insulin and skin thickness are mostly missing, so the model learns little from them.
- The score is a model output, not a clinical probability.

## Tech stack

Python, pandas, scikit-learn, FastAPI, Pydantic, JavaScript, Render (API), GitHub Pages (frontend)

## Run locally

```
pip install -r requirements.txt
python train.py
python -m uvicorn backend.main:app --reload
```

Then open `frontend/index.html` and set `API_URL` to `http://127.0.0.1:8000`.
