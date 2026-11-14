from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

HIGH = {
    "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
    "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
}
LOW = {
    "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
    "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31,
}
REQUIRED_ONLY = {"Pregnancies": 1, "DiabetesPedigreeFunction": 0.4, "Age": 40}


def prob(patient):
    return client.post("/predict", json=patient).json()["probability"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_valid_patient_returns_full_response():
    r = client.post("/predict", json=HIGH)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert len(body["top_factors"]) == 4
    assert "disclaimer" in body


def test_flag_matches_threshold():
    body = client.post("/predict", json=HIGH).json()
    assert body["flagged"] == (body["probability"] >= body["threshold"])


def test_high_risk_patient_scores_above_low_risk_patient():
    assert prob(HIGH) > prob(LOW)


def test_higher_glucose_raises_risk():
    low_glucose = {**LOW, "Glucose": 85}
    high_glucose = {**LOW, "Glucose": 180}
    assert prob(high_glucose) > prob(low_glucose)


def test_missing_optional_values_still_work():
    r = client.post("/predict", json=REQUIRED_ONLY)
    assert r.status_code == 200


def test_rejects_age_below_21():
    assert client.post("/predict", json={**HIGH, "Age": 5}).status_code == 422


def test_rejects_impossible_glucose():
    assert client.post("/predict", json={**HIGH, "Glucose": 5000}).status_code == 422


def test_rejects_negative_pregnancies():
    assert client.post("/predict", json={**HIGH, "Pregnancies": -1}).status_code == 422


def test_rejects_missing_required_field():
    bad = {k: v for k, v in HIGH.items() if k != "Age"}
    assert client.post("/predict", json=bad).status_code == 422