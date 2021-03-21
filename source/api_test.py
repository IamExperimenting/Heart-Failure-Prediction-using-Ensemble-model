from fastapi.testclient import TestClient

from api import app

client = TestClient(app)



def test_read_item_bad_token():
    response = client.get("/test/bad_data", headers={"X-Token": "hailhydra"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Token header"}


def test_read_inexistent_item():
    response = client.get("/test/notin", headers={"X-Token": "fastapitest"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Item not found"}


def test_create_item():
    response = client.post(
        "/test/",
        headers={"X-Token": "fastapitest"},
        json={
  "age": 75,
  "anaemia": 0,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000.00,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking": 0,
  "time": 4
},
    )
    assert response.status_code == 200
    assert response.json() == {
  "age": 75,
  "anaemia": 0,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000.00,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking": 0,
  "time": 4
}


def test_create_item_bad_token():
    response = client.post(
        "/test/",
        headers={"X-Token": "hailhydra"},
        json={
  "age": 75,
  "anaemia": 0,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000.00,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking": 0,
  "time": 4
},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Token header"}


def test_create_existing_item():
    response = client.post(
        "/test/",
        headers={"X-Token": "fastapitest"},
        json={
  "age": 75,
  "anaemia": 0,
  "creatinine_phosphokinase": 582,
  "diabetes": 0,
  "ejection_fraction": 20,
  "high_blood_pressure": 1,
  "platelets": 265000.00,
  "serum_creatinine": 1.9,
  "serum_sodium": 130,
  "sex": 1,
  "smoking": 0,
  "time": 4
},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Item already exists"}
