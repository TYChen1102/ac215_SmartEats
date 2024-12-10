import pytest
import json
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.service import app, make_prediction, send_output, extract_input_step_2, food_to_nutrition_step_2, transform_data_step_3, send_output_step_3

# Create a test client for the API
client = TestClient(app)

# Mock data
mock_image = b"fake_image_data"
mock_weight = 200.0
mock_food_item = "Burger"
mock_step1_output = {'food': 'Burger', 'Weight': 150}
mock_step2_output = pd.DataFrame({
    'Description': ['Burger'],
    'Energy': ['250 kcal'],
    'Carbohydrate': ['20 g'],
    'Protein': ['10 g'],
    'Fat': ['15 g'],
    'Cosine similarity score': [0.9]
})
mock_step2_output_extracted = {
    'Description': 'Burger',
    'Energy': '250 kcal',
    'Carbohydrate': '20 g',
    'Protein': '10 g',
    'Fat': '15 g',
    'Cosine similarity score': 0.9
}
mock_step3_output = {
    'Obesity': '0.25',
    'Diabetes': '0.15',
    'High Cholesterol': '0.1',
    'Hypertension': '0.2'
}

# Test API endpoint
def test_predict_food_from_image():
    with patch("cv2.imread", return_value="mock_img"), \
         patch("cv2.resize"), \
         patch("numpy.expand_dims"), \
         patch("api.service.make_prediction", return_value=("Burger", 0.95, {})), \
         patch("api.service.extract_input_step_2", return_value=mock_step1_output), \
         patch("api.service.food_to_nutrition_step_2", return_value=mock_step2_output), \
         patch("api.service.extract_input_step_3", return_value=mock_step2_output_extracted), \
         patch("api.service.transform_data_step_3", return_value=mock_step3_output), \
         patch("api.service.joblib.load"), \
         patch("api.service.send_output_step_3"):
        
        response = client.post(
            "/predict",
            files={"file": ("image.png", mock_image)},
            data={"weight": mock_weight}
        )
        assert response.status_code == 200
        assert "meal_info" in response.json()

# Test make_prediction function
def test_make_prediction():
    mock_img = MagicMock()
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.1, 0.9, 0.0]]
    with patch("api.service.model", mock_model):
        label, prob, food_map = make_prediction(mock_img)
        assert label == 'Crispy Chicken'
        assert prob == 0.9

# Test send_output function
def test_send_output():
    with patch("api.service.storage.Client") as mock_storage_client:
        mock_bucket = mock_storage_client.return_value.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        send_output("Burger", 0.95)
        mock_blob.upload_from_filename.assert_called_once_with("step1_output.json")

# Test extract_input_step_2 function
def test_extract_input_step_2():
    with patch("api.service.storage.Client") as mock_storage_client:
        mock_bucket = mock_storage_client.return_value.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        mock_blob.download_as_text.return_value = json.dumps(mock_step1_output)
        result = extract_input_step_2()
        assert result == mock_step1_output

@pytest.fixture
def mock_api_response():
    """Mock response from the USDA API."""
    return {
        "foods": [
            {
                "description": "Apple Pie",
                "servingSize": 100,
                "servingSizeUnit": "g",
                "foodNutrients": [
                    {"nutrientName": "Energy", "value": 237, "unitName": "kcal"},
                    {"nutrientName": "Carbohydrate, by difference", "value": 34, "unitName": "g"},
                    {"nutrientName": "Total lipid (fat)", "value": 11, "unitName": "g"},
                    {"nutrientName": "Protein", "value": 2, "unitName": "g"}
                ]
            }
        ]
    }

# Test food_to_nutrition_step_2 function
def test_food_to_nutrition_step_2(mock_api_response):
    """Test the `food_to_nutrition_step_2` function."""
    with patch("api.service.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_api_response

        food_item = "Apple Pie"
        weight = 150
        output = food_to_nutrition_step_2(food_item, weight, "test_api_key")

        # Verify output structure
        assert isinstance(output, pd.DataFrame)
        assert "Description" in output.columns
        assert "Energy" in output.columns
        assert "Protein" in output.columns
        assert "Fat" in output.columns
        assert "Carbohydrate" in output.columns

        # Verify calculations
        assert output["Carbohydrate"].iloc[0] == "51.0 g"
        assert output["Energy"].iloc[0] == "355.5 kcal"
        assert output["Protein"].iloc[0] == "3.0 g"
        assert output["Fat"].iloc[0] == "16.5 g"

# Test transform_data_step_3 function
def test_transform_data_step_3():
    result = transform_data_step_3(mock_step2_output_extracted)
    assert "Carbohydrate (G)" in result.columns
    assert "Energy (KCAL)" in result.columns
    assert result.loc[0, "Carbohydrate (G)"] == 20.0
    assert result.loc[0, "Energy (KCAL)"] == 250.0

# Test send_output_step_3 function
def test_send_output_step_3():
    with patch("api.service.storage.Client") as mock_storage_client:
        mock_bucket = mock_storage_client.return_value.bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        send_output_step_3(mock_step3_output)
        mock_blob.upload_from_filename.assert_called_once_with("step3_output.json")
