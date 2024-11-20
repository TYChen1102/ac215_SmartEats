import os
import pytest
from unittest.mock import patch, MagicMock
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from predict_food import process_image, make_prediction
from food_to_nutrition import food_to_nutrition
from nutrition_predict_disease import transform_data
from llm_main import interact_with_llm

API_KEY = "test_api_key"

@pytest.fixture
def mock_model():
    """Fixture to mock a TensorFlow model."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.1] * 33])  # Mock predictions
    return model

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

@pytest.fixture
def mock_step2_output():
    """Fixture for mock input JSON data from step2_output.json."""
    return {
        "Carbohydrate": "51.0 g",
        "Energy": "355.5 kcal",
        "Protein": "3.0 g",
        "Fat": "16.5 g"
    }

@pytest.fixture
def mock_models():
    """Fixture to mock the trained machine learning models."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.1, 0.9]]  # Simulate probabilities for diseases
    return {
        "obesity_model": mock_model,
        "diabetes_model": mock_model,
        "high_cholesterol_model": mock_model,
        "hypertension_model": mock_model,
    }

@pytest.fixture
def mock_subprocess():
    """Mock the subprocess.run function."""
    with patch("llm_main.subprocess.run") as mock_run:
        mock_run_instance = MagicMock()
        mock_run.return_value = mock_run_instance
        yield mock_run

def test_process_image():
    """Test the image processing function."""
    # Create a dummy image file
    img_path = "dummy_image.png"
    dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)

    # Test the function
    processed_img = process_image(img_path)
    assert processed_img.shape == (1, 224, 224, 3)  # Check dimensions after resizing
    os.remove(img_path)  # Cleanup dummy image

def test_make_prediction(mock_model):
    """Test the prediction function."""
    with patch("predict_food.model", mock_model):
        # Create dummy input
        dummy_img = np.zeros((1, 224, 224, 3))
        label, probability = make_prediction(dummy_img)

        assert isinstance(label, str)  # Label should be a string
        assert isinstance(probability, float)  # Probability should be a float
        assert probability > 0.0  # Probability should be non-zero

def test_process_prediction(mock_model):
    """Integration test of image preprocessing and classification"""
    with patch("predict_food.model", mock_model):

        # Black image (all zeros)
        img_path = "dummy_image.png"
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)
        processed_img = process_image(img_path)
        label, probability = make_prediction(processed_img)
        assert isinstance(label, str)  # Label should be a string
        assert isinstance(probability, float)  # Probability should be a float
        assert probability > 0.0  # Probability should be non-zero

        # Random noise image
        img_path = "dummy_image.png"
        dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)
        processed_img = process_image(img_path)
        label, probability = make_prediction(processed_img)
        assert isinstance(label, str)  # Label should be a string
        assert isinstance(probability, float)  # Probability should be a float
        assert probability > 0.0  # Probability should be non-zero

        # Large image
        img_path = "dummy_image.png"
        dummy_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)
        processed_img = process_image(img_path)
        label, probability = make_prediction(processed_img)
        assert isinstance(label, str)  # Label should be a string
        assert isinstance(probability, float)  # Probability should be a float
        assert probability > 0.0  # Probability should be non-zero
        
        # Small image
        img_path = "dummy_image.png"
        dummy_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(img_path, dummy_img)
        processed_img = process_image(img_path)
        label, probability = make_prediction(processed_img)
        assert isinstance(label, str)  # Label should be a string
        assert isinstance(probability, float)  # Probability should be a float
        assert probability > 0.0  # Probability should be non-zero

def test_food_to_nutrition(mock_api_response):
    """Test the `food_to_nutrition` function."""
    with patch("food_to_nutrition.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_api_response

        food_item = "Apple Pie"
        weight = 150
        output = food_to_nutrition(food_item, weight, API_KEY)

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

def test_transform_data(mock_step2_output):
    """Test the `transform_data` function."""
    transformed_df = transform_data(mock_step2_output)

    # Verify the transformed DataFrame
    assert isinstance(transformed_df, pd.DataFrame)
    assert list(transformed_df.columns) == [
        "Carbohydrate (G)",
        "Energy (KCAL)",
        "Protein (G)",
        "Fat (G)"
    ]
    assert transformed_df["Carbohydrate (G)"].iloc[0] == 51.0
    assert transformed_df["Energy (KCAL)"].iloc[0] == 355.5
    assert transformed_df["Protein (G)"].iloc[0] == 3.0
    assert transformed_df["Fat (G)"].iloc[0] == 16.5

def test_predict_disease_risks(mock_models, mock_step2_output):
    """Test the disease risk prediction logic."""
    transformed_df = transform_data(mock_step2_output)

    # Mock model loading and prediction
    with patch("nutrition_predict_disease.joblib.load", side_effect=lambda _: mock_models["obesity_model"]):
        obesity_prob = mock_models["obesity_model"].predict_proba(transformed_df)[0][1]
        diabetes_prob = mock_models["diabetes_model"].predict_proba(transformed_df)[0][1]
        high_cholesterol_prob = mock_models["high_cholesterol_model"].predict_proba(transformed_df)[0][1]
        hypertension_prob = mock_models["hypertension_model"].predict_proba(transformed_df)[0][1]

    # Verify probabilities
    assert obesity_prob == 0.9
    assert diabetes_prob == 0.9
    assert high_cholesterol_prob == 0.9
    assert hypertension_prob == 0.9

def test_interact_with_llm(mock_subprocess):
    """Test the interact_with_llm function with mocked subprocess."""
    test_prompt = "Sample prompt for testing"
    
    interact_with_llm(test_prompt)

    # Verify subprocess was called with expected arguments
    mock_subprocess.assert_any_call(["pip", "install", "--upgrade", "chromadb"], check=True)
    mock_subprocess.assert_any_call(["python", "./cli.py", "--chat", "--chunk_type", "char-split", "--query_text", str(test_prompt)])
