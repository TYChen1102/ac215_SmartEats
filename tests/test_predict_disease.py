import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from nutrition_predict_disease import (
    extract_input,
    transform_data,
    send_output
)


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
def mock_storage_client():
    """Fixture to mock the GCP storage client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    return mock_client


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


def test_extract_input(mock_storage_client, mock_step2_output):
    """Test the `extract_input` function."""
    mock_blob = mock_storage_client.bucket("ac215smarteat").blob
    mock_blob.return_value.download_as_text.return_value = json.dumps(mock_step2_output)

    with patch("nutrition_predict_disease.storage.Client", return_value=mock_storage_client):
        result = extract_input()

        # Verify the output is parsed correctly
        assert result == mock_step2_output
        assert result["Carbohydrate"] == "51.0 g"
        assert result["Energy"] == "355.5 kcal"


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


def test_send_output(mock_storage_client):
    """Test the `send_output` function."""
    mock_output = {
        "Obesity": "0.9",
        "Diabetes": "0.9",
        "High Cholesterol": "0.9",
        "Hypertension": "0.9"
    }

    with patch("nutrition_predict_disease.storage.Client", return_value=mock_storage_client):
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            send_output(mock_output)

            # Check that JSON output file is created
            mock_open.assert_called_once_with("step3_output.json", "w")

            # Verify GCP upload
            mock_blob = mock_storage_client.bucket("ac215smarteat").blob("shared_results/step3_output.json")
            mock_blob.upload_from_filename.assert_called_once_with("step3_output.json")
