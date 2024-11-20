import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from food_to_nutrition import extract_input, food_to_nutrition, send_output

GCP_PROJECT = "test_project"
BUCKET_NAME = "test_bucket"
API_KEY = "test_api_key"


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


def test_extract_input(mock_storage_client):
    """Test the `extract_input` function."""
    mock_step1_data = json.dumps({"food": "Apple Pie"})
    mock_weight_data = json.dumps({"Weight": 150})

    # Mock GCP storage client behavior
    mock_blob = mock_storage_client.bucket(BUCKET_NAME).blob
    mock_blob.side_effect = [
        MagicMock(download_as_text=lambda: mock_step1_data),
        MagicMock(download_as_text=lambda: mock_weight_data),
    ]

    with patch("food_to_nutrition.storage.Client", return_value=mock_storage_client):
        result = extract_input()

        assert result["food"] == "Apple Pie"
        assert result["Weight"] == 150


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


def test_send_output(mock_storage_client):
    """Test the `send_output` function."""
    # Mock input dataframe
    output_df = pd.DataFrame({
        "Description": ["Apple Pie"],
        "Cosine similarity score": [0.95],
        "Carbohydrate": ["51.0 g"],
        "Energy": ["355.5 kcal"],
        "Protein": ["3.0 g"],
        "Fat": ["16.5 g"]
    })

    with patch("food_to_nutrition.storage.Client", return_value=mock_storage_client):
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            send_output(output_df)

            # Check that JSON output file is created
            mock_open.assert_called_once_with("step2_output.json", "w")

            # Verify GCP upload
            mock_blob = mock_storage_client.bucket(BUCKET_NAME).blob("shared_results/step2_output.json")
            mock_blob.upload_from_filename.assert_called_once_with("step2_output.json")
