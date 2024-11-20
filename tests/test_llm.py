import pytest
from unittest.mock import patch, MagicMock
import json
from llm_main import extract_input, interact_with_llm

bucket_name = "ac215smarteat"

@pytest.fixture
def mock_google_storage():
    """Mock Google Cloud Storage interactions."""
    with patch("llm_main.storage.Client") as mock_storage_client:
        mock_client_instance = MagicMock()
        mock_bucket = MagicMock()

        # Mock blob for step2_output.json
        mock_blob_step2 = MagicMock()
        mock_blob_step2.download_as_text.return_value = json.dumps({
            "Description": "Grilled Chicken Salad",
            "Energy": "300 kcal",
            "Carbohydrate": "10g",
            "Protein": "25g",
            "Fat": "10g",
        })

        # Mock blob for step3_output.json
        mock_blob_step3 = MagicMock()
        mock_blob_step3.download_as_text.return_value = json.dumps({
            "Obesity": 0.25,
            "Diabetes": 0.25,
            "High Cholesterol": 0.25,
            "Hypertension": 0.25,
        })

        # Set bucket and blob behavior
        mock_bucket.blob.side_effect = lambda name: mock_blob_step2 if "step2_output.json" in name else mock_blob_step3
        mock_client_instance.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client_instance

        yield mock_storage_client

def test_extract_input(mock_google_storage):
    """Test the extract_input function with mocked Google Cloud Storage."""
    expected_output = (
        "This is the nutrition content and calories of the user's meal: "
        "Grilled Chicken Salad.\n"
        "Energy: 300 kcal, Carbohydrates: 10g, Protein: 25g, Fat: 10g."
        "The risk of 4 potential relavant diseases are Obesity: 0.25, Diabetes:0.25, \n"
        "High Cholesterol: 0.25, Hypertension: 0.25. Could you give us some dietary advice based on these information?"
    )

    result = extract_input()
    assert result == expected_output


@pytest.fixture
def mock_subprocess():
    """Mock the subprocess.run function."""
    with patch("llm_main.subprocess.run") as mock_run:
        mock_run_instance = MagicMock()
        mock_run.return_value = mock_run_instance
        yield mock_run

def test_interact_with_llm(mock_subprocess):
    """Test the interact_with_llm function with mocked subprocess."""
    test_prompt = "Sample prompt for testing"
    
    interact_with_llm(test_prompt)

    # Verify subprocess was called with expected arguments
    mock_subprocess.assert_any_call(["pip", "install", "--upgrade", "chromadb"], check=True)
    mock_subprocess.assert_any_call(["python", "./cli.py", "--chat", "--chunk_type", "char-split", "--query_text", str(test_prompt)])
