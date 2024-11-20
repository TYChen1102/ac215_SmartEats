import os
import pytest
from unittest.mock import patch, MagicMock
import json
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from predict_food import download, process_image, make_prediction, send_output

# Mock global variables
GCP_PROJECT = "test_project"
BUCKET_NAME = "test_bucket"

@pytest.fixture
def mock_model():
    """Fixture to mock a TensorFlow model."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.1] * 33])  # Mock predictions
    return model

@pytest.fixture
def mock_storage_client():
    """Fixture to mock the GCP storage client."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    return mock_client

# def test_download(mock_storage_client):
#     """Test the download function."""
#     with patch("predict_food.storage.Client", return_value=mock_storage_client):
#         download()

#         # Verify the blob's `download_to_filename` method was called
#         mock_blob = mock_storage_client.bucket(BUCKET_NAME).blob("shared_results/test_food.png")
#         mock_blob.download_to_filename.assert_called_once_with("test_food.png")

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

# def test_send_output(mock_storage_client):
#     """Test the send_output function."""
#     with patch("predict_food.storage.Client", return_value=mock_storage_client):
#         label = "Pizza"
#         probability = 0.85
#         send_output(label, probability)

#         # Check output JSON
#         with open("step1_output.json", "r") as f:
#             output = json.load(f)

#         assert output["food"] == label
#         assert output["prob"] == probability
# 
#         # Verify the blob's `upload_from_filename` method was called
#         mock_blob = mock_storage_client.bucket(BUCKET_NAME).blob("shared_results/step1_output.json")
#         mock_blob.upload_from_filename.assert_called_once_with("step1_output.json")
