import pytest
import numpy as np
import cv2
import requests
from time import sleep

# URL of the FastAPI server
BASE_URL = "http://localhost:9000/predict"

# Helper function to convert a NumPy image to bytes
def image_to_bytes(img):
    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()

# Fixture to wait for the server to be ready
@pytest.fixture(scope="module", autouse=True)
def wait_for_server():
    """
    Waits for the FastAPI server to start before running tests.
    """
    for _ in range(10):  # Retry up to 10 times
        try:
            response = requests.get("http://localhost:9000/docs")  # Check API health
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            sleep(1)
    else:
        pytest.fail("API server did not start in time")

# Test different image types
@pytest.mark.parametrize("image_type", ["zero", "random", "large", "small"])
def test_predict_endpoint(image_type):
    """
    Tests the /predict endpoint with different image types.
    """
    # Generate different types of test images
    if image_type == "zero":
        # Black image (all zeros)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    elif image_type == "random":
        # Random noise image
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    elif image_type == "large":
        # Large image (resize to 500x500)
        img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    elif image_type == "small":
        # Small image (resize to 50x50, will need upscaling in the API)
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    else:
        pytest.fail("Unknown image type")

    # Convert image to bytes
    img_bytes = image_to_bytes(img)

    # Send a POST request to the /predict endpoint
    response = requests.post(BASE_URL, files={"file": ("test.png", img_bytes, "image/png")})

    # Assert the response status code is 200
    assert response.status_code == 200, f"Failed for {image_type} image: {response.content}"

    # Parse the JSON response
    response_json = response.json()

    # Assert that the response contains expected keys
    assert "meal_info" in response_json, f"Missing 'meal_info' in response for {image_type} image"
    assert "text" in response_json, f"Missing 'text' in response for {image_type} image"

    # Print the results for debugging
    print(f"Test {image_type} image passed.")
    print(f"Response: {response_json}")

