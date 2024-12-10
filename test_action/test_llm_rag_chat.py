import pytest
import os
import shutil
import base64
from fastapi.testclient import TestClient
from api.routers.llm_rag_chat import router
from api.utils.chat_utils import ChatHistoryManager

# Initialize TestClient
app = router
client = TestClient(app)

# Test constants
TEST_MODEL = "llm-rag"
TEST_HISTORY_DIR = "chat-history"
TEST_SESSION_ID = "test-session"
HEADERS = {"X-Session-ID": TEST_SESSION_ID}

@pytest.fixture(scope="function", autouse=True)
def cleanup():
    """Clean up the test directory after each test"""
    if os.path.exists(TEST_HISTORY_DIR):
        shutil.rmtree(TEST_HISTORY_DIR)
    yield
    if os.path.exists(TEST_HISTORY_DIR):
        shutil.rmtree(TEST_HISTORY_DIR)

@pytest.fixture
def chat_manager():
    """Initialize a ChatHistoryManager for testing"""
    return ChatHistoryManager(model=TEST_MODEL, history_dir=TEST_HISTORY_DIR)

def test_get_chats_initially_empty(chat_manager):
    """Test that the chat history is initially empty"""
    response = client.get("/chats", headers=HEADERS)
    assert response.status_code == 200
    assert len(response.json()) == 0

def test_start_chat(chat_manager):
    """Test starting a new chat"""
    payload = {"content": "Hello, this is a test message."}
    response = client.post("/chats", json=payload, headers=HEADERS)
    assert response.status_code == 200

    # Verify that a chat file is created
    chat_dir = os.path.join(TEST_HISTORY_DIR, TEST_MODEL, TEST_SESSION_ID)
    assert os.path.exists(chat_dir)
    chat_files = os.listdir(chat_dir)
    assert len(chat_files) == 1

def test_continue_chat(chat_manager):
    """Test continuing an existing chat"""
    # Start a new chat
    payload = {"content": "Hello, this is the first message."}
    response = client.post("/chats", json=payload, headers=HEADERS)
    assert response.status_code == 200
    chat_id = response.json()["chat_id"]

    # Continue the chat
    payload = {"content": "This is a follow-up message."}
    response = client.post(f"/chats/{chat_id}", json=payload, headers=HEADERS)
    assert response.status_code == 200

    # Verify the chat file still exists and is updated
    chat_dir = os.path.join(TEST_HISTORY_DIR, TEST_MODEL, TEST_SESSION_ID)
    chat_files = os.listdir(chat_dir)
    assert len(chat_files) == 1

def test_get_specific_chat(chat_manager):
    """Test retrieving a specific chat"""
    # Start a new chat
    payload = {"content": "Hello, this is a test message."}
    response = client.post("/chats", json=payload, headers=HEADERS)
    assert response.status_code == 200
    chat_id = response.json()["chat_id"]

    # Retrieve the specific chat
    response = client.get(f"/chats/{chat_id}", headers=HEADERS)
    assert response.status_code == 200

def test_get_chat_image(chat_manager, tmp_path):
    """Test retrieving an image from a chat"""
    
    # Define the test image
    image_filename = 'test_image.png'
    
    # Read the image file and encode it in base64
    with open(os.path.join(image_filename), 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # Start a new chat with an image message
    payload = {
        "content": "This is a message with an image.",
        "image": image_base64,
    }
    response = client.post("/chats", json=payload, headers=HEADERS)
    assert response.status_code == 200
    chat_id = response.json()["chat_id"]

    # Verify image directory exists
    image_dir = os.path.join(TEST_HISTORY_DIR, TEST_MODEL, "images", chat_id)
    assert os.path.exists(image_dir)
    image_files = os.listdir(image_dir)
    assert len(image_files) == 1

    # Verify the saved image file exists
    saved_image_path = os.path.join(image_dir, image_files[0])
    assert os.path.exists(saved_image_path)