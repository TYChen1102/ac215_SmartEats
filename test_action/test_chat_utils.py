import pytest
import os
import json
import shutil
from api.utils.chat_utils import ChatHistoryManager

# Test constants
TEST_MODEL = "test-model"
TEST_HISTORY_DIR = "test-chat-history"
TEST_SESSION_ID = "test-session"
TEST_CHAT_ID = "chat-001"
TEST_MESSAGE_ID = "msg-001"
TEST_IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wMAAAMBAQF4jOgAAAAASUVORK5CYII="
)  # Single pixel PNG
TEST_CHAT_DATA = {
    "chat_id": TEST_CHAT_ID,
    "dts": 1234567890,
    "messages": [
        {
            "message_id": TEST_MESSAGE_ID,
            "content": "Hello, world!",
            "image": TEST_IMAGE_BASE64,
        }
    ],
}

@pytest.fixture
def chat_history_manager():
    # Create the ChatHistoryManager instance for testing
    return ChatHistoryManager(model=TEST_MODEL, history_dir=TEST_HISTORY_DIR)

@pytest.fixture(autouse=True)
def cleanup():
    # Cleanup the test directory after each test
    yield
    if os.path.exists(TEST_HISTORY_DIR):
        shutil.rmtree(TEST_HISTORY_DIR)

def test_save_chat(chat_history_manager):
    # Save a chat and verify it is saved correctly
    chat_history_manager.save_chat(TEST_CHAT_DATA, TEST_SESSION_ID)
    chat_file = os.path.join(
        TEST_HISTORY_DIR, TEST_MODEL, TEST_SESSION_ID, f"{TEST_CHAT_ID}.json"
    )
    assert os.path.exists(chat_file)

    with open(chat_file, "r", encoding="utf-8") as f:
        saved_chat = json.load(f)

    # Verify chat data
    assert saved_chat["chat_id"] == TEST_CHAT_ID
    assert len(saved_chat["messages"]) == 1
    assert "image_path" in saved_chat["messages"][0]

def test_save_image(chat_history_manager):
    # Save an image and verify it is saved correctly
    image_path = chat_history_manager._save_image(
        TEST_CHAT_ID, TEST_MESSAGE_ID, TEST_IMAGE_BASE64
    )
    assert image_path
    full_image_path = os.path.join(TEST_HISTORY_DIR, TEST_MODEL, image_path)
    assert os.path.exists(full_image_path)

def test_load_image(chat_history_manager):
    # Save and load an image
    image_path = chat_history_manager._save_image(
        TEST_CHAT_ID, TEST_MESSAGE_ID, TEST_IMAGE_BASE64
    )
    loaded_image_data = chat_history_manager._load_image(image_path)
    assert loaded_image_data == TEST_IMAGE_BASE64

def test_get_chat(chat_history_manager):
    # Save a chat and retrieve it
    chat_history_manager.save_chat(TEST_CHAT_DATA, TEST_SESSION_ID)
    retrieved_chat = chat_history_manager.get_chat(TEST_CHAT_ID, TEST_SESSION_ID)
    assert retrieved_chat is not None
    assert retrieved_chat["chat_id"] == TEST_CHAT_ID

def test_get_recent_chats(chat_history_manager):
    # Save multiple chats and retrieve recent ones
    for i in range(3):
        chat_id = f"chat-{i:03}"
        chat_data = {
            "chat_id": chat_id,
            "dts": 1234567890 + i,
            "messages": [{"message_id": f"msg-{i:03}", "content": f"Message {i}"}],
        }
        chat_history_manager.save_chat(chat_data, TEST_SESSION_ID)

    recent_chats = chat_history_manager.get_recent_chats(TEST_SESSION_ID, limit=2)
    assert len(recent_chats) == 2
    assert recent_chats[0]["chat_id"] == "chat-002"
    assert recent_chats[1]["chat_id"] == "chat-001"
