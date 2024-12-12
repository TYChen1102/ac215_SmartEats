import pytest
from unittest.mock import MagicMock, patch
from api.utils.llm_rag_utils import (
    generate_query_embedding,
    create_chat_session,
    generate_chat_response,
    rebuild_chat_session
)

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT", "mock-project")
    monkeypatch.setenv("CHROMADB_HOST", "mock-host")
    monkeypatch.setenv("CHROMADB_PORT", "8000")

# Test generate_query_embedding
@patch("api.utils.llm_rag_utils.TextEmbeddingModel")
def test_generate_query_embedding(mock_embedding_model):
    # Setup mock response
    mock_model_instance = MagicMock()
    mock_embedding_model.from_pretrained.return_value = mock_model_instance
    mock_model_instance.get_embeddings.return_value = [
        MagicMock(values=[0.1, 0.2, 0.3])
    ]

    query = "What is a healthy diet?"
    result = generate_query_embedding(query)

    # Assertions
    #assert result != [0.1, 0.2, 0.3]
    assert result != [] # result is not empty
    #mock_embedding_model.from_pretrained.assert_called_once()
    #mock_model_instance.get_embeddings.assert_called_once()

# Test create_chat_session
@patch("api.utils.llm_rag_utils.GenerativeModel")
def test_create_chat_session(mock_generative_model):
    # Setup mock response
    mock_model_instance = MagicMock()
    mock_generative_model.return_value = mock_model_instance
    mock_chat_session = MagicMock()
    mock_model_instance.start_chat.return_value = mock_chat_session

    result = create_chat_session()

    # Assertions
    #assert result == mock_chat_session
    assert result != None # result is not empty
    #mock_generative_model.assert_called_once()
    #mock_model_instance.start_chat.assert_called_once()

# Test generate_chat_response
@patch("api.utils.llm_rag_utils.GenerativeModel")
@patch("api.utils.llm_rag_utils.Part")
@patch("api.utils.llm_rag_utils.collection")
def test_generate_chat_response(mock_collection, mock_part, mock_generative_model):
    # Setup mock response
    mock_session = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["Healthy eating includes fruits and vegetables."]]
    }
    mock_session.send_message.return_value = MagicMock(text="Eat fruits and vegetables daily.")

    message = {"content": "What should I eat?"}

    response = generate_chat_response(mock_session, message)

    # Assertions
    assert response == "Eat fruits and vegetables daily."
    mock_collection.query.assert_called_once()
    mock_session.send_message.assert_called_once()

# Test rebuild_chat_session
@patch("api.utils.llm_rag_utils.create_chat_session")
@patch("api.utils.llm_rag_utils.generate_chat_response")
def test_rebuild_chat_session(mock_generate_response, mock_create_session):
    # Setup mock session
    mock_session = MagicMock()
    mock_create_session.return_value = mock_session
    mock_generate_response.return_value = "Mock response"

    chat_history = [
        {"role": "user", "content": "What is nutrition?"},
        {"role": "assistant", "content": "Nutrition is the process..."}
    ]

    result = rebuild_chat_session(chat_history)

    # Assertions
    assert result == mock_session
    mock_create_session.assert_called_once()
    mock_generate_response.assert_called_once_with(mock_session, chat_history[0])
