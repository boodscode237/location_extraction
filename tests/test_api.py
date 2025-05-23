import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app
from app.models.loaders import get_spacy_nlp, get_bilstm_model, get_bilstm_word2idx, get_bilstm_idx2tag

@pytest.fixture
def client():
    """Fixture to provide a TestClient instance for the FastAPI app."""
    return TestClient(app)

@pytest.fixture
def mock_spacy_nlp():
    """Fixture to mock the get_spacy_nlp function."""
    with patch("app.models.loaders.get_spacy_nlp") as mock:
        yield mock

@pytest.fixture
def mock_bilstm_model():
    """Fixture to mock the get_bilstm_model function."""
    with patch("app.models.loaders.get_bilstm_model") as mock:
        yield mock

@pytest.fixture
def mock_bilstm_mappings():
    """Fixture to mock word2idx and idx2tag mappings."""
    with patch("app.models.loaders.get_bilstm_word2idx") as mock_word2idx, \
         patch("app.models.loaders.get_bilstm_idx2tag") as mock_idx2tag:
        yield mock_word2idx, mock_idx2tag

def test_health_check(client):
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is healthy"}

def test_frontend_serves_html(client):
    """Test the / endpoint serves HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Location Extractor" in response.text

def test_extract_with_spacy_success(client, mock_spacy_nlp):
    """Test successful location extraction with spaCy."""
    # Mock spaCy model to return a doc with entities
    class MockEntity:
        def __init__(self, text, label):
            self.text = text
            self.label_ = label

    class MockDoc:
        def __init__(self):
            self.ents = [
                MockEntity("Paris", "GPE")
            ]

    mock_spacy_nlp.return_value = lambda text: MockDoc()
    
    payload = {"text": "I visited London and Paris."}
    response = client.post("/extract-with-spacy/", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["input_text"] == payload["text"]
    assert json_response["extracted_locations"] == ["Paris"]
    assert json_response["model_used"] == "spaCy"
    assert json_response["error_message"] is None

def test_extract_with_spacy_empty_input(client):
    """Test spaCy endpoint with empty input."""
    payload = {"text": ""}
    response = client.post("/extract-with-spacy/", json=payload)
    assert response.status_code == 422
    assert "value is not a valid string" in response.text or "min_length" in response.text

def test_extract_with_spacy_model_unavailable(client, mock_spacy_nlp):
    """Test spaCy endpoint when model is not loaded."""
    mock_spacy_nlp.return_value = None
    payload = {"text": "I visited London and Paris."}
    response = client.post("/extract-with-spacy/", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "SpaCy model is not available or not loaded."

def test_extract_with_bilstm_success(client, mock_bilstm_model, mock_spacy_nlp, mock_bilstm_mappings):
    """Test successful location extraction with BiLSTM-CRF."""
    class MockToken:
        def __init__(self, text):
            self.text = text
    
    class MockDoc:
        def __init__(self, tokens):
            self.tokens = [MockToken(t) for t in tokens]
        
        def __iter__(self):
            return iter(self.tokens)
    
    mock_spacy_nlp.return_value = type("MockSpacy", (), {
        "tokenizer": lambda text: MockDoc(["I", "visited", "London", "and", "Paris"])
    })()

    # Mock BiLSTM model decode method
    def mock_decode(word_ids, mask):
        return [[0, 0, 1, 0, 2]]  # Simulating tags: O, O, B-LOC, O, I-LOC
    
    mock_bilstm_model.return_value = type("MockModel", (), {"decode": mock_decode})()
    
    # Mock word2idx and idx2tag
    mock_word2idx, mock_idx2tag = mock_bilstm_mappings
    mock_word2idx.return_value = {"I": 2, "visited": 3, "London": 4, "and": 5, "Paris": 6, "": 0}
    mock_idx2tag.return_value = {0: "O", 1: "B-LOC", 2: "I-LOC"}
        
    payload = {"text": "I visited London and Paris."}
    response = client.post("/extract-with-bilstm/", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["input_text"] == payload["text"]
    assert json_response["extracted_locations"] == ["London Paris"]
    assert json_response["model_used"] == "BiLSTM-CRF"
    assert json_response["error_message"] is None

def test_extract_with_bilstm_empty_input(client):
    """Test BiLSTM endpoint with empty input."""
    payload = {"text": ""}
    response = client.post("/extract-with-bilstm/", json=payload)
    assert response.status_code == 422
    assert "value is not a valid string" in response.text or "min_length" in response.text

def test_extract_with_bilstm_model_unavailable(client, mock_bilstm_model):
    """Test BiLSTM endpoint when model is not loaded."""
    mock_bilstm_model.return_value = None
    payload = {"text": "I visited London and Paris."}
    response = client.post("/extract-with-bilstm/", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "BiLSTM-CRF model or its mappings are not available."