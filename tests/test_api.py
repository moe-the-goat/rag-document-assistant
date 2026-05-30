import io

def test_health_check(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "running", "service": "AI Document Assistant (RAG)"}

def test_status(test_client, mock_vector_store):
    response = test_client.get("/status")
    assert response.status_code == 200
    assert "total_chunks" in response.json()
    assert response.json()["status"] == "empty"

def test_upload_missing_auth(test_client):
    # Setup API KEY
    import app.auth
    app.auth.API_KEY = "test_key"

    # Create fake file
    file_content = b"Fake PDF"
    file_obj = io.BytesIO(file_content)

    response = test_client.post("/upload", files={"file": ("test.pdf", file_obj, "application/pdf")})
    assert response.status_code == 401

    app.auth.API_KEY = ""  # Reset

def test_upload_invalid_type(test_client):
    file_obj = io.BytesIO(b"Fake executable")
    response = test_client.post("/upload", files={"file": ("test.exe", file_obj, "application/x-msdownload")})
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_upload_success(test_client, mock_vector_store, mock_llm):
    file_obj = io.BytesIO(b"%PDF-1.4 Fake PDF content")
    response = test_client.post("/upload", files={"file": ("test.pdf", file_obj, "application/pdf")})

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert "doc_id" in data
    assert data["chunks_added"] == 5

def test_ask_question(test_client, mock_vector_store, mock_llm):
    payload = {
        "question": "What is the meaning of life?",
        "model": "qwen2.5:latest"
    }
    response = test_client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "This is a mocked answer from the AI."
    assert "conversation_id" in data

def test_rate_limiting(test_client, mock_vector_store, mock_llm):
    # We will spam the /ask endpoint to trigger 429 Too Many Requests
    # Since rate limit is 30/minute

    payload = {"question": "test", "model": "test"}
    responses = []

    # Fire 35 requests
    for _ in range(35):
        responses.append(test_client.post("/ask", json=payload))

    status_codes = [r.status_code for r in responses]
    assert 200 in status_codes
    assert 429 in status_codes  # Rate limit exceeded
