import app.document_registry as registry

def test_register_and_get_document(clean_db):
    doc_id = registry.register_document(
        original_name="test.pdf",
        file_hash="hash123",
        file_type=".pdf",
        file_size=1024,
        file_path="/tmp/test.pdf",
        chunk_count=10
    )

    assert doc_id is not None
    doc = registry.get_document(doc_id)
    assert doc is not None
    assert doc["original_name"] == "test.pdf"
    assert doc["file_hash"] == "hash123"
    assert doc["chunk_count"] == 10

def test_find_duplicate(clean_db):
    registry.register_document(
        original_name="test.pdf",
        file_hash="hash_dup",
        file_type=".pdf",
        file_size=1024,
        file_path="/tmp/test.pdf",
        chunk_count=10
    )

    dup = registry.find_duplicate("hash_dup")
    assert dup is not None
    assert dup["original_name"] == "test.pdf"

    non_dup = registry.find_duplicate("hash_new")
    assert non_dup is None

def test_delete_document(clean_db):
    doc_id = registry.register_document(
        original_name="test.pdf",
        file_hash="hash123",
        file_type=".pdf",
        file_size=1024,
        file_path="/tmp/test.pdf",
        chunk_count=10
    )

    assert registry.get_document(doc_id) is not None
    registry.delete_document(doc_id)
    assert registry.get_document(doc_id) is None

def test_list_documents(clean_db):
    registry.register_document("doc1.pdf", "hash1", ".pdf", 100, "/tmp/1", 5)
    registry.register_document("doc2.txt", "hash2", ".txt", 200, "/tmp/2", 10)

    docs = registry.list_documents()
    assert len(docs) == 2
    # Check ordering (newest first)
    assert docs[0]["original_name"] == "doc2.txt"
    assert docs[1]["original_name"] == "doc1.pdf"
