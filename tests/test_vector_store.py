from app.vector_store import VectorStore

def test_vector_store_add_and_search(clean_db):
    vs = VectorStore()

    # Create mock chunks
    chunks = [
        "Mohammad is a 4th-year computer engineering student",
        "The weather is nice today",
        "He entered Birzeit University in 2022"
    ]

    vs.add_chunks(chunks, doc_id="doc_cv")

    # Verify count
    assert vs.total_chunks == 3

    # Test hybrid search
    res = vs.search("Where does Mohammad study?", top_k=2)
    assert len(res) > 0
    assert "text" in res[0]

def test_vector_store_delete(clean_db):
    vs = VectorStore()

    chunks_doc1 = ["Doc 1 content"]
    chunks_doc2 = ["Doc 2 content"]

    vs.add_chunks(chunks_doc1, doc_id="id1")
    vs.add_chunks(chunks_doc2, doc_id="id2")

    assert vs.total_chunks == 2

    # Delete doc1
    removed = vs.delete_by_doc_id("id1")
    assert removed == 1
    assert vs.total_chunks == 1

    # Verify only doc 2 remains
    res = vs.search("Doc", top_k=5)
    assert len(res) == 1

def test_vector_store_filter_by_doc_ids(clean_db):
    vs = VectorStore()

    chunks_doc1 = ["Apples are red"]
    chunks_doc2 = ["Apples are green"]

    vs.add_chunks(chunks_doc1, doc_id="id1")
    vs.add_chunks(chunks_doc2, doc_id="id2")

    # Search with filter
    res = vs.search("Apples", top_k=5, doc_ids=["id2"])

    assert len(res) == 1
