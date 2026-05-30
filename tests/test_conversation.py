import app.conversation as conv

def test_create_and_get_conversation(clean_db):
    conv_id = conv.create_conversation("My Test Chat")
    assert conv_id is not None

    c = conv.get_conversation(conv_id)
    assert c is not None
    assert c["title"] == "My Test Chat"

def test_add_and_get_messages(clean_db):
    conv_id = conv.create_conversation("Chat")

    conv.add_message(conv_id, "user", "Hello there", model_used="qwen2.5:latest")
    conv.add_message(conv_id, "assistant", "Hi!", context_chunks=[{"text": "ctx"}], model_used="qwen2.5:latest")

    msgs = conv.get_messages(conv_id)
    assert len(msgs) == 2

    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello there"
    assert msgs[0]["context_chunks"] == []

    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Hi!"
    assert len(msgs[1]["context_chunks"]) == 1

    assert conv.get_message_count(conv_id) == 2

def test_rename_conversation(clean_db):
    conv_id = conv.create_conversation("Old Title")
    conv.rename_conversation(conv_id, "New Title")
    c = conv.get_conversation(conv_id)
    assert c["title"] == "New Title"

def test_delete_conversation_cascades(clean_db):
    conv_id = conv.create_conversation("Chat")
    conv.add_message(conv_id, "user", "Hello", "test")

    assert conv.get_message_count(conv_id) == 1

    # Delete conversation
    success = conv.delete_conversation(conv_id)
    assert success is True

    # Should be gone
    assert conv.get_conversation(conv_id) is None

    # Messages should be deleted too (cascade)
    assert len(conv.get_messages(conv_id)) == 0

def test_auto_title():
    assert conv._auto_title("Short question") == "Short question"
    long_q = "This is a very long question that goes well beyond the sixty character limit we established"
    assert len(conv._auto_title(long_q)) == 60
    assert conv._auto_title(long_q).endswith("...")
