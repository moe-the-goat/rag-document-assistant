const API_URL = ''; // Relative path since we will proxy in Nginx or access directly
let currentConversationId = null;

// DOM Elements
const sidebar = document.getElementById('sidebar');
const toggleSidebarBtn = document.getElementById('toggle-sidebar');
const openSidebarBtn = document.getElementById('open-sidebar');
const newChatBtn = document.getElementById('new-chat-btn');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const conversationsList = document.getElementById('conversations-list');
const documentLibrary = document.getElementById('document-library');
const modelSelect = document.getElementById('model-select');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages-container');
const welcomeScreen = document.getElementById('welcome-screen');
const currentChatTitle = document.getElementById('current-chat-title');

// Initialize Marked with highlight.js
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    }
});

// Sidebar Toggle
toggleSidebarBtn.addEventListener('click', () => {
    sidebar.classList.add('closed');
    openSidebarBtn.style.display = 'block';
});
openSidebarBtn.addEventListener('click', () => {
    sidebar.classList.remove('closed');
    openSidebarBtn.style.display = 'none';
});

// Textarea auto-resize and send button state
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    sendBtn.disabled = this.value.trim() === '';
});

// Enter to send (Shift+Enter for new line)
chatInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) sendMessage();
    }
});

sendBtn.addEventListener('click', sendMessage);

newChatBtn.addEventListener('click', () => {
    currentConversationId = null;
    currentChatTitle.textContent = "New Conversation";
    messagesContainer.innerHTML = '';
    messagesContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = 'block';
    loadConversations();
});

// File Upload
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadStatus.textContent = 'Uploading...';
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok) {
            uploadStatus.textContent = `Success: ${data.chunks_added} chunks added.`;
            loadDocuments();
        } else {
            uploadStatus.textContent = `Error: ${data.detail}`;
        }
    } catch (err) {
        uploadStatus.textContent = 'Upload failed.';
    }
    setTimeout(() => uploadStatus.textContent = '', 3000);
});

async function loadConversations() {
    try {
        const res = await fetch('/conversations');
        const convos = await res.json();
        conversationsList.innerHTML = '';
        convos.forEach(c => {
            const div = document.createElement('div');
            div.className = `list-item ${c.conversation_id === currentConversationId ? 'active' : ''}`;

            const titleSpan = document.createElement('span');
            titleSpan.textContent = c.title;
            titleSpan.style.flexGrow = '1';
            titleSpan.style.overflow = 'hidden';
            titleSpan.style.textOverflow = 'ellipsis';

            const delBtn = document.createElement('span');
            delBtn.innerHTML = '🗑️';
            delBtn.style.cursor = 'pointer';
            delBtn.onclick = async (e) => {
                e.stopPropagation();
                await fetch(`/conversations/${c.conversation_id}`, { method: 'DELETE' });
                if (c.conversation_id === currentConversationId) newChatBtn.click();
                else loadConversations();
            };

            div.appendChild(titleSpan);
            div.appendChild(delBtn);
            div.onclick = () => selectConversation(c.conversation_id, c.title);
            conversationsList.appendChild(div);
        });
    } catch (e) { console.error('Failed to load conversations', e); }
}

async function selectConversation(id, title) {
    currentConversationId = id;
    currentChatTitle.textContent = title;
    welcomeScreen.style.display = 'none';
    messagesContainer.innerHTML = '';
    loadConversations(); // refresh active state

    try {
        const res = await fetch(`/conversations/${id}`);
        const data = await res.json();
        if (data.messages) {
            data.messages.forEach(m => appendMessage(m.role, m.content, m.context_chunks));
        }
    } catch (e) { console.error('Failed to load messages', e); }
}

async function loadDocuments() {
    try {
        const res = await fetch('/documents');
        const docs = await res.json();
        documentLibrary.innerHTML = '';
        docs.forEach(d => {
            const div = document.createElement('div');
            div.className = 'list-item';
            div.innerHTML = `
                <div style="flex-grow:1; overflow:hidden; text-overflow:ellipsis;" title="${d.original_name}">
                    ${d.original_name}
                </div>
                <span style="cursor:pointer;" onclick="deleteDocument('${d.doc_id}')">🗑️</span>
            `;
            documentLibrary.appendChild(div);
        });
    } catch (e) { console.error('Failed to load docs', e); }
}

window.deleteDocument = async (id) => {
    await fetch(`/documents/${id}`, { method: 'DELETE' });
    loadDocuments();
}

function appendMessage(role, content, contextChunks = []) {
    welcomeScreen.style.display = 'none';
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${role}`;

    const bubble = document.createElement('div');
    bubble.className = `bubble ${role}`;

    if (role === 'ai') {
        bubble.innerHTML = marked.parse(content);
        if (contextChunks && contextChunks.length > 0) {
            const ctxToggle = document.createElement('div');
            ctxToggle.className = 'context-toggle';
            ctxToggle.innerHTML = '🔍 View Context Source';

            const ctxContent = document.createElement('div');
            ctxContent.className = 'context-content';
            ctxContent.textContent = contextChunks.join('\n\n---\n\n');

            ctxToggle.onclick = () => {
                ctxContent.style.display = ctxContent.style.display === 'block' ? 'none' : 'block';
            };

            bubble.appendChild(ctxToggle);
            bubble.appendChild(ctxContent);
        }
    } else {
        bubble.textContent = content;
    }

    wrapper.appendChild(bubble);
    messagesContainer.appendChild(wrapper);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showTyping() {
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ai`;
    wrapper.id = 'typing-indicator';
    wrapper.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    messagesContainer.appendChild(wrapper);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    showTyping();

    const payload = {
        question: text,
        model: modelSelect.value
    };
    if (currentConversationId) payload.conversation_id = currentConversationId;

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        hideTyping();

        if (res.ok) {
            const data = await res.json();
            if (data.conversation_id && !currentConversationId) {
                currentConversationId = data.conversation_id;
                loadConversations();
            }
            appendMessage('ai', data.answer, data.context_chunks);
        } else {
            const err = await res.json();
            appendMessage('ai', `**Error:** ${err.detail || 'Failed to get answer.'}`);
        }
    } catch (e) {
        hideTyping();
        appendMessage('ai', `**Error:** Could not reach the server.`);
    }
}

// Init
loadConversations();
loadDocuments();
