const API_URL = ''; // Relative path since we will proxy in Nginx or access directly
let currentConversationId = null;

// DOM Elements
const sidebar = document.getElementById('sidebar');
const toggleSidebarBtn = document.getElementById('toggle-sidebar');
const openSidebarBtn = document.getElementById('open-sidebar');
const newChatBtn = document.getElementById('new-chat-btn');
const fileInput = document.getElementById('file-input');
const conversationsList = document.getElementById('conversations-list');
const documentLibrary = document.getElementById('document-library');
const modelSelectBtn = document.getElementById('model-select-btn');
const selectedModelText = document.getElementById('selected-model-text');
const modelDropdown = document.getElementById('model-dropdown');
let currentModel = 'qwen2.5:latest';
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const messagesContainer = document.getElementById('messages-container');
const welcomeScreen = document.getElementById('welcome-screen');
const currentChatTitle = document.getElementById('current-chat-title');

const settingsBtn = document.getElementById('settings-btn');
const settingsModal = document.getElementById('settings-modal');
const closeSettingsBtn = document.getElementById('close-settings-btn');
const clearAllChatsBtn = document.getElementById('clear-all-chats-btn');

const exportBtn = document.getElementById('export-btn');
const exportDropdown = document.getElementById('export-dropdown');
const exportMdBtn = document.getElementById('export-md-btn');
const exportPdfBtn = document.getElementById('export-pdf-btn');

// Initialize Marked with highlight.js
marked.setOptions({
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    }
});

// Custom Dropdown Logic
if (modelSelectBtn) {
    modelSelectBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        modelDropdown.classList.toggle('show');
    });

    document.addEventListener('click', (e) => {
        if (modelDropdown && !modelDropdown.contains(e.target) && !modelSelectBtn.contains(e.target)) {
            modelDropdown.classList.remove('show');
        }
    });

    document.querySelectorAll('.dropdown-item').forEach(item => {
        item.addEventListener('click', () => {
            document.querySelectorAll('.dropdown-item').forEach(i => i.classList.remove('selected'));
            item.classList.add('selected');
            currentModel = item.getAttribute('data-value');
            selectedModelText.textContent = item.querySelector('.dropdown-title').textContent;
            modelDropdown.classList.remove('show');
        });
    });
}

// Sidebar Toggle
toggleSidebarBtn.addEventListener('click', () => {
    sidebar.classList.add('closed');
    openSidebarBtn.style.display = 'block';
});
openSidebarBtn.addEventListener('click', () => {
    sidebar.classList.remove('closed');
    openSidebarBtn.style.display = 'none';
});

// Settings Modal
settingsBtn.addEventListener('click', () => settingsModal.style.display = 'flex');
closeSettingsBtn.addEventListener('click', () => settingsModal.style.display = 'none');
settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) settingsModal.style.display = 'none';
});

clearAllChatsBtn.addEventListener('click', async () => {
    if (confirm('Are you sure you want to delete all conversations? This cannot be undone.')) {
        await fetch('/conversations', { method: 'DELETE' });
        settingsModal.style.display = 'none';
        newChatBtn.click();
    }
});

// Export Dropdown
exportBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    exportDropdown.classList.toggle('show');
});
document.addEventListener('click', (e) => {
    if (!exportDropdown.contains(e.target) && !exportBtn.contains(e.target)) {
        exportDropdown.classList.remove('show');
    }
});

exportMdBtn.addEventListener('click', async () => {
    exportDropdown.classList.remove('show');
    if (!currentConversationId) return showToast('No conversation to export', 'error');

    try {
        const res = await fetch(`/conversations/${currentConversationId}`);
        const data = await res.json();
        if (!data.messages) return;

        let mdContent = `# ${data.title}\n\n`;
        data.messages.forEach(m => {
            mdContent += `### ${m.role === 'user' ? 'User' : 'Nexus AI'}\n${m.content}\n\n`;
        });

        const blob = new Blob([mdContent], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${data.title.replace(/[^a-z0-9]/gi, '_')}.md`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) {
        showToast('Failed to export markdown', 'error');
    }
});

exportPdfBtn.addEventListener('click', () => {
    exportDropdown.classList.remove('show');
    if (!currentConversationId || document.querySelector('.chat-empty')) return showToast('No conversation to export', 'error');

    const element = document.getElementById('messages-container');
    const opt = {
        margin:       10,
        filename:     `${currentChatTitle.textContent.replace(/[^a-z0-9]/gi, '_')}.pdf`,
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2, useCORS: true, logging: false },
        jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
    };
    showToast('Generating PDF...', 'info');
    html2pdf().set(opt).from(element).save().then(() => {
        showToast('PDF Exported successfully', 'success');
    });
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
    document.querySelector('.chat-area').classList.add('chat-empty');
    loadConversations();
});

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// File Upload
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    showToast('Uploading...', 'info');
    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });

        let data;
        try {
            data = await res.json();
        } catch (e) {
            data = { detail: "Server returned invalid response." };
        }

        if (res.ok) {
            showToast(`Success: ${data.chunks_added} chunks added.`, 'success');
            loadDocuments();
        } else {
            if (res.status === 409) {
                showToast(`⚠️ Duplicate: ${data.detail || 'File already exists.'}`, 'error');
            } else {
                showToast(`❌ Error: ${data.detail || 'Upload failed.'}`, 'error');
            }
        }
    } catch (err) {
        showToast('❌ Upload failed. Could not reach server.', 'error');
    }

    // Clear the input so the same file can be selected again if needed
    fileInput.value = '';
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

            const titleInput = document.createElement('input');
            titleInput.type = 'text';
            titleInput.className = 'conv-title-input';
            titleInput.value = c.title;
            titleInput.style.display = 'none';

            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'conv-actions';

            const editBtn = document.createElement('button');
            editBtn.className = 'btn icon-btn';
            editBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>';

            const delBtn = document.createElement('button');
            delBtn.className = 'btn icon-btn';
            delBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>';

            editBtn.onclick = (e) => {
                e.stopPropagation();
                titleSpan.style.display = 'none';
                actionsDiv.style.display = 'none';
                titleInput.style.display = 'block';
                titleInput.focus();
            };

            const saveRename = async () => {
                const newTitle = titleInput.value.trim();
                if (newTitle && newTitle !== c.title) {
                    await fetch(`/conversations/${c.conversation_id}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ title: newTitle })
                    });
                    if (c.conversation_id === currentConversationId) currentChatTitle.textContent = newTitle;
                }
                loadConversations();
            };

            titleInput.onblur = saveRename;
            titleInput.onkeydown = (e) => {
                if (e.key === 'Enter') saveRename();
                if (e.key === 'Escape') loadConversations();
            };

            delBtn.onclick = async (e) => {
                e.stopPropagation();
                await fetch(`/conversations/${c.conversation_id}`, { method: 'DELETE' });
                if (c.conversation_id === currentConversationId) newChatBtn.click();
                else loadConversations();
            };

            actionsDiv.appendChild(editBtn);
            actionsDiv.appendChild(delBtn);

            div.appendChild(titleSpan);
            div.appendChild(titleInput);
            div.appendChild(actionsDiv);
            div.onclick = () => {
                if (titleInput.style.display !== 'block') selectConversation(c.conversation_id, c.title);
            };
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
        if (data.messages && data.messages.length > 0) {
            document.querySelector('.chat-area').classList.remove('chat-empty');
            data.messages.forEach(m => appendMessage(m.role, m.content, m.context_chunks));
        } else {
            welcomeScreen.style.display = 'block';
            messagesContainer.appendChild(welcomeScreen);
            document.querySelector('.chat-area').classList.add('chat-empty');
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
    document.querySelector('.chat-area').classList.remove('chat-empty');
    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${role}`;

    const bubble = document.createElement('div');
    bubble.className = `bubble ${role}`;

    if (role === 'ai') {
        renderStreamingContent(content, bubble);
        attachContextChunks(bubble, contextChunks);
        attachCopyButtons(bubble, content);
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

function createStreamingBubble() {
    welcomeScreen.style.display = 'none';
    document.querySelector('.chat-area').classList.remove('chat-empty');

    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ai`;

    const bubble = document.createElement('div');
    bubble.className = `bubble ai`;

    wrapper.appendChild(bubble);
    messagesContainer.appendChild(wrapper);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return { wrapper, bubble };
}

function renderStreamingContent(rawText, container) {
    let html = "";
    let mainText = rawText;

    const thinkStart = rawText.indexOf('<think>');
    if (thinkStart !== -1) {
        const thinkEnd = rawText.indexOf('</think>');
        let thinkContent = "";
        let isClosed = false;

        if (thinkEnd !== -1) {
            thinkContent = rawText.substring(thinkStart + 7, thinkEnd);
            mainText = rawText.substring(0, thinkStart) + rawText.substring(thinkEnd + 8);
            isClosed = true;
        } else {
            thinkContent = rawText.substring(thinkStart + 7);
            mainText = rawText.substring(0, thinkStart);
            isClosed = false;
        }

        let summaryText = isClosed ? "💡 Thought Process" : "🧠 Thinking...";
        html += `<details class="think-block" ${isClosed ? '' : 'open'}>
                    <summary>${summaryText}</summary>
                    <div class="think-content">${marked.parse(thinkContent)}</div>
                 </details>`;
    }

    html += marked.parse(mainText);
    container.innerHTML = html;
}

function attachContextChunks(bubble, contextChunks) {
    if (!contextChunks || contextChunks.length === 0) return;
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

function attachCopyButtons(bubble, rawText) {
    // Add copy code buttons to pre blocks
    const preBlocks = bubble.querySelectorAll('pre');
    preBlocks.forEach(pre => {
        const btn = document.createElement('button');
        btn.className = 'copy-code-btn';
        btn.textContent = 'Copy code';
        btn.onclick = () => {
            const code = pre.querySelector('code');
            navigator.clipboard.writeText(code ? code.innerText : pre.innerText);
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy code', 2000);
        };
        pre.appendChild(btn);
    });

    // Add copy answer button at the bottom
    const copyAnsBtn = document.createElement('button');
    copyAnsBtn.className = 'copy-answer-btn';
    copyAnsBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy answer';

    // Extract actual answer without think tags for copying
    let cleanText = rawText;
    const thinkStart = rawText.indexOf('<think>');
    const thinkEnd = rawText.indexOf('</think>');
    if (thinkStart !== -1 && thinkEnd !== -1) {
        cleanText = rawText.substring(0, thinkStart) + rawText.substring(thinkEnd + 8);
    } else if (thinkStart !== -1) {
        cleanText = rawText.substring(0, thinkStart);
    }

    copyAnsBtn.onclick = () => {
        navigator.clipboard.writeText(cleanText.trim());
        copyAnsBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';
        setTimeout(() => {
            copyAnsBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy answer';
        }, 2000);
    };
    bubble.appendChild(copyAnsBtn);
}


async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    appendMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    const payload = {
        question: text,
        model: currentModel
    };
    if (currentConversationId) payload.conversation_id = currentConversationId;

    const { bubble } = createStreamingBubble();
    bubble.innerHTML = '<div class="typing-indicator" style="padding: 0;"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';

    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json();
            bubble.innerHTML = marked.parse(`**Error:** ${err.detail || 'Failed to get answer.'}`);
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullAnswer = "";
        let contextChunks = [];
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                if (fullAnswer === "") {
                    bubble.innerHTML = marked.parse(`**Error:** Received empty response from server.`);
                }
                break;
            }

            buffer += decoder.decode(value, { stream: true });

            let boundary = buffer.indexOf('\n\n');
            while (boundary !== -1) {
                const event = buffer.substring(0, boundary);
                buffer = buffer.substring(boundary + 2);

                if (event.startsWith('data: ')) {
                    const dataStr = event.substring(6);
                    if (dataStr === '[DONE]') break;

                    try {
                        const data = JSON.parse(dataStr);
                        if (data.conversation_id && !currentConversationId) {
                            currentConversationId = data.conversation_id;
                            loadConversations();
                        }
                        if (data.context_chunks) {
                            contextChunks = data.context_chunks;
                        }
                        if (data.chunk) {
                            fullAnswer += data.chunk;
                            renderStreamingContent(fullAnswer, bubble);
                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                        }
                    } catch (e) {
                        console.error("Parse error:", e, "Data string:", dataStr);
                    }
                }
                boundary = buffer.indexOf('\n\n');
            }
        }

        attachContextChunks(bubble, contextChunks);
        attachCopyButtons(bubble, fullAnswer);

    } catch (e) {
        bubble.innerHTML = marked.parse(`**Error:** Could not reach the server.`);
    }
}

// Init
loadConversations();
loadDocuments();
