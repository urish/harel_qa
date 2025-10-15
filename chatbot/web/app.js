// API configuration
const API_BASE_URL = window.location.origin;

// DOM elements
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question');
const categorySelect = document.getElementById('category');
const chatMessages = document.getElementById('chat-messages');
const sendBtn = document.getElementById('send-btn');
const loadingIndicator = document.getElementById('loading');

// Add message to chat
function addMessage(content, isUser = false, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    // Preserve newlines in the content
    messageContent.style.whiteSpace = 'pre-wrap';
    messageContent.textContent = content;

    messageDiv.appendChild(messageContent);

    // Add sources if provided (bot messages only)
    if (!isUser && sources && sources.length > 0) {
        // Extract referenced document numbers from the answer
        const referencedDocs = new Set();
        const refPattern = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
        let match;
        while ((match = refPattern.exec(content)) !== null) {
            const nums = match[1].replace(/\s/g, '').split(',');
            nums.forEach(n => {
                if (n) referencedDocs.add(parseInt(n));
            });
        }

        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';

        const sourcesLabel = document.createElement('div');
        sourcesLabel.className = 'sources-label';
        sourcesLabel.textContent = 'מקורות:';
        sourcesDiv.appendChild(sourcesLabel);

        sources.forEach((source, idx) => {
            const sourceItem = document.createElement('div');
            const sourceNumber = idx + 1;
            const isReferenced = referencedDocs.has(sourceNumber);
            sourceItem.className = `source-item${isReferenced ? ' referenced' : ''}`;

            // Extract filename from path
            const filename = source.source_file.split('/').pop();

            sourceItem.textContent = `[${sourceNumber}] ${filename}, עמוד ${source.page_number}`;
            sourcesDiv.appendChild(sourceItem);
        });

        messageDiv.appendChild(sourcesDiv);
    }

    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show/hide loading indicator
function setLoading(isLoading) {
    loadingIndicator.style.display = isLoading ? 'flex' : 'none';
    sendBtn.disabled = isLoading;
    questionInput.disabled = isLoading;
    categorySelect.disabled = isLoading;
}

// Send question to API
async function sendQuestion(question, category) {
    try {
        const response = await fetch(`${API_BASE_URL}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                category: category
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'שגיאה בשרת');
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Handle form submission
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const question = questionInput.value.trim();
    const category = categorySelect.value;

    if (!question || !category) {
        alert('אנא בחר קטגוריה והקלד שאלה');
        return;
    }

    // Add user message
    addMessage(question, true);

    // Clear input
    questionInput.value = '';

    // Show loading
    setLoading(true);

    try {
        // Send question and get answer
        const response = await sendQuestion(question, category);

        // Add bot response with sources
        addMessage(response.answer, false, response.sources);
    } catch (error) {
        addMessage('מצטער, אירעה שגיאה. אנא נסה שוב.', false, null);
    } finally {
        setLoading(false);
        questionInput.focus();
    }
});

// Focus on input when category is selected
categorySelect.addEventListener('change', () => {
    if (categorySelect.value) {
        questionInput.focus();
    }
});
