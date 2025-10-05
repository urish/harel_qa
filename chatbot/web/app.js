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
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;

    messageDiv.appendChild(messageContent);
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
        return data.answer;
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
        const answer = await sendQuestion(question, category);

        // Add bot response
        addMessage(answer, false);
    } catch (error) {
        addMessage('מצטער, אירעה שגיאה. אנא נסה שוב.', false);
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
