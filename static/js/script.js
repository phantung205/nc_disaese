// Custom JavaScript for Diabetes Prediction App

document.addEventListener('DOMContentLoaded', function () {
    // Form validation for manual input
    const manualForm = document.getElementById('manual-form');
    if (manualForm) {
        manualForm.addEventListener('submit', function (e) {
            const inputs = manualForm.querySelectorAll('input[required]');
            let isValid = true;

            inputs.forEach(input => {
                if (!input.value.trim()) {
                    input.classList.add('is-invalid');
                    isValid = false;
                } else {
                    input.classList.remove('is-invalid');
                    input.classList.add('is-valid');
                }
            });

            if (!isValid) {
                e.preventDefault();
                showAlert('Vui lòng điền đầy đủ thông tin!', 'danger');
            }
        });
    }

    // File upload validation
    const fileInput = document.querySelector('#file');
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            const file = this.files[0];
            if (file) {
                const allowedExtensions = ['.csv', '.xlsx', '.xls'];
                const fileName = file.name.toLowerCase();
                const isValidType = allowedExtensions.some(ext => fileName.endsWith(ext));

                if (!isValidType) {
                    showAlert('Chỉ chấp nhận file CSV hoặc Excel!', 'warning');
                    this.value = '';
                } else if (file.size > 10 * 1024 * 1024) { // 10MB limit
                    showAlert('File không được vượt quá 10MB!', 'warning');
                    this.value = '';
                }
            }
        });
    }

    // Image upload validation and preview
    const imageInput = document.querySelector('#image_file');
    const previewContainer = document.querySelector('#preview-container');
    const previewImage = document.querySelector('#image-preview');
    if (imageInput) {
        imageInput.addEventListener('change', function () {
            const file = this.files[0];
            if (file) {
                const allowedExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'];
                const fileName = file.name.toLowerCase();
                const isValidType = allowedExtensions.some(ext => fileName.endsWith(ext));

                if (!isValidType) {
                    showAlert('Chỉ chấp nhận ảnh PNG, JPG, JPEG, BMP, TIFF hoặc GIF!', 'warning');
                    this.value = '';
                    if (previewContainer) previewContainer.style.display = 'none';
                    if (previewImage) previewImage.src = '#';
                } else if (file.size > 10 * 1024 * 1024) {
                    showAlert('Ảnh không được vượt quá 10MB!', 'warning');
                    this.value = '';
                    if (previewContainer) previewContainer.style.display = 'none';
                    if (previewImage) previewImage.src = '#';
                } else {
                    if (previewImage && previewContainer) {
                        const reader = new FileReader();
                        reader.onload = function (e) {
                            previewImage.src = e.target.result;
                            previewContainer.style.display = 'block';
                        };
                        reader.readAsDataURL(file);
                    }
                }
            } else {
                if (previewContainer) previewContainer.style.display = 'none';
                if (previewImage) previewImage.src = '#';
            }
        });
    }

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});

// Utility function to show alerts
function showAlert(message, type) {
    const alertContainer = document.querySelector('.container-fluid') || document.querySelector('.container');
    const alertHTML = `
        <div class="alert alert-${type} alert-dismissible fade show mt-3" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    alertContainer.insertAdjacentHTML('afterbegin', alertHTML);
}

// Floating Chat Functionality
document.addEventListener('DOMContentLoaded', function () {
    const chatToggle = document.getElementById('chat-toggle');
    const chatWindow = document.getElementById('chat-window');
    const chatClose = document.getElementById('chat-close');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');

    // Load chat history from localStorage
    loadChatHistory();

    // Toggle chat window
    chatToggle.addEventListener('click', function () {
        chatWindow.style.display = chatWindow.style.display === 'flex' ? 'none' : 'flex';
    });

    // Close chat window
    chatClose.addEventListener('click', function () {
        chatWindow.style.display = 'none';
    });

    // Clear chat history
    document.getElementById('clear-chat').addEventListener('click', function () {
        chatMessages.innerHTML = '<div class="message bot">Xin chào! Tôi là trợ lý AI về bệnh tiểu đường. Bạn có câu hỏi gì không?</div>';
        localStorage.removeItem('chatHistory');
    });

    // Send message
    function sendMessage() {
        const message = chatInput.value.trim();
        if (message) {
            // Add user message
            addMessage(message, 'user');
            saveChatHistory();
            chatInput.value = '';

            // Send to server
            fetch('/chatbot/api', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message }),
            })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        addMessage('Lỗi: ' + data.error, 'bot');
                    } else {
                        addMessage(data.answer, 'bot');
                    }
                    saveChatHistory();
                })
                .catch(error => {
                    addMessage('Lỗi kết nối: ' + error.message, 'bot');
                    saveChatHistory();
                });
        }
    }

    // Add message to chat
    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Save chat history to localStorage
    function saveChatHistory() {
        const messages = [];
        const messageElements = chatMessages.querySelectorAll('.message');
        messageElements.forEach(msg => {
            messages.push({
                text: msg.textContent,
                type: msg.classList.contains('user') ? 'user' : 'bot'
            });
        });
        localStorage.setItem('chatHistory', JSON.stringify(messages));
    }

    // Load chat history from localStorage
    function loadChatHistory() {
        const history = localStorage.getItem('chatHistory');
        if (history) {
            const messages = JSON.parse(history);
            messages.forEach(msg => {
                addMessage(msg.text, msg.type);
            });
        } else {
            // Add welcome message if no history
            addMessage('Xin chào! Tôi là trợ lý AI về bệnh tiểu đường. Bạn có câu hỏi gì không?', 'bot');
        }
    }

    // Send on button click
    sendButton.addEventListener('click', sendMessage);

    // Send on Enter key
    chatInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});