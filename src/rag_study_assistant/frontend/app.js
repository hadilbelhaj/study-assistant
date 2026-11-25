
const chatForm = document.getElementById("chat-form");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messagesContainer = document.getElementById("messages");
const welcomeScreen = document.getElementById("welcome-screen");
const newChatButton = document.getElementById("new-chat-button");
const connectionStatus = document.getElementById("connection-status");
const connectionLabel = document.getElementById("connection-label");

let conversationId = null;
let isSending = false;


// --------------------------------------------------
// Connection status
// --------------------------------------------------

async function checkConnection() {
    try {
        const response = await fetch("/health/ready");

        if (!response.ok) {
            throw new Error("Assistant is not ready");
        }

        connectionStatus.classList.remove("offline");
        connectionLabel.textContent = "Assistant online";
    } catch {
        connectionStatus.classList.add("offline");
        connectionLabel.textContent = "Assistant unavailable";
    }
}


// --------------------------------------------------
// Message rendering
// --------------------------------------------------

function scrollToBottom() {
    const chatContainer = document.getElementById("chat-container");
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function createElement(tag, className, text = "") {
    const element = document.createElement(tag);
    element.className = className;
    element.textContent = text;
    return element;
}

function formatPages(source) {
    const start = source.page_start;
    const end = source.page_end;

    if (start == null) return "Unknown";

    if (end == null || end === start) {
        return String(start);
    }

    return `${start}-${end}`;
}

function renderSources(sources) {
    if (!Array.isArray(sources) || sources.length === 0) {
        return null;
    }

    const details = document.createElement("details");
    details.className = "sources";

    const summary = createElement(
        "summary",
        "",
        `${sources.length} course source${sources.length === 1 ? "" : "s"}`
    );

    details.appendChild(summary);

    for (const source of sources) {
        const card = document.createElement("div");
        card.className = "source-card";

        const title = createElement(
            "div",
            "source-title",
            `${source.id} · ${source.lecture || source.source_file || "Course source"}`
        );

        card.appendChild(title);

        const fields = [
            ["Course", source.course],
            ["Section", source.section],
            ["Document", source.source_file],
            ["Pages", formatPages(source)]
        ];

        for (const [label, value] of fields) {
            if (!value) continue;

            const detail = createElement(
                "div",
                "source-detail",
                `${label}: ${value}`
            );

            card.appendChild(detail);
        }

        details.appendChild(card);
    }

    return details;
}

function appendMessage(role, content, sources = [], state = "") {
    const message = document.createElement("article");
    message.className = `message ${role} ${state}`.trim();

    const avatar = createElement(
        "div",
        "avatar",
        role === "user" ? "YOU" : "AI"
    );

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";

    const bubble = createElement("div", "message-bubble", content);
    messageContent.appendChild(bubble);

    if (role === "assistant" && sources.length > 0) {
        const sourceElement = renderSources(sources);

        if (sourceElement) {
            messageContent.appendChild(sourceElement);
        }
    }

    message.appendChild(avatar);
    message.appendChild(messageContent);

    messagesContainer.appendChild(message);
    scrollToBottom();

    return message;
}


// --------------------------------------------------
// Loading state
// --------------------------------------------------

function setLoading(loading) {
    isSending = loading;
    messageInput.disabled = loading;
    sendButton.disabled = loading;
    sendButton.textContent = loading ? "…" : "↑";
}


// --------------------------------------------------
// Send a question
// --------------------------------------------------

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message || isSending) {
        return;
    }

    welcomeScreen.hidden = true;

    appendMessage("user", message);

    messageInput.value = "";
    messageInput.style.height = "auto";

    setLoading(true);

    const pendingMessage = appendMessage(
        "assistant",
        "Je consulte vos supports de cours...",
        [],
        "pending"
    );

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                conversation_id: conversationId
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "An error occurred while processing your question."
            );
        }

        if (!data.conversation_id || typeof data.answer !== "string") {
            throw new Error("The server returned an invalid response.");
        }

        // Reuse this ID for all subsequent messages in this conversation.
        conversationId = data.conversation_id;

        pendingMessage.remove();

        appendMessage(
            "assistant",
            data.answer,
            data.sources || []
        );

    } catch (error) {
        pendingMessage.remove();

        appendMessage(
            "assistant",
            error.message || "Unable to contact the assistant.",
            [],
            "error"
        );

    } finally {
        setLoading(false);
        messageInput.focus();
    }
});


// --------------------------------------------------
// New conversation
// --------------------------------------------------

function startNewConversation() {
    if (isSending) return;

    conversationId = null;
    messagesContainer.replaceChildren();
    welcomeScreen.hidden = false;

    messageInput.value = "";
    messageInput.focus();
}

newChatButton.addEventListener("click", startNewConversation);


// --------------------------------------------------
// Example questions
// --------------------------------------------------

document.querySelectorAll(".suggestion-card").forEach((card) => {
    card.addEventListener("click", () => {
        messageInput.value = card.dataset.prompt || "";
        messageInput.focus();
        chatForm.requestSubmit();
    });
});


// --------------------------------------------------
// Textarea behavior
// --------------------------------------------------

messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = `${messageInput.scrollHeight}px`;
});

messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
    }
});


// --------------------------------------------------
// Initialization
// --------------------------------------------------

checkConnection();
