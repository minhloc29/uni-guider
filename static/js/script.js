document.getElementById("form").addEventListener('submit', (e) => {
    e.preventDefault(); // prevent default page refresh
    progressConversation();
});
function createLoadingIcon() {
    const loadingIcon = document.createElement("div");
    loadingIcon.classList.add("loading-icon");
    return loadingIcon;
}

async function progressConversation() {
    const userInput = document.getElementById("user-input");
    // document.getElementById("chatbot-conversation-container").scrollIntoView({ behavior: "smooth", block: "start" });
        const question = userInput.value.trim();
    if(!question) return;

    addMessage(question, "user");
    console.log("User question: ", question);
    userInput.value = "";

    const botMessageDiv = addMessage("", "bot"); // Create an empty message for streaming
    const loadingIcon = createLoadingIcon();
    botMessageDiv.appendChild(loadingIcon);
    setTimeout(() => window.scrollTo(0, scrollY), 0);

    const response = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: question })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        if (botMessageDiv.contains(loadingIcon)) {
            botMessageDiv.removeChild(loadingIcon);
        }
        botMessageDiv.textContent += decoder.decode(value, { stream: true });

    }
}

// just the bot answer show on the frontend
function addMessage(text, sender){
    const chatContainer = document.createElement("div");
    if (sender === "user") {
        chatContainer.classList.add("chat-container-user");
    } 
    else {
        chatContainer.classList.add("chat-container-bot");
    }


    // Create and set profile image
    const profilePic = document.createElement("img");
    profilePic.src = "static/images/xin_vc.jpg";
    profilePic.classList.add("logo");

    // Create chat content container
    const chatContent = document.createElement("div");
    chatContent.classList.add("chat-content");

    // Create username element
    const usernameDiv = document.createElement("div");
    usernameDiv.classList.add("username");
    usernameDiv.textContent = "HUST GUIDER";

    // Create message element
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message");
    messageDiv.textContent = text;

    if(sender === "bot"){
        chatContent.appendChild(usernameDiv);
    }

    chatContent.appendChild(messageDiv);

    if(sender === "bot"){
        chatContainer.appendChild(profilePic);
    }

    chatContainer.appendChild(chatContent);

    document.getElementById("chatbot-answer-box").appendChild(chatContainer);

    setTimeout(() => {
        const chatContainer = document.getElementById("chatbot-answer-box").lastElementChild;
        if (chatContainer) {
            chatContainer.scrollIntoView({ behavior: "smooth", block: "end" });
        }
    }, 100);
    
    return messageDiv; 
}