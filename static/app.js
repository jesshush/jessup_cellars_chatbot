class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button'),
            inputField: document.querySelector('.chatbox__footer input'),
            messages: document.querySelector('.chatbox__messages')
        };

        this.state = false;
        this.messages = [];  
    }

    display() {
        const { openButton, chatBox, sendButton, inputField } = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        inputField.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    toggleState(chatbox) {
        this.state = !this.state;
        chatbox.classList.toggle('chatbox--active', this.state);
    }

    onSendButton(chatbox) {
        const textField = chatbox.querySelector('input');
        let text = textField.value;
        if (text === "") {
            return;
        }
    
        let msg1 = { name: "User", message: text };
        this.messages.push(msg1);
    
        fetch($SCRIPT_ROOT + '/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            console.log("Response data:", data);  // Debug response
            let msg2 = { name: "Sam", message: data.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';
        })
        .catch(error => {
            console.error('Error:', error);
            let errorMsg = { name: "Sam", message: "I'm sorry, I encountered an error. Please try again." };
            this.messages.push(errorMsg);
            this.updateChatText(chatbox);
            textField.value = '';
        });
    }
    

    updateChatText(chatbox) {
        const html = this.messages.slice().reverse().map(item => 
            `<div class="messages__item messages__item--${item.name === 'Sam' ? 'visitor' : 'operator'}">${item.message}</div>`
        ).join('');
        
        chatbox.querySelector('.chatbox__messages').innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();
