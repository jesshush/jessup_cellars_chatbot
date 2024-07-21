import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, stem
from model import NeuralNet
import os
from docx import Document
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_docx(docx_path):
    if not os.path.exists(docx_path):
        logger.error(f"Error: The file '{docx_path}' does not exist.")
        return ""
    
    try:
        doc = Document(docx_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        logger.error(f"Error reading DOCX file: {str(e)}")
        return ""

docx_path = 'Corpus.docx'
docx_text = extract_text_from_docx(docx_path)

if not docx_text:
    logger.warning("Failed to extract text from DOCX. Proceeding with empty DOCX content.")
else:
    logger.info(f"Successfully extracted {len(docx_text)} characters from the DOCX.")

with open('intents.json', 'r') as f:
    intents = json.load(f)

# Create intents from docx content
docx_intents = []
sentences = sent_tokenize(docx_text)
for i, sentence in enumerate(sentences):
    docx_intents.append({
        "tag": f"docx_content_{i}",
        "patterns": [sentence],
        "responses": [sentence]
    })

# Combine intents
all_intents = intents['intents'] + docx_intents

# Process data
all_words = []
tags = []
xy = []

for intent in all_intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = word_tokenize(pattern.lower())
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

logger.info(f"{len(xy)} patterns")
logger.info(f"{len(tags)} tags: {tags}")
logger.info(f"{len(all_words)} unique stemmed words: {all_words}")

# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 16
output_size = len(tags)

logger.info(f"Input size: {input_size}, Output size: {output_size}")

# Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create dataset and dataloader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

logger.info(f'Final loss: {loss.item():.4f}')

# Prepare data for saving
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# Save the data
FILE = "data.pth"
torch.save(data, FILE)
logger.info(f'Training complete. File saved to {FILE}')

def get_response(sentence, context=None):
    sentence = word_tokenize(sentence.lower())
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in all_intents:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                if context and context['last_response'] in response:
                    # Find the index of the last response in the full text
                    start_index = docx_text.find(context['last_response'])
                    if start_index != -1:
                        # Get the next sentence
                        remaining_text = docx_text[start_index + len(context['last_response']):]
                        next_sentences = sent_tokenize(remaining_text)
                        if next_sentences:
                            response += " " + next_sentences[0]
                return response, tag

    return "I'm not sure how to respond to that. Can you please rephrase or ask something else?", None

logger.info("Starting conversation loop")
print("Let's chat! (type 'quit' to exit)")
context = {'last_response': None, 'last_tag': None}
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    if sentence.lower() == "tell me more about it":
        if context['last_response']:
            resp, tag = get_response(context['last_tag'], context)
        else:
            resp = "I'm sorry, but I don't have any previous context to expand on. Could you please ask a specific question?"
            tag = None
    else:
        resp, tag = get_response(sentence)
        context['last_response'] = resp
        context['last_tag'] = tag

    print(f"Bot: {resp}")

logger.info("Conversation ended")
print("Thanks for chatting!")