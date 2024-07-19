import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "jess"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.9:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "pdf_content":
                    # For PDF content, we'll do a more thorough search
                    pdf_text = intent['responses'][0]  
                    relevant_info = []
                    for word in sentence:
                        if word.lower() in pdf_text.lower():
                            # Find the sentence containing the word
                            sentences = pdf_text.split('.')
                            for s in sentences:
                                if word.lower() in s.lower():
                                    relevant_info.append(s.strip())
                                    break
                    if relevant_info:
                        return f"Here's what I found in our wine catalog: {' '.join(relevant_info)}"
                    else:
                        return "I couldn't find specific information about that in our wine catalog. Can you please be more specific?"
                else:
                    return random.choice(intent['responses'])

    return "I'm sorry, I don't have enough information to answer that. Please contact our business owners for more details."