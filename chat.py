import torch
import torch.nn as nn
from nltk_utils import bag_of_words, tokenize
from model import NeuralNet
import random
import json
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
all_intents = data["intents"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(data["model_state"])
model.eval()

def get_response(sentence, context=None):
    sentence = tokenize(sentence)
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
                if context and context['last_response']:
                    if tag == context['last_tag']:
                        start_index = context['last_response'].find(context['last_response'])
                        if start_index != -1:
                            remaining_text = context['last_response'][start_index + len(context['last_response']):]
                            next_sentences = remaining_text.split(".")
                            if next_sentences:
                                response += " " + next_sentences[0]
                return response, tag

    return "I'm not sure how to respond to that. Can you please rephrase or ask something else?", None
