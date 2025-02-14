from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
import torch
from translate import Translator

from gtts import gTTS
import os

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import speech_recognition
import pyttsx3
import sys

#imports for hand sign detection
import pickle
import cv2
import mediapipe as mp
import numpy as np
from inference_classifier import get_video_input

import random
from fuzzywuzzy import fuzz


#Loading Pre-trained model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("Capstone-V4\intents.json", 'r') as json_data:
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

bot_name = "Anu"
print("Let's chat! (type 'quit' to exit)")


app = Flask(__name__)#Initilisation of flask 

@app.route("/")#defines a route ("/") that renders the home page using an HTML 
def index():
    return render_template('chat.html')

#Getting input from the user
# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     return get_Chat_response(input)

##############################################


def get_intent(input_text):
    highest_score = 0
    best_intent = None
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            score = fuzz.partial_ratio(input_text.lower(), pattern.lower())
            if score > highest_score:
                highest_score = score
                best_intent = intent
    if highest_score >= 79:  # Adjust threshold as needed
        
        return best_intent
    else:
        return None

def get_response(input_text):
    matched_intent = get_intent(input_text)
    if matched_intent:
        print("Hello")
        return random.choice(matched_intent['responses'])
    else:
        return "I'm sorry, I didn't understand that."

# @app.route("/get", methods=["GET", "POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     return get_response(input)

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    language = request.form["target_language"]
    input = msg
    response = get_Chat_response(input)
    translated_response = str(translate_text(response, language))
    return jsonify({"response": translated_response})

def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation

###############################################3333


def get_Chat_response(text):

    # sentence = "do you use credit cards?"
    sentence = str(text)
    if sentence == "quit":
        sys.exit()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.79:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        #print(f"{bot_name}: I do not understand...")
        return f"{bot_name}: I do not understand..."
    
def text_to_speech(text1, filename='output.mp3'):
    tts = gTTS(text=text1, lang='en')
    tts.save(filename)
    os.system(f'open {filename}')
    
@app.route("/mic_get", methods=["GET", "POST"])
def chat_mic():
    language = request.form["target_language"]
    recognizer = speech_recognition.Recognizer()
    print("speak in mic")
    #while True:
    try:
        with speech_recognition.Microphone() as mic:
            print("Computer:","speak")
            recognizer.adjust_for_ambient_noise(mic, duration=0.5)
            audio = recognizer.listen(mic, timeout = 1)

            text = recognizer.recognize_google(audio)
            input = text.lower()
            print("USER:",input)
            response = {
                "value1": text,
                "value2": str(translate_text(get_Chat_response_mic(input), language))
            }
            engine = pyttsx3.init()
            engine.say(response["value2"])
            engine.runAndWait()
            return jsonify(response)
    except:
        recognizer = speech_recognition.Recognizer()
        #continue
    print("end mic")
    

def get_Chat_response_mic(text):

    # sentence = "do you use credit cards?"
    sentence = str(text)
    if sentence == "quit":
        sys.exit()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.79:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        #print(f"{bot_name}: I do not understand...")
        return f"{bot_name}: I do not understand..."




if __name__ == '__main__':
    app.run()