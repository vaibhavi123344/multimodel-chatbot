# from gtts import gTTS 
# import os

# tts = gTTS(text="Hi how are u", lang='en')
# tts.save("output.mp3")
# os.system("start output.mp3")
import pyttsx3

engine = pyttsx3.init()
engine.say("Hi how are u")
engine.runAndWait()