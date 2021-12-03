import speech_recognition as sr
from googletrans import Translator

r = sr.Recognizer()
text_data = sr.AudioFile("./0b56bcfe_nohash_0.wav")

with text_data as source:
    audio = r.record(source)
    text = r.recognize_google(audio)
    translator = Translator()
    translation = translator.translate(text, dest="la").text
    print(translation)
