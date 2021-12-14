import speech_recognition as sr
import playsound
from gtts import gTTS
import os
import wolframalpha
from selenium import webdriver

num = 1

def assitant_speak(output):
    global num 

    num += 1

    print("Person : ", output)

    toSpeak = gTTS(text = output, lang = "en", slow = False)

    file = "./" + str(num) + ".mp3"
   
    toSpeak.save(file)

    playsound.playsound(file, True)
    os.remove(file)

def get_audio():
    rObject = sr.Recognizer()
    audio = ''

    with sr.Microphone() as source:
        print("Speak ...")

        audio = rObject.listen(source, phrase_time_limit = 5)

        print("Stop.")

        try:
            text = rObject.recognize_google(audio,language = "en-US")

            print("Your : ", text)
            
            return text
        except:
            assitant_speak("Could not undestand")

            return 0

if __name__ == '__main__':
    assitant_speak("Whats your name?")

    name = "Human"
    name = get_audio().lower()

    assitant_speak("Hello " + name)

    while(1):
        assitant_speak("What can I do for you?")

        text = get_audio()

        if type(text) != int:
            text = text.lower()

        if text == 0:
            continue

        if "exit" in str(text):
            assitant_speak("bye")