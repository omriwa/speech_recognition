from gtts import gTTS
import os

text = "This is a test of Human speech production"
language = "en"
obj = gTTS(text=text, lang=language, slow=False)
mp3Path = "./humanSpeechProduction/test.mp3"
obj.save(mp3Path)
