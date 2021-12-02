from gtts import gTTS

tts_en = gTTS("hello", lang="en")
tts_en.save("./hello.mp3")

tts_fr = gTTS("bonjour", lang="fr")
tts_fr.save("./hello_bonjour.mp3")
