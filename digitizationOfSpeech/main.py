import speech_recognition as sr

# Create recognizer
r = sr.Recognizer()
audio = sr.AudioFile("./audioFileAnalysis/T08-violin.wav")

with audio as source:
    a = r.record(source)

# print(r.recognize_google(a))

# Capture segments
with audio as source:
    a = r.record(source, duration=4)

print(r.recognize_google(a))
