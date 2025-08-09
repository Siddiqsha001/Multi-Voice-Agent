import os
import asyncio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import requests
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from deepgram import DeepgramClient,SpeakOptions
from deepgram.clients.listen import PrerecordedOptions

load_dotenv()
DEEPGRAM_API_KEY=os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY not found in environment variables")
dg_client=DeepgramClient(DEEPGRAM_API_KEY)
async def transcribe_file(audio_file_path):
    try:
        with open(audio_file_path, "rb") as audio:
            buffer=audio.read()

        response=await dg_client.listen.prerecorded.v("1").transcribe_file(
            {
                "buffer":buffer,
                "mimetype":"audio/wav"
            },
            PrerecordedOptions(
                model="nova",
                language="en-US",
                smart_format=True
            )
        )
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return ""

def listen_and_transcribe(duration=5,sample_rate=16000):
    print("Listening... Speak now.")
    recording=sd.rec(int(duration * sample_rate),samplerate=sample_rate,channels=1,dtype='int16')
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as f:
        wav.write(f.name,sample_rate,recording)
        audio_file_path=f.name

    transcript=asyncio.run(transcribe_file(audio_file_path))
    os.unlink(audio_file_path)
    print("You said:", transcript)
    return transcript

def speak(text,agent="planner"):
    voice_map={"optimist":"aura-joy","realist":"aura-solemn","planner":"aura-echo"}
    model=voice_map.get(agent,"aura-echo")
    options=SpeakOptions(model=model,encoding="linear16",sample_rate=24000)
    try:
        response=dg_client.speak.v("1").stream_memory({"text":text},options)
        audio_data=response.stream.getvalue()
        with tempfile.NamedTemporaryFile(delete=False,suffix=".wav") as f:
            f.write(audio_data)
            f.flush()
            audio=AudioSegment.from_wav(f.name)
            play(audio)
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        
if __name__=="__main__":
    text=listen_and_transcribe()
    if text:
        if "money" in text.lower():
            response="Financially, consider comparing outcomes before investing."
            agent="realist"
        elif "motivate" in text.lower() or "tired" in text.lower():
            response="You’ve come so far, don’t give up now! You’ve got this!"
            agent="optimist"
        else:
            response="Let’s plan your success, step by step!"
            agent="planner"
        speak(response,agent)