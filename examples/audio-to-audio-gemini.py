import os
import torch
from dotenv import load_dotenv
from RealtimeSTT.audio_recorder import AudioToTextRecorder
from kokoro import KPipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load API key
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.6,
    max_tokens=25,
)

tts_pipeline = KPipeline(
    lang_code="a", 
    device="cuda" if torch.cuda.is_available() else "cpu",
)

def query_llm(prompt: str) -> str:
    system_prompts = SystemMessage(content=(
        "You are a speaking agent. Reply in natural spoken English. "
        "No lists or bullets. Plain words. Max 25 tokens."
    ))
    user_message = HumanMessage(content=prompt)
    resp = llm.invoke([system_prompts, user_message])
    return resp.content.strip()

def speak_text(text: str):
    for _, _, audio in tts_pipeline(text, voice="af_heart", speed=1.0):
        # Each chunk is a numpy array at 24kHz
        import sounddevice as sd
        sd.play(audio, 24000)
        sd.wait()

def main():
    print("🟢 Speak now...")
    recorder = AudioToTextRecorder(language="en")

    while True:
        text = recorder.text()  # blocking
        print("🎤 You said:", text)

        reply = query_llm(text)
        print("🤖 Gemini replies:", reply)

        speak_text(reply)
        print("🟢 Speak now...")

if __name__ == "__main__":
    main()
