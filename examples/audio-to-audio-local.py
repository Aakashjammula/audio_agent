import os
import torch
from RealtimeSTT.audio_recorder import AudioToTextRecorder
from kokoro import KPipeline
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the Ollama LLM with the gemma3:1b model
# Note: The 'max_tokens' parameter is not directly supported in the same way,
# but the prompt instructs the model to limit its response length.
llm = ChatOllama(
    model="gemma3:1b",
    temperature=0.6,
)

# Initialize the Text-to-Speech pipeline
tts_pipeline = KPipeline(
    lang_code="a",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

def query_llm(prompt: str) -> str:
    """Sends a prompt to the local LLM and gets a response."""
    system_prompts = SystemMessage(content=(
        "You are a speaking agent. Reply in natural spoken English. "
        "No lists or bullets. Plain words. Max 25 tokens."
    ))
    user_message = HumanMessage(content=prompt)
    resp = llm.invoke([system_prompts, user_message])
    return resp.content.strip()

def speak_text(text: str):
    """Converts text to speech and plays it."""
    for _, _, audio in tts_pipeline(text, voice="af_heart", speed=1.0):
        # Each chunk is a numpy array at 24kHz
        import sounddevice as sd
        sd.play(audio, 24000)
        sd.wait()

def main():
    """Main loop to listen, process, and speak."""
    print("🟢 Speak now...")
    # Initialize the audio recorder
    recorder = AudioToTextRecorder(language="en")

    while True:
        # Record audio and convert to text
        text = recorder.text()  # This is a blocking call
        print("🎤 You said:", text)

        # Get a reply from the LLM
        reply = query_llm(text)
        print("🤖 Ollama replies:", reply)

        # Speak the reply
        speak_text(reply)
        print("🟢 Speak now...")

if __name__ == "__main__":
    main()
