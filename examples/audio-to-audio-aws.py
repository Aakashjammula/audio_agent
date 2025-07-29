import asyncio
import json
import os
import time
import pyaudio
import sys
import boto3
import sounddevice
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
import re
import torch
import numpy as np
from dotenv import load_dotenv
load_dotenv()


from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler



# --- Configuration ---
MODEL_ID = os.getenv('MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')



config = {
    'log_level': 'info',
    'region': AWS_REGION,
    'polly': {'Engine': 'neural', 'LanguageCode': 'en-US', 'VoiceId': 'Joanna', 'OutputFormat': 'pcm'},
    'vad': {'threshold': 0.5, 'silence_sec': 1.5} # VAD threshold for Silero
}



# --- Centralized State Management ---
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.interrupt_event = threading.Event()
        self.user_started_speaking = asyncio.Event() # Event to signal main loop
        self._is_bot_speaking = False



    def is_bot_speaking(self):
        with self.lock: return self._is_bot_speaking
    def start_bot_speech(self):
        with self.lock:
            self._is_bot_speaking = True
            self.interrupt_event.clear()
    def stop_bot_speech(self):
        with self.lock:
            self._is_bot_speaking = False
            self.interrupt_event.clear()
    def interrupt(self):
        if self.is_bot_speaking(): self.interrupt_event.set()
    def was_interrupted(self):
        return self.interrupt_event.is_set()



def printer(text, level):
    if config['log_level'] in ('info', 'debug') and level == 'info':
        print(text, flush=True)
    elif config['log_level'] == 'debug' and level == 'debug':
        print(text, flush=True)



def to_audio_generator(converse_stream, app_state):
    prefix = ''
    if converse_stream:
        for event in converse_stream:
            if app_state.was_interrupted(): break
            if 'contentBlockDelta' in event:
                text = event['contentBlockDelta']['delta']['text']
                parts = re.split(r'(?<=[.!?])\s*', text)
                if len(parts) > 1:
                    sentences = parts[:-1]
                    new_prefix = parts[-1]
                    for sentence in sentences:
                        if sentence:
                            full_sentence = prefix + sentence
                            print(full_sentence, flush=True, end=' ')
                            yield full_sentence
                            prefix = ''
                    prefix += new_prefix
                else:
                    prefix += text
        if prefix.strip() and not app_state.was_interrupted():
            print(prefix, flush=True, end='')
            yield prefix



class AudioPlayer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.audio_stream = None
        self.lock = threading.Lock()
    def play(self, audio_data_stream, app_state):
        with self.lock:
            if self.audio_stream: self.stop()
            self.audio_stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        try:
            while not app_state.was_interrupted():
                data = audio_data_stream.read(1024)
                if not data: break
                if self.audio_stream and self.audio_stream.is_active(): self.audio_stream.write(data)
                else: break
        finally:
            if audio_data_stream: audio_data_stream.close()
            self.stop()
    def stop(self):
        with self.lock:
            if self.audio_stream:
                try:
                    if self.audio_stream.is_active(): self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except Exception: pass
                finally: self.audio_stream = None
    def terminate(self):
        self.stop()
        self.p.terminate()



class BedrockWrapper:
    def __init__(self, audio_player, app_state):
        self.audio_player = audio_player
        self.app_state = app_state
        self.polly = boto3.client('polly', region_name=config['region'])
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])
        self.messages_history = []



    def _get_system_prompt(self):
        return "You are a friendly and helpful voice assistant. Keep your responses concise and conversational."



    def _speak_text(self, text):
        if self.app_state.was_interrupted() or not text.strip(): return
        try:
            polly_response = self.polly.synthesize_speech(Text=text, Engine=config['polly']['Engine'], LanguageCode=config['polly']['LanguageCode'], VoiceId=config['polly']['VoiceId'], OutputFormat='pcm')
            self.audio_player.play(polly_response['AudioStream'], self.app_state)
        except Exception as e:
            printer(f"[ERROR] Polly synthesis failed: {e}", "info")



    def invoke_bedrock(self, text):
        self.app_state.start_bot_speech()
        try:
            system_prompt = self._get_system_prompt()
            self.messages_history.append({"role": "user", "content": [{"text": text}]})



            response_stream = self.bedrock_runtime.converse_stream(
                modelId=MODEL_ID,
                messages=self.messages_history,
                system=[{"text": system_prompt}]
            )



            full_assistant_response = []
            audio_gen = to_audio_generator(response_stream['stream'], self.app_state)
           
            for sentence in audio_gen:
                self._speak_text(sentence)
                full_assistant_response.append(sentence)



            # Append the full response to history for context in the next turn
            if full_assistant_response:
                self.messages_history.append({
                    "role": "assistant",
                    "content": [{"text": " ".join(full_assistant_response)}]
                })



        except Exception as e:
            printer(f"[ERROR] Bedrock invocation error: {e}", "info")
            self._speak_text("I'm sorry, I encountered an error. Please try again.")
        finally:
            self.app_state.stop_bot_speech()



class TranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_transcript = []
    async def handle_transcript_event(self, transcript_event):
        results = transcript_event.transcript.results
        if results and results[0].alternatives and not results[0].is_partial:
            transcript = results[0].alternatives[0].transcript
            print(transcript, end=" ", flush=True)
            self.full_transcript.append(transcript)
    def get_full_transcript(self):
        return " ".join(self.full_transcript).strip()



class MicStream:
    def __init__(self, app_state, audio_player):
        self.app_state, self.audio_player = app_state, audio_player
       
        # --- Silero VAD Setup ---
        try:
            self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        except Exception as e:
            printer(f"[FATAL] Could not load Silero VAD model. Please check network connection and torch installation: {e}", "info")
            sys.exit(1)
           
        self.samplerate = 16000  # Silero VAD standard sample rate
        self.frame_length = 512 # A common frame size
       
        self.transcribe_client = TranscribeStreamingClient(region=config['region'])
        printer("[INFO] Silero VAD and MicStream initialized.", "info")



    def _is_speech(self, chunk):
        """Checks if a given audio chunk contains speech."""
        audio_int16 = np.frombuffer(chunk, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        try:
            speech_prob = self.vad_model(torch.from_numpy(audio_float32), self.samplerate).item()
            return speech_prob > config['vad']['threshold']
        except Exception as e:
            printer(f"[WARN] VAD processing error: {e}", "debug")
            return False



    async def _mic_stream_generator(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()
        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, bytes(indata))
        stream = sounddevice.RawInputStream(
            channels=1, samplerate=self.samplerate, callback=callback, blocksize=self.frame_length, dtype="int16")
        with stream:
            while True: yield await input_queue.get()



    async def run_vad_listener(self):
        """A persistent background task that only listens for the start of speech."""
        printer("[INFO] VAD listener started.", "info")
        async for chunk in self._mic_stream_generator():
            if self.app_state.is_bot_speaking():
                if self._is_speech(chunk):
                    printer("\n[INFO] Barge-in detected! Interrupting bot.", "info")
                    self.app_state.interrupt()
                    self.audio_player.stop()
            elif not self.app_state.user_started_speaking.is_set():
                if self._is_speech(chunk):
                    self.app_state.user_started_speaking.set()



    async def listen_and_transcribe(self):
        """This function is called only after the VAD has detected speech."""
        printer("\n[INFO] Speech detected! Starting transcription...", "info")
       
        async def _write_chunks(stream):
            silence_frames = 0
            async for chunk in self._mic_stream_generator():
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
                if not self._is_speech(chunk):
                    silence_frames += 1
                    if silence_frames >= int((self.samplerate / self.frame_length) * config['vad']['silence_sec']):
                        await stream.input_stream.end_stream()
                        return
                else:
                    silence_frames = 0



        try:
            stream = await self.transcribe_client.start_stream_transcription(
                language_code="en-US", media_sample_rate_hz=self.samplerate, media_encoding="pcm",
            )
            handler = TranscriptHandler(stream.output_stream)
            await asyncio.gather(_write_chunks(stream), handler.handle_events())
            final_transcript = handler.get_full_transcript()
            if final_transcript:
                printer(f"\n[INFO] User input: {final_transcript}", "info")
            return final_transcript
        except Exception as e:
            printer(f"[ERROR] An error occurred during transcription: {e}", "info")
            return None



    def cleanup(self):
        pass # No explicit cleanup needed for the torch model



async def main():
    loop = asyncio.get_event_loop()
    print_startup_info()


    app_state = AppState()
    audio_player = AudioPlayer()
    bedrock_wrapper = BedrockWrapper(audio_player, app_state)
    mic_stream_manager = MicStream(app_state, audio_player)
   
    try:
        # --- WELCOME MESSAGE LOGIC ---
        try:
            welcome_message = "Hello. I am V Assist. How can I help you today?"
            printer(f"[INFO] Speaking welcome message...", "info")
           
            def speak_welcome():
                app_state.start_bot_speech()
                bedrock_wrapper._speak_text(welcome_message)
                app_state.stop_bot_speech()



            welcome_task = loop.run_in_executor(ThreadPoolExecutor(max_workers=1), speak_welcome)
            await welcome_task
           
        except Exception as e:
            printer(f"[ERROR] Failed to play welcome message: {e}", "info")
        # --- END WELCOME MESSAGE LOGIC ---



        # Start the persistent VAD listener as a background task
        vad_task = asyncio.create_task(mic_stream_manager.run_vad_listener())
       
        while True:
            printer("\n[INFO] Waiting for user to speak...", "info")
            await app_state.user_started_speaking.wait()
            app_state.user_started_speaking.clear()



            transcript = await mic_stream_manager.listen_and_transcribe()



            if transcript and not app_state.is_bot_speaking():
                bedrock_task = loop.run_in_executor(
                    ThreadPoolExecutor(max_workers=1),
                    bedrock_wrapper.invoke_bedrock,
                    transcript
                )
                await bedrock_task
           
            printer("[INFO] Turn complete.", "info")



    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("\nExiting application...")
        if 'vad_task' in locals() and not vad_task.done():
            vad_task.cancel()
        if mic_stream_manager: mic_stream_manager.cleanup()
        if audio_player: audio_player.terminate()



def print_startup_info():
    print("*************************************************************")
    print(f"[INFO] Bedrock Model: {MODEL_ID}")
    print("[INFO] Voice assistant is ready.")
    print("*************************************************************")



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")