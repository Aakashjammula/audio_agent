# 🎙️ Audio Agent 

This project showcases various architectures for building real-time, conversational voice agents. It provides four distinct examples, each demonstrating a different approach to the audio-to-audio pipeline, from fully modular cloud services to completely local, offline-first setups.

## 🚀 The Architectural Examples

This repository contains four different implementations to compare and contrast various approaches.

### 1. The Modular Cloud Pipeline

**File:** `examples/audio-to-audio-aws.py`

This approach assembles a voice agent using separate, best-in-class cloud services for each component of the pipeline. It offers maximum flexibility and control.

* **Architecture:** STT (API) -> VAD (Local) -> LLM (API) -> TTS (API)
* **Components:**
    * **STT:** Amazon Transcribe (Streaming)
    * **VAD:** Silero VAD (for endpointing)
    * **LLM:** Amazon Bedrock (Claude 3 Haiku)
    * **TTS:** Amazon Polly
* **Key Features:** Custom barge-in logic, robust VAD for silence detection.


### 2. The Hybrid Pipeline (Cloud LLM + Local TTS and STT)

**File:** `examples/audio-to-audio-gemini.py`

This example balances cloud intelligence with local processing for lower TTS latency and cost. It's a great choice for applications where response speed is critical and a GPU is available.

* **Architecture:** STT (Local) -> LLM (API) -> TTS (Local)
* **Components:**
    * **STT:** `RealtimeSTT` (Local,well optmized whisper)
    * **LLM:** Google Gemini 2.0 Flash (via LangChain)
    * **TTS:** `Kokoro-82M` (local, high-performance model)
* **Key Features:** Low-latency local transcription and speech synthesis and optimized and includes vad.


### 3. The End-to-End S2S API

**File:** `examples/audio-to-audio-nova-sonic.py`

This approach uses a single, highly integrated API that handles the entire audio-to-audio conversation flow. It simplifies development by abstracting away the underlying components.

* **Architecture:** A single bidirectional stream for Audio In -> S2S Model -> Audio Out
* **Components:**
    * **S2S Model:** Amazon Nova-Sonic
* **Key Features:** Built-in tool use capabilities, barge-in detection, and low-latency bidirectional streaming managed by a single API.


### 4. The Fully Local Pipeline

**File:** `examples/audio-to-audio-local.py`

This example runs entirely on your local machine, requiring no internet connection after initial model downloads. It is the best choice for privacy-focused applications or offline environments.

* **Architecture:** STT (Local) -> LLM (Local) -> TTS (Local)
* **Components:**
    * **STT:** `RealtimeSTT`
    * **LLM:** Ollama (`gemma3:1b`)
    * **TTS:** `Kokoro`
* **Key Features:** Runs completely offline, ensures data privacy, no API costs.


## 🙏 Acknowledgements

🔊 **RealtimeSTT**: Forked and extended from [KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT). Huge thanks to [@KoljaB](https://github.com/KoljaB) for the original low-latency STT recording system used in the Hybrid and Local examples.

## ⚙️ Installation

### Requirements

* **Python 3.10+**
* [`uv`](https://github.com/astral-sh/uv) — A fast, Rust-based Python project manager.
* **Ollama** (for the local example). [Download here](https://ollama.com/).


### 1. Install `uv`

#### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```


#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation: `uv --version`

### 2. Clone \& Set Up the Project

From the root directory (where `pyproject.toml` is located):

```bash
uv sync
```

This command creates a virtual environment and installs all dependencies listed in `pyproject.toml`.

### 3. Set Up Local Models (For Local Pipeline)

If you plan to run the fully local example, you must first install and run Ollama. Then, pull the required model:

```bash
ollama pull gemma3:1b
```


## 🔑 Environment Setup

Create a `.env` file in the project root. Add the API keys required for the examples you wish to run. The local pipeline does not require any keys.

```ini
# For the Gemini Hybrid Pipeline
GOOGLE_API_KEY="your_google_api_key_here"

# For the AWS Modular and Nova-Sonic Pipelines
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_REGION="us-east-1" # Or your preferred region
```

The applications use `python-dotenv` to load these keys automatically.

## 🧪 Running the Examples

Run the desired agent from the **project root** to ensure correct module paths.

#### 1. Run the AWS Modular Pipeline

```bash
uv run python -m examples.audio-to-audio-aws
```


#### 2. Run the Gemini Hybrid Pipeline

```bash
uv run python -m examples.audio-to-audio-gemini
```


#### 3. Run the Nova-Sonic S2S Pipeline

```bash
uv run python -m examples.audio-to-audio-nova-sonic
```


#### 4. Run the Fully Local Pipeline

*Ensure the Ollama application is running before executing this command.*

```bash
uv run python -m examples.audio-to-audio-local
```


## 📁 Directory Structure

```
.
├── examples/
│   ├── audio-to-audio-aws.py         # Modular Cloud Agent
│   ├── audio-to-audio-gemini.py      # Hybrid Agent (Cloud LLM + Local TTS)
│   ├── audio-to-audio-nova-sonic.py  # End-to-End S2S Agent
│   └── audio-to-audio-local.py       # Fully Local Agent
├── RealtimeSTT/
│   ├── ...
├── .env                            # Environment variables (API keys)
├── pyproject.toml                  # Project dependencies & metadata
└── uv.lock                         # Locked package versions
```


## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

## 📄 License

MIT License. See `LICENSE` file for details.

