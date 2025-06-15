# 🎙️ Audio Agent

This project builds a real-time voice agent that:

1. Transcribes speech using `RealtimeSTT`
2. Generates a response using **Gemini 2.0 Flash** via **LangChain**
3. Speaks the response using the `Kokoro-82M` TTS model

---

## 🙏 Acknowledgements

🔊 **RealtimeSTT**: Forked and extended from [KoljaB/RealtimeSTT](https://github.com/KoljaB/RealtimeSTT).  
Huge thanks to [@KoljaB](https://github.com/KoljaB) for the original low-latency STT recording system.

---

## 🚀 Requirements

- **Python 3.10+**
- [`uv`](https://github.com/astral-sh/uv) — Rust-based Python project manager

---

## ⚙️ Installation

### 1. Install `uv`

#### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:

```bash
uv --version
```

---

### 2. Clone & Set Up the Project

From the root directory (where `pyproject.toml` is located):

```bash
uv sync
```

This installs all dependencies listed in `pyproject.toml` using `uv`.

If you're starting fresh and `pyproject.toml` doesn't exist yet:

```bash
uv init
uv add -r requirements.txt
```

---

## 📁 Directory Structure

```
.
├── examples/
│   └── audio-to-audio.py       # Main voice agent script
├── RealtimeSTT/                # Custom STT recorder module
│   ├── audio_recorder.py
│   └── __init__.py
├── .env                        # Environment variables (e.g., API keys)
├── pyproject.toml              # Dependency & project metadata
├── uv.lock                     # Exact locked package versions
```

---

## 🧪 Run the Voice Agent

Run from the **project root**:

```bash
uv run python -m examples.audio-to-audio_gemini
```

This ensures the correct import path is used (i.e., `RealtimeSTT` is in scope).

---

## 🔑 Environment Setup

Create a `.env` file in the root with necessary keys:

```ini
GOOGLE_API_KEY=your_key_here
```

Use `python-dotenv` or equivalent to load this automatically.

---

## 🧹 Common `uv` Commands

| Command               | Description                                |
| --------------------- | ------------------------------------------ |
| `uv add <package>`    | Add new package to project                 |
| `uv remove <package>` | Remove a package                           |
| `uv sync`             | Sync env from `pyproject.toml` / `uv.lock` |
| `uv run <command>`    | Run command inside virtual environment     |
| `uv lock`             | Rebuild the lock file                      |

---

## ✅ Quick Start

| Step         | Command                                    |
| ------------ | ------------------------------------------ |
| Install uv   | (see above)                                |
| Install deps | `uv sync`                                  |
| Run agent    | `uv run python -m examples.audio-to-audio` |

---

## 🎧 What Happens

1. You speak into the mic.
2. `RealtimeSTT` transcribes your input.
3. LangChain + Gemini 2.0 Flash generates a spoken-style reply.
4. `Kokoro-82M` TTS speaks the response.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

---

## 📄 License

MIT License. See `LICENSE` file for details.

