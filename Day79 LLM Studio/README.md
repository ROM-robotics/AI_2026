# ◆ LLM Studio — Terminal-based LLM Playground

A **TUI (Terminal User Interface)** application for running, managing, and serving Large Language Models locally. Inspired by LM Studio, built entirely for the terminal.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ◆ LLM Studio v1.0                                                     │
│                                                                         │
│  ┌──────────┐  ┌─────────────────────────────────────────────────────┐  │
│  │ 🏠 Home   │  │  ╔══════════════════════════════════════════════╗   │  │
│  │ 💬 Chat   │  │  ║     ◆  Welcome to LLM Studio  ◆            ║   │  │
│  │ 📦 Models │  │  ║     Your Local LLM Playground               ║   │  │
│  │ 🌐 Server │  │  ╚══════════════════════════════════════════════╝   │  │
│  │ ⚙️ Config  │  │                                                     │  │
│  └──────────┘  │  [📦 Models: 3]  [🌐 Server: OFF]  [🖥 CPU: 8]    │  │
│                 └─────────────────────────────────────────────────────┘  │
│  ◆ No model loaded          Server: OFF              Ctrl+Q: Quit       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌──────────────────────────────────────────┐
│           TUI Frontend (Textual)         │   ← Python + Rich/Textual
├──────────────────────────────────────────┤
│         Application Logic (Python)       │   ← Config, Model Manager
├──────────────────────────────────────────┤
│      API Server (FastAPI + Uvicorn)      │   ← OpenAI-compatible REST API
├──────────────────────────────────────────┤
│    Inference Engine (llama-cpp-python)    │   ← C++ (llama.cpp) bindings
└──────────────────────────────────────────┘
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **TUI/Interface** | Python + Textual | Beautiful terminal UI with mouse support |
| **Backend** | Python + FastAPI | Model management & API server |
| **Inference Engine** | llama.cpp (via Python bindings) | High-performance C++ LLM inference |
| **Model Format** | GGUF | Quantized model format for efficient CPU/GPU inference |

## Features

### 🏠 Home Dashboard
- System overview with model count, server status, storage info
- Quick-action buttons for navigation

### 💬 Interactive Chat
- Real-time streaming responses with token-by-token display
- Chat history management
- Configurable system prompt

### 📦 Model Management
- **Browse local models** — list all downloaded GGUF files
- **Search HuggingFace** — find GGUF models from HF repos
- **Download models** — download directly with progress bar
- **Load/Unload** — manage model in memory
- **Delete models** — clean up storage

### 🌐 OpenAI-Compatible API Server
- Start/stop server from the TUI
- **Full OpenAI API compatibility:**
  - `POST /v1/chat/completions` (streaming supported)
  - `POST /v1/completions` (streaming supported)
  - `POST /v1/embeddings`
  - `GET /v1/models`
- Optional API key authentication
- CORS support
- Works with any OpenAI client library

### ⚙️ Settings
- Temperature, Top-P, Top-K, Repeat Penalty
- Context length, Max tokens
- CPU threads, GPU layers
- Batch size, Random seed
- System prompt
- Model storage directory
- All settings persisted to `~/.llm_studio/config.yaml`

## Installation

### 1. Clone & Install

```bash
cd "Day79 LLM Studio"
pip install -r requirements.txt
```

### 2. For GPU Support (Optional)

```bash
# CUDA support
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Metal (macOS) support
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Install as Package (Optional)

```bash
pip install -e .
```

## Usage

### Launch the TUI

```bash
python run.py
# or if installed as package:
llm-studio
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F1` | Home screen |
| `F2` | Chat screen |
| `F3` | Models screen |
| `F4` | Server screen |
| `F5` | Settings screen |
| `Ctrl+Q` | Quit |
| `Ctrl+T` | Toggle dark/light theme |
| `Tab` | Navigate between elements |
| `Enter` | Send message / Activate button |

### Quick Start

1. **Launch** → `python run.py`
2. **Go to Models** (F3) → Download tab → Enter a repo ID like `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
3. **Search** → Select a quantization (e.g., Q4_K_M) → Download
4. **Local Models** tab → Select model → Click "Load"
5. **Go to Chat** (F2) → Start chatting!
6. **Optionally** → Go to Server (F4) → Start the OpenAI-compatible API

### Using the API Server

Once the server is running, use any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",  # or your configured key
)

response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Or with `curl`:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Project Structure

```
Day79 LLM Studio/
├── run.py                          # Entry point
├── setup.py                        # Package setup
├── requirements.txt                # Dependencies
├── README.md
└── llm_studio/
    ├── __init__.py
    ├── app.py                      # Main TUI application
    ├── config.py                   # Configuration management
    ├── models/
    │   ├── __init__.py
    │   ├── manager.py              # Model download/list/delete
    │   └── engine.py               # Inference engine (llama.cpp)
    ├── server/
    │   ├── __init__.py
    │   └── api.py                  # OpenAI-compatible REST API
    └── ui/
        ├── __init__.py
        ├── widgets/
        │   ├── __init__.py
        │   ├── sidebar.py          # Navigation sidebar
        │   ├── message_list.py     # Chat message bubbles
        │   └── status_bar.py       # Bottom status bar
        ├── screens/
        │   ├── __init__.py
        │   ├── home.py             # Dashboard
        │   ├── chat.py             # Chat interface
        │   ├── models.py           # Model management
        │   ├── server.py           # API server control
        │   └── settings.py         # Configuration
        └── styles/
            └── app.tcss            # Textual CSS styles
```

## Configuration

Settings are stored in `~/.llm_studio/config.yaml`:

```yaml
models_dir: ~/.llm_studio/models
theme: dark
system_prompt: You are a helpful AI assistant.
inference:
  n_ctx: 4096
  n_threads: 4
  n_gpu_layers: 0
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  repeat_penalty: 1.1
  max_tokens: 2048
  n_batch: 512
  seed: -1
server:
  host: 0.0.0.0
  port: 1234
  api_key: null
  cors_origins:
    - "*"
```

## Requirements

- Python 3.10+
- A terminal with Unicode support (most modern terminals)
- GGUF model files (download via the app or manually place in `~/.llm_studio/models/`)

## License

MIT
