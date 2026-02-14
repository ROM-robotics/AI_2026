# 🚀 LLM Studio — How to Run

## 1. Dependencies Install လုပ်ပါ

```bash
cd "Day79 LLM Studio"
pip install -r requirements.txt
```

## 2. GPU Support (Optional)

CPU တစ်ခုတည်းနဲ့ run လို့ရပါတယ်။ GPU နဲ့ မြန်ချင်ရင်အောက်ကအတိုင်း install လုပ်ပါ။

### NVIDIA GPU (CUDA)
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### macOS Apple Silicon (Metal)
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## 3. App ကို Run ပါ

```bash
python run.py
```

ဒါဆိုရင် Terminal ထဲမှာ LLM Studio TUI ပွင့်လာပါမယ်။

---

## 4. App ထဲရောက်ရင် — Step by Step

### Step 1: Model Download လုပ်ပါ

1. `F3` နှိပ်ပါ (Models screen)
2. **"Download from HuggingFace"** tab ကို သွားပါ
3. Search box မှာ HuggingFace repo ID ထည့်ပါ — ဥပမာ:
   ```
   TheBloke/Mistral-7B-Instruct-v0.2-GGUF
   ```
4. **🔍 Search** နှိပ်ပါ
5. GGUF file list ထဲက ကြိုက်တဲ့ quantization ရွေးပါ (ဥပမာ `Q4_K_M`)
6. **⬇ Download Selected** နှိပ်ပါ
7. Download ပြီးတဲ့အထိ စောင့်ပါ

### Step 2: Model Load လုပ်ပါ

1. **"Local Models"** tab ကို ပြန်သွားပါ
2. Download ထားတဲ့ model ကို cursor နဲ့ ရွေးပါ
3. **▶ Load Selected** နှိပ်ပါ
4. Model load ပြီးရင် status bar မှာ model name ပေါ်လာပါမယ်

### Step 3: Chat လုပ်ပါ

1. `F2` နှိပ်ပါ (Chat screen)
2. အောက်ဆုံး input box မှာ message ရိုက်ပါ
3. `Enter` နှိပ်ပါ (ဒါမှမဟုတ် **Send** button နှိပ်ပါ)
4. Assistant ရဲ့ response ကို token-by-token streaming နဲ့ ပြပေးပါမယ်

### Step 4: API Server ဖွင့်ပါ (Optional)

1. `F4` နှိပ်ပါ (Server screen)
2. Host, Port, API Key ပြင်ချင်ရင် ပြင်ပါ (default: `0.0.0.0:1234`)
3. **▶ Start Server** နှိပ်ပါ
4. Server ဖွင့်ပြီးရင် OpenAI-compatible API ကို အသုံးပြုလို့ရပါပြီ

---

## 5. API Server ကို အသုံးပြုပုံ

### Python (OpenAI Client)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
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

### curl

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/models` | Model list |
| `POST` | `/v1/chat/completions` | Chat (streaming supported) |
| `POST` | `/v1/completions` | Text completion (streaming supported) |
| `POST` | `/v1/embeddings` | Embeddings |
| `GET` | `/health` | Health check |

---

## 6. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F1` | Home (Dashboard) |
| `F2` | Chat |
| `F3` | Models |
| `F4` | Server |
| `F5` | Settings |
| `Ctrl+Q` | Quit |
| `Ctrl+T` | Dark/Light Theme Toggle |
| `Tab` | Navigate between UI elements |
| `Enter` | Send message / Activate button |

---

## 7. Settings ပြင်ချင်ရင်

`F5` နှိပ်ပြီး Settings screen မှာ ပြင်လို့ရပါတယ်:

| Setting | Default | Description |
|---------|---------|-------------|
| Context Length | 4096 | Max context window (tokens) |
| Max Tokens | 2048 | Max response length |
| Temperature | 0.7 | Randomness (0 = deterministic) |
| Top P | 0.9 | Nucleus sampling |
| Top K | 40 | Top-K sampling |
| Threads | 4 | CPU threads |
| GPU Layers | 0 | GPU offload layers (0 = CPU only) |

**"💾 Save Settings"** နှိပ်ရင် `~/.llm_studio/config.yaml` မှာ save လုပ်ပေးပါမယ်။

---

## 8. Recommended Models (စမ်းဖို့)

| Model | Size | Repo ID |
|-------|------|---------|
| TinyLlama 1.1B | ~600MB | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` |
| Phi-2 | ~1.5GB | `TheBloke/phi-2-GGUF` |
| Mistral 7B | ~4GB | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` |
| Llama 2 7B | ~4GB | `TheBloke/Llama-2-7B-Chat-GGUF` |
| Llama 2 13B | ~7GB | `TheBloke/Llama-2-13B-chat-GGUF` |

> **Tip:** `Q4_K_M` quantization က quality နဲ့ size balance ကောင်းပါတယ်။ RAM နည်းရင် `Q2_K` ကို သုံးပါ။

---

## Troubleshooting

### "llama-cpp-python is not installed" error
```bash
pip install llama-cpp-python
```

### Model load မရရင်
- RAM လုံလောက်ရဲ့လား စစ်ပါ (7B model = ~4GB RAM လိုပါတယ်)
- GGUF format ဟုတ်ရဲ့လား စစ်ပါ

### Server start မရရင်
- Port 1234 ကို တခြား program သုံးနေသလား စစ်ပါ
- Model load ပြီးမှ server start လုပ်ပါ
