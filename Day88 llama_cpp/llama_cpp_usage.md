# llama.cpp အသုံးပြုနည်း (How to Use llama.cpp)

## မိတ်ဆက် (Introduction)

llama.cpp သည် Meta ၏ LLaMA models များကို C/C++ ဖြင့် ရေးသားထားသော inference engine တစ်ခုဖြစ်ပြီး၊ CPU ပေါ်တွင် efficient တွင် run နိုင်သည့် tool ဖြစ်ပါတယ်။

## အဓိက Features များ

- ✅ Pure C/C++ implementation
- ✅ Apple Silicon support
- ✅ AVX, AVX2, AVX512 support
- ✅ GPU support (CUDA, Metal, OpenCL)
- ✅ 4-bit, 5-bit, 8-bit quantization
- ✅ Low memory requirements

## Installation

### Linux/Ubuntu တွင် Install လုပ်ခြင်း

```bash
# Dependencies များ install လုပ်ခြင်း
sudo apt-get update
sudo apt-get install build-essential git cmake

# Repository ကို clone လုပ်ခြင်း
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build လုပ်ခြင်း
make

# GPU support (CUDA) ပါ build လုပ်လိုပါက
make LLAMA_CUDA=1
```

### macOS တွင် Install လုပ်ခြင်း

```bash
# Homebrew install လုပ်ထားရမည်
brew install cmake

# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Apple Silicon (Metal) support ပါ build လုပ်ရန်
make LLAMA_METAL=1
```

## Model Download နှင့် Conversion

### Step 1: Model Download လုပ်ခြင်း

```bash
# Hugging Face မှ model download လုပ်ခြင်း
# Example: LLaMA 2 7B model
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

### Step 2: Model ကို GGUF format သို့ convert လုပ်ခြင်း

```bash
# Python dependencies များ install လုပ်ခြင်း
pip install -r requirements.txt

# Model ကို convert လုပ်ခြင်း
python convert.py /path/to/Llama-2-7b-chat-hf

# Quantize လုပ်ခြင်း (memory သက်သာစေရန်)
./quantize /path/to/model.gguf /path/to/model-q4_0.gguf q4_0
```

## အခြေခံ Usage

### Command Line မှ အသုံးပြုခြင်း

```bash
# Basic inference
./main -m models/llama-2-7b-chat.Q4_0.gguf -p "Hello, how are you?" -n 128

# Interactive mode
./main -m models/llama-2-7b-chat.Q4_0.gguf -i

# With more parameters
./main -m models/llama-2-7b-chat.Q4_0.gguf \
  -p "Explain quantum computing in simple terms" \
  -n 512 \
  --temp 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.1
```

### အရေးကြီးသော Parameters များ

- `-m`: Model file path
- `-p`: Prompt text
- `-n`: Number of tokens to generate (default: 128)
- `-t`: Number of threads to use (default: 4)
- `-c`: Context size (default: 512)
- `--temp`: Temperature (0.0-2.0, default: 0.8)
- `--top-k`: Top-K sampling (default: 40)
- `--top-p`: Top-P sampling (default: 0.9)
- `--repeat-penalty`: Penalty for repeating tokens (default: 1.1)

## Python Bindings အသုံးပြုခြင်း

### Installation

```bash
pip install llama-cpp-python
```

### Basic Example

```python
from llama_cpp import Llama

# Model load လုပ်ခြင်း
llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_0.gguf",
    n_ctx=2048,  # Context window
    n_threads=4,  # Number of CPU threads
    n_gpu_layers=35  # GPU layers (if using GPU)
)

# Inference
output = llm(
    "Q: What is the capital of Myanmar? A: ",
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    echo=True
)

print(output['choices'][0]['text'])
```

### Chat Completion Example

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_0.gguf",
    n_ctx=2048,
    chat_format="llama-2"
)

# Chat completion
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=256
)

print(response['choices'][0]['message']['content'])
```

### Streaming Example

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/llama-2-7b-chat.Q4_0.gguf")

# Streaming output
stream = llm(
    "Write a short story about a robot: ",
    max_tokens=500,
    stream=True
)

for output in stream:
    text = output['choices'][0]['text']
    print(text, end='', flush=True)
```

## Advanced Usage

### Server Mode (REST API)

```bash
# Start server
./server -m models/llama-2-7b-chat.Q4_0.gguf --host 0.0.0.0 --port 8080

# Python မှ API call လုပ်ခြင်း
import requests

response = requests.post(
    "http://localhost:8080/completion",
    json={
        "prompt": "What is AI?",
        "n_predict": 128,
        "temperature": 0.7
    }
)
print(response.json())
```

### Batch Processing

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/llama-2-7b-chat.Q4_0.gguf")

prompts = [
    "What is Python?",
    "Explain machine learning.",
    "What is deep learning?"
]

for prompt in prompts:
    output = llm(prompt, max_tokens=128)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {output['choices'][0]['text']}")
    print("-" * 50)
```

## Quantization အမျိုးအစားများ

| Type | Bits | Size Reduction | Quality |
|------|------|----------------|---------|
| q4_0 | 4-bit | ~75% smaller | Good |
| q4_1 | 4-bit | ~75% smaller | Better |
| q5_0 | 5-bit | ~70% smaller | Very Good |
| q5_1 | 5-bit | ~70% smaller | Excellent |
| q8_0 | 8-bit | ~50% smaller | Near Original |

### Quantization လုပ်နည်း

```bash
# q4_0 quantization (recommended for most cases)
./quantize model.gguf model-q4_0.gguf q4_0

# q5_1 quantization (better quality)
./quantize model.gguf model-q5_1.gguf q5_1

# q8_0 quantization (highest quality)
./quantize model.gguf model-q8_0.gguf q8_0
```

## Performance Optimization

### CPU Optimization

```bash
# Use all available threads
./main -m model.gguf -t $(nproc) -p "prompt"

# Enable AVX2
make LLAMA_AVX2=1

# Enable AVX512
make LLAMA_AVX512=1
```

### GPU Acceleration

```bash
# CUDA support (NVIDIA)
make LLAMA_CUDA=1
./main -m model.gguf -ngl 35 -p "prompt"

# Metal support (Apple Silicon)
make LLAMA_METAL=1
./main -m model.gguf -ngl 1 -p "prompt"

# OpenCL support
make LLAMA_CLBLAST=1
```

## Practical Examples

### Example 1: Myanmar Language Q&A

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/llama-2-7b-chat.Q4_0.gguf",
    n_ctx=2048
)

prompt = """
Question: မြန်မာနိုင်ငံ၏ မြို့တော်ကို ဘာလို့ ရန်ကုန်မှ နေပြည်တော်သို့ ပြောင်းလဲခဲ့သနည်း?
Answer: """

output = llm(prompt, max_tokens=256, temperature=0.7)
print(output['choices'][0]['text'])
```

### Example 2: Code Generation

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/codellama-7b.Q4_0.gguf")

prompt = """
Write a Python function to calculate fibonacci numbers:
"""

output = llm(prompt, max_tokens=512, temperature=0.2)
print(output['choices'][0]['text'])
```

### Example 3: Text Summarization

```python
from llama_cpp import Llama

llm = Llama(model_path="./models/llama-2-7b-chat.Q4_0.gguf")

text = """
[Long text here...]
"""

prompt = f"Summarize the following text in 3 bullet points:\n\n{text}\n\nSummary:"

output = llm(prompt, max_tokens=256)
print(output['choices'][0]['text'])
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce context size
   ./main -m model.gguf -c 512 -p "prompt"
   
   # Use smaller quantization
   ./quantize model.gguf model-q4_0.gguf q4_0
   ```

2. **Slow Inference**
   ```bash
   # Increase threads
   ./main -m model.gguf -t 8 -p "prompt"
   
   # Use GPU
   ./main -m model.gguf -ngl 35 -p "prompt"
   ```

3. **Model Loading Failed**
   ```bash
   # Check model file integrity
   ls -lh models/
   
   # Re-download or re-convert model
   ```

## Resources

- **Official Repository**: https://github.com/ggerganov/llama.cpp
- **Documentation**: https://github.com/ggerganov/llama.cpp/wiki
- **Python Bindings**: https://github.com/abetlen/llama-cpp-python
- **Pre-quantized Models**: https://huggingface.co/TheBloke

## Conclusion

llama.cpp သည် powerful ပြီး efficient သော LLM inference tool ဖြစ်ပါတယ်။ Local machine ပေါ်မှာ GPU မလိုပဲ LLM models များကို run နိုင်စေပါတယ်။

---

**Last Updated**: January 25, 2026
