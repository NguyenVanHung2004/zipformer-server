# ZipFormer Realtime ASR Server

Server WebSocket nhận diện giọng nói (Speech-to-Text) realtime đa ngôn ngữ, dựa trên **Zipformer Transducer** thông qua [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Hỗ trợ **Tiếng Việt** (mặc định, dùng model fine-tuned nội bộ) và **Tiếng Anh** (GigaSpeech Zipformer). Tối ưu cho CPU, INT8 quantization, chạy tốt trên các instance free-tier của Zeabur.

---

## ✨ Tính năng

- 🎙️ **Streaming ASR qua WebSocket** – gửi PCM 16-bit mono 16 kHz, nhận về JSON theo format Deepgram (`channel.alternatives[].transcript`).
- 🇻🇳 **Vietnamese**: dùng model Zipformer INT8 được fine-tune nội bộ (`model_vi_fine_tune/local/*_epoch-3-avg-1_scaled_0.1.onnx`).
- 🇬🇧 **English**: tự động tải model GigaSpeech Zipformer (`model_en/`).
- 🔇 **Silero VAD** – threshold 0.55, cắt câu theo khoảng lặng (≥ 0.8s) → trả về bản `is_final: true`.
- ⏱️ **Partial decode** mỗi ~0.4s (dynamic interval theo độ dài audio) → cập nhật `is_final: false` liên tục.
- 🛡️ **Forced segmentation** nếu người dùng nói liên tục > 8 giây không im lặng.
- 🎛️ **DSP pipeline**: Bandpass 150 Hz – 3.4 kHz + AGC (target RMS ~ −20 dB, max gain 10×) + soft clipping `tanh`.
- 🧹 **Hallucination filter** loại bỏ các filler "ừ/à/ờ/um/uh" lặp lại.
- 📝 **Word-level timestamps** (start/end tuyệt đối tính từ đầu session) kèm `speaker` tag.
- 🧑‍🤝‍🧑 **Smart paragraphing** – tự đổi `speaker` khi khoảng lặng giữa 2 câu > 2 giây.
- 🔥 **Hotwords** boosting (`hotwords.txt`, score 10.0).
- ⬇️ **Auto-download**: model VAD, Zipformer INT8 EN/VI được script tự tải về khi khởi động (chỉ cần GitHub Releases truy cập được).

---

## 🧱 Tech stack

| Thành phần | Vai trò |
|---|---|
| Python ≥ 3.9 | Runtime |
| `sherpa-onnx==1.12.20` | ONNX Runtime + Zipformer Transducer inference |
| `websockets` | WebSocket server (asyncio) |
| `numpy`, `scipy` | DSP (filter, AGC) |
| Silero VAD (ONNX) | Voice activity detection |
| Zeabur | PaaS deployment platform (Docker-based, hỗ trợ Git) |

---

## 📁 Cấu trúc thư mục

```
test_ZipFormer/
├── server.py                       # WebSocket ASR server (entrypoint)
├── requirements.txt                # Python dependencies
├── hotwords.txt                    # Từ khoá boosting
├── test_mic_client.py              # Client VI – mic realtime → server
├── test_mic_client_en.py           # Test offline model (xem file để biết chi tiết)
├── wav_1.wav                       # Audio mẫu
├── .gitignore                      # Bỏ qua model_vi/, model_en/, .venv/, __pycache__/
├── .gitattributes                  # Coi *.onnx là binary (tránh diff)
├── model_vi/                       # (gitignored) Vietnamese base Zipformer INT8 + silero_vad – auto-download
│   ├── tokens.txt
│   ├── encoder-*.int8.onnx
│   ├── decoder-*.int8.onnx
│   ├── joiner-*.int8.onnx
│   └── silero_vad.onnx
├── model_en/                       # (gitignored) English GigaSpeech – auto-download
├── model_hy/                       # (tracked) Fallback Zipformer fine-tune cũ, FP32
│   ├── token.txt
│   ├── encoder-epoch-20-avg-10.onnx
│   ├── decoder-epoch-20-avg-10.onnx
│   └── joiner-epoch-20-avg-10.onnx
└── model_ft/                       # Vietnamese fine-tune mới nhất – auto-download từ GitHub Releases
    ├── tokens.txt
    ├── encoder-epoch-*-avg-*.int8.onnx
    ├── decoder-epoch-*-avg-*.int8.onnx
    └── joiner-epoch-*-avg-*.int8.onnx
```

> ⚠️ `model_vi/` và `model_en/` đã `.gitignore` và được script trong `server.py` tải tự động khi khởi động lần đầu (xem `check_and_download_models()` và `check_and_download_en_models()`).
>
> `model_ft/` được `check_and_download_finetuned_model()` tự động tải từ GitHub Releases của repo [`NguyenVanHung2004/zipFormerModel`](https://github.com/NguyenVanHung2004/zipFormerModel/releases) khi server khởi động. Nếu tải fail, hệ thống fallback sang `model_hy/` (đã commit sẵn trong repo).

---

## 🚀 Cài đặt cục bộ (Local)

### 1. Yêu cầu môi trường
- Python **3.9 – 3.11** (sherpa-onnx 1.12.20 chưa hỗ trợ 3.12 wheels trên mọi OS)
- Git
- RAM tối thiểu ~1 GB (model INT8 ~120 MB tổng)
- Kết nối Internet lần đầu để auto-download model

### 2. Clone & setup

```bash
git clone <your-repo-url> test_ZipFormer
cd test_ZipFormer

# Tạo virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

### 3. Model Vietnamese — đã có sẵn `model_hy/` fallback trong repo

Repo đã commit sẵn `model_hy/` (fine-tune cũ, FP32, dùng làm fallback). Khi chạy server:

1. `check_and_download_finetuned_model()` được gọi → tải `model_ft/` (INT8 mới nhất) từ GitHub Releases của [`NguyenVanHung2004/zipFormerModel`](https://github.com/NguyenVanHung2004/zipFormerModel/releases).
2. Nếu tải thành công → server dùng `model_ft/`.
3. Nếu fail (offline, rate-limit, …) → server tự động fallback sang `model_hy/`.

Tóm lại: bạn **không cần tự tải model** — chỉ cần internet lần đầu.

### 4. Khởi động server

```bash
python server.py
```

Output mong đợi (lần đầu sẽ tải model ~2-5 phút tuỳ băng thông):

```
⏳ Downloading https://...sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2 to asr_model.tar.bz2...
✅ Downloaded (12345.67 KB)
📦 Extracting ASR...
✅ INT8 ASR Model Ready
🔍 Checking for latest fine-tuned model at https://api.github.com/repos/NguyenVanHung2004/zipFormerModel/releases/latest
🆕 New fine-tuned model found: v2.0 ... Downloading...
✅ All 4 int8 components found.
🎉 Fine-tuned model updated to v2.0!
⏳ Loading Vietnamese Recognizer...
⏳ Loading English Recognizer...
✅ Systems Ready!
🚀 Server UPDATED VERSION (Async Partial Decode) đang lắng nghe tại ws://0.0.0.0:6006
```

Custom port:

```bash
PORT=8765 python server.py
```

---

## ☁️ Deploy lên Zeabur bằng Git

Zeabur cho phép deploy trực tiếp từ GitHub repo bằng cách detect `Dockerfile` hoặc `Python` project (tự dựng runtime từ `requirements.txt`). Dưới đây là flow khuyến nghị.

### 1. Chuẩn bị `Dockerfile` (khuyến nghị)

Tạo file `Dockerfile` ở thư mục gốc (cùng cấp `server.py`):

```dockerfile
FROM python:3.11-slim

# Cần cho sherpa-onnx wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Lắng nghe mọi interface, PORT sẽ do Zeabur set env
ENV PORT=6006
EXPOSE 6006

# Health check đơn giản (Zeabur sẽ tự check)
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import socket,sys;s=socket.socket();s.settimeout(3);s.connect(('localhost',int(__import__('os').getenv('PORT','6006'))));s.close()" || exit 1

CMD ["python", "server.py"]
```

Commit & push:

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for Zeabur deployment"
git push origin main
```

### 2. Tạo project trên Zeabur

1. Truy cập https://zeabur.com → **New Project**.
2. Chọn **Deploy from GitHub** → chọn repo vừa push.
3. Zeabur sẽ tự detect `Dockerfile` và bắt đầu build.
4. Sau khi build xong, vào tab **Networking** của service:
   - Tạo một **TCP Public** port mapping `6006 → 6006` (hoặc map ra một port public bất kỳ, ví dụ `44390`).
   - Hoặc dùng **Custom Domain** nếu bạn đã trỏ domain.

> Lưu ý: WebSocket cần được expose qua TCP, không phải HTTP/HTTPS endpoint. Trên Zeabur, **TCP Public** port là lựa chọn đúng.

### 3. Cấu hình biến môi trường (tuỳ chọn)

Vào tab **Variables** của service trên Zeabur, thêm nếu muốn:

| Biến | Mặc định | Mô tả |
|---|---|---|
| `PORT` | `6006` | Zeabur sẽ tự inject port nếu dùng auto-binding |
| `GITHUB_TOKEN` | _(không có)_ | _(không dùng — đã bỏ phần auto-update từ GitHub Releases)_ |

### 4. Theo dõi log

Trong tab **Logs** của service, chờ tới khi thấy:

```
🚀 Server UPDATED VERSION (Async Partial Decode) đang lắng nghe tại ws://0.0.0.0:6006
```

Tức là đã sẵn sàng. Lưu lại URL public (ví dụ `tcp.zeabur.io:12345`) để dùng cho client.

### 5. Update/Re-deploy

Mỗi lần push lên nhánh đã chọn (mặc định `main`):

```bash
git add .
git commit -m "update: ..."
git push origin main
```

Zeabur sẽ tự động detect commit mới, rebuild và restart service. Model đã tải về container trước đó sẽ bị mất vì container stateless – hãy cân nhắc dùng **Zeabur Volume** mount vào `/app/model_vi`, `/app/model_en` để cache model giữa các lần deploy (khuyến nghị cho free-tier khỏi tải lại mỗi lần).

---

## 🔌 Kết nối & sử dụng WebSocket

### Endpoint

```
ws://<host>:<port>                # Mặc định VI
ws://<host>:<port>?language=en    # English
ws://<host>:<port>?language=vi    # Vietnamese (tường minh)
```

Ví dụ:

```
ws://localhost:6006
ws://tcp.zeabur.io:12345
wss://asr.your-domain.com
```

### Audio format (Client → Server)

| Trường | Giá trị |
|---|---|
| Encoding | PCM |
| Sample width | 16-bit (little-endian) |
| Channels | 1 (mono) |
| Sample rate | 16000 Hz |

Client gửi **raw bytes** của mảng `int16` qua `WebSocket.send()`. Không cần header, không JSON wrapping. Mỗi message nên là một chunk ~20 – 100 ms (320 – 1600 samples). Server tự gộp nối.

### Response format (Server → Client)

Mỗi JSON message có dạng tương thích **Deepgram Nova**:

```json
{
  "channel": {
    "alternatives": [
      {
        "transcript": "xin chào các bạn",
        "confidence": 0.5,
        "speaker": 0,
        "words": []
      }
    ]
  },
  "is_final": false
}
```

#### Partial transcript (`is_final: false`)

```json
{
  "channel": {
    "alternatives": [{
      "transcript": "xin chào",
      "confidence": 0.5
    }]
  },
  "is_final": false
}
```
- Gửi liên tục trong khi người dùng đang nói (mỗi ~0.4s).
- Không chứa `words[]` và không chứa `speaker`.

#### Final transcript (`is_final: true`)

```json
{
  "channel": {
    "alternatives": [{
      "transcript": "xin chào các bạn",
      "confidence": 1.0,
      "speaker": 0,
      "words": [
        {
          "word": "xin",
          "start": 0.12,
          "end": 0.28,
          "confidence": 1.0,
          "speaker": 0
        },
        {
          "word": "chào",
          "start": 0.28,
          "end": 0.55,
          "confidence": 1.0,
          "speaker": 0
        }
      ]
    }]
  },
  "is_final": true
}
```
- Trigger khi VAD phát hiện kết thúc câu (im lặng ≥ 0.8s) **hoặc** khi buffer đạt 8 giây (forced segmentation).
- `words[].start` và `words[].end` là **timestamp tuyệt đối** tính từ khi client bắt đầu stream (đơn vị: giây).
- `speaker` đổi 0 ↔ 1 khi khoảng lặng giữa 2 câu > 2 giây (heuristic paragraph segmentation, không phải diarization thật).

### Ví dụ client bằng Python

```python
import asyncio, websockets, sounddevice as sd, numpy as np, json

URI = "ws://localhost:6006"           # VI mặc định
# URI = "ws://localhost:6006?language=en"

async def main():
    async with websockets.connect(URI) as ws:
        loop = asyncio.get_running_loop()
        q = asyncio.Queue()

        def cb(indata, frames, time, status):
            loop.call_soon_threadsafe(q.put_nowait, indata.copy())

        stream = sd.InputStream(channels=1, dtype="int16",
                                samplerate=16000, callback=cb)
        stream.start()

        async def sender():
            while True:
                chunk = await q.get()
                await ws.send(chunk.tobytes())

        async def receiver():
            async for msg in ws:
                data = json.loads(msg)
                if "channel" in data:
                    alt = data["channel"]["alternatives"][0]
                    is_final = data.get("is_final", False)
                    prefix = "✅" if is_final else "⏳"
                    print(f"{prefix} {alt['transcript']}")

        await asyncio.gather(sender(), receiver())

asyncio.run(main())
```

File mẫu đầy đủ: xem `test_mic_client.py` (VI) và `test_mic_client_en.py` (test offline model).

---

## 🧪 Test nhanh

```bash
# Chạy server ở terminal 1
python server.py

# Chạy mic client ở terminal 2 (VI)
python test_mic_client.py
```

Nói vào mic, terminal sẽ hiển thị partial transcript realtime và từng final transcript khi bạn ngưng nói.

---

## 🔧 Tuỳ chỉnh & troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| Build fail trên Zeabur vì thiếu `g++` | Nếu dùng `Dockerfile` với `sherpa-onnx` wheel prebuilt → không cần C compiler. Còn nếu Zeabur tự build từ `requirements.txt` thì wheel vẫn có sẵn trên PyPI; nếu vẫn fail, thêm `RUN apt-get install -y build-essential` trước `pip install`. |
| Model tải lại mỗi lần deploy | Mount Zeabur Volume vào `/app/model_vi` và `/app/model_en` để cache auto-download. `model_hy/` đã commit sẵn nên fallback luôn có. |
| `model_ft/` không tải được | Set biến môi trường `GITHUB_TOKEN` để tăng rate-limit. Nếu vẫn fail, server tự fallback sang `model_hy/` (FP32). |
| Quá nhiều false positive từ VAD | Tăng `vad_config.silero_vad.threshold` (hiện 0.55) trong `server.py`. |
| Câu dài bị nuốt mất cuối | Server có **forced segmentation** tại 8s nếu không có khoảng lặng. |
| Muốn hotwords riêng | Sửa `hotwords.txt` (mỗi dòng một từ), hotwords score đang là `10.0`. |
| Test EN nhưng tải model fail | Kiểm tra outbound HTTPS tới `github.com` và `huggingface.co`. Model `model_en/` là optional, server vẫn chạy nếu thiếu. |

---

## 📜 License

MIT (hoặc tuỳ bạn cập nhật). Model Zipformer của k2-fsa/sherpa-onnx tuân theo license tương ứng trên GitHub Releases của họ.
