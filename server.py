import asyncio
import websockets
import sherpa_onnx
import os
import json
import numpy as np
import logging
import sys
from collections import deque
import glob
import urllib.request
import tarfile
import re
import shutil

# --- AUTO-DOWNLOAD MODELS (Robust Version) ---
def download_file(url, target_path, min_size=1024):
    print(f"⏳ Downloading {url} to {target_path}...")
    try:
        # Use simple urlretrieve but verify size afterwards
        urllib.request.urlretrieve(url, target_path)
        
        size = os.path.getsize(target_path)
        if size < min_size:
            print(f"❌ File too small ({size} bytes). Probable 404/Error page. Deleting...")
            os.remove(target_path)
            return False
            
        print(f"✅ Downloaded ({size/1024:.2f} KB)")
        return True
    except Exception as e:
        print(f"❌ Download Failed: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return False

def check_and_download_models():
    # 1. Define URLs
    # [OPTIMIZATION] Use INT8 Quantized model (Smaller, Faster on CPU)
    asr_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-int8-2025-04-20.tar.bz2"
    
    # Alternative Mirror for VAD if GitHub fails
    vad_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    
    # 2. Check & Download VAD
    if not os.path.exists("model_vi/silero_vad.onnx") or os.path.getsize("model_vi/silero_vad.onnx") < 1024:
        os.makedirs("model_vi", exist_ok=True)
        # Try primary URL
        if not download_file(vad_url, "model_vi/silero_vad.onnx"):
            # Try mirror if primary fails
            print("⚠️ Retrying with mirror...")
            download_file("https://huggingface.co/csukuangfj/silero-vad-onnx/resolve/main/silero_vad.onnx", "model_vi/silero_vad.onnx")

    # 3. Check & Download ASR (INT8)
    # Check for INT8 specific files to avoid using old FP32 models
    if not glob.glob("model_vi/*int8.onnx"):
        print("⚡ Old/Missing model detected. Downloading INT8 Zipformer...")
        
        # Cleanup old ONNX files (except VAD) to avoid mix-ups
        for f in glob.glob("model_vi/*.onnx"):
            if "silero_vad" not in f:
                try: os.remove(f) 
                except: pass

        filename = "asr_model.tar.bz2"
        if download_file(asr_url, filename):
            print("📦 Extracting ASR...")
            try:
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(".")
                
                # Folder name usually matches tarball name (extracted directory)
                extracted_dir = "sherpa-onnx-zipformer-vi-int8-2025-04-20"
                if os.path.exists(extracted_dir):
                    os.makedirs("model_vi", exist_ok=True) 
                    
                    for f in os.listdir(extracted_dir):
                        src = os.path.join(extracted_dir, f)
                        dst = os.path.join("model_vi", f)
                        if os.path.exists(dst):
                            if os.path.isdir(dst):
                                shutil.rmtree(dst)
                            else:
                                os.remove(dst)
                        shutil.move(src, dst)
                    os.rmdir(extracted_dir)
                
                print("✅ INT8 ASR Model Ready")
            except Exception as e:
                print(f"❌ Extraction Failed: {e}")
            finally:
                if os.path.exists(filename): os.remove(filename)

# Run check immediately
check_and_download_models()

def check_and_download_en_models():
    """Download English Zipformer model (GigaSpeech) into model_en/."""
    asr_en_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-gigaspeech-2023-12-12.tar.bz2"
    en_model_dir = "model_en"
    os.makedirs(en_model_dir, exist_ok=True)

    # 1. Copy shared VAD from model_vi if already downloaded
    vad_src = "model_vi/silero_vad.onnx"
    vad_dst = os.path.join(en_model_dir, "silero_vad.onnx")
    if os.path.exists(vad_src) and not os.path.exists(vad_dst):
        shutil.copy2(vad_src, vad_dst)
        print("✅ Copied silero_vad.onnx to model_en/")

    # 2. Check & Download English ASR
    if not glob.glob(os.path.join(en_model_dir, "encoder-*.onnx")):
        filename = "asr_model_en.tar.bz2"
        if download_file(asr_en_url, filename):
            print("📦 Extracting English Zipformer ASR...")
            try:
                with tarfile.open(filename, "r:bz2") as tar:
                    tar.extractall(".")

                extracted_dir = "sherpa-onnx-zipformer-gigaspeech-2023-12-12"
                if os.path.exists(extracted_dir):
                    # Clean old onnx except VAD
                    for f in glob.glob(os.path.join(en_model_dir, "*.onnx")):
                        if "silero_vad" not in f:
                            os.remove(f)
                    # Move new files
                    for f in os.listdir(extracted_dir):
                        shutil.move(os.path.join(extracted_dir, f), en_model_dir)
                    os.rmdir(extracted_dir)
                print("✅ English Zipformer ASR Model Ready")
            except Exception as e:
                print(f"❌ English ASR Extraction Failed: {e}")
            finally:
                if os.path.exists(filename): os.remove(filename)
        else:
            print("⚠️ English ASR model download failed. English transcription will be unavailable.")
    else:
        print("✅ English Zipformer model already present.")

# Run En check
check_and_download_en_models()

# --- AUTO-UPDATE: Fine-tuned Vietnamese model from GitHub Releases ---
FINETUNED_REPO = "NguyenVanHung2004/zipFormerModel"
FINETUNED_MODEL_DIR = "model_ft"
FINETUNED_VERSION_FILE = os.path.join(FINETUNED_MODEL_DIR, ".version")

def check_and_download_finetuned_model():
    """
    Polls GitHub Releases API for the latest int8 fine-tuned model.
    Downloads only if a newer version is available.
    Requires 4 components: tokens.txt, encoder*.int8.onnx, decoder*.int8.onnx, joiner*.int8.onnx
    Falls back to model_hy/ if anything goes wrong.
    """
    import json as _json
    import urllib.error

    api_url = f"https://api.github.com/repos/{FINETUNED_REPO}/releases/latest"
    print(f"🔍 Checking for latest fine-tuned model at {api_url}...")

    try:
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json",
                                                        "User-Agent": "zipformer-server"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            release = _json.loads(resp.read().decode())
    except Exception as e:
        print(f"⚠️  Cannot reach GitHub API: {e}. Skipping fine-tune model update.")
        return

    latest_tag = release.get("tag_name", "")
    if not latest_tag:
        print("⚠️  No tag found in release. Skipping.")
        return

    # Read cached version
    cached_tag = ""
    if os.path.exists(FINETUNED_VERSION_FILE):
        with open(FINETUNED_VERSION_FILE, "r") as f:
            cached_tag = f.read().strip()

    if cached_tag == latest_tag:
        print(f"✅ Fine-tuned model already up-to-date ({latest_tag}). Skipping download.")
        return

    print(f"🆕 New fine-tuned model found: {latest_tag} (cached: '{cached_tag or 'none'}'). Checking assets...")

    # Find the 4 required int8 assets
    assets = release.get("assets", [])
    asset_map = {a["name"]: a["browser_download_url"] for a in assets}

    # Identify int8 encoder/decoder/joiner by pattern
    def find_asset(prefix, suffix=".int8.onnx"):
        candidates = [name for name in asset_map if name.startswith(prefix) and name.endswith(suffix)]
        return asset_map[candidates[0]] if candidates else None

    tokens_url    = asset_map.get("tokens.txt")
    encoder_url   = find_asset("encoder")
    decoder_url   = find_asset("decoder")
    joiner_url    = find_asset("joiner")

    missing = [n for n, u in [("tokens.txt", tokens_url), ("encoder*.int8.onnx", encoder_url),
                               ("decoder*.int8.onnx", decoder_url), ("joiner*.int8.onnx", joiner_url)] if not u]
    if missing:
        print(f"⚠️  Release {latest_tag} is missing required files: {missing}. Skipping.")
        return

    print(f"✅ All 4 int8 components found. Downloading to '{FINETUNED_MODEL_DIR}/'...")
    os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)

    # Helper: extract filename from URL
    def fname(url):
        return url.split("/")[-1]

    downloads = [
        (tokens_url,  os.path.join(FINETUNED_MODEL_DIR, "tokens.txt")),
        (encoder_url, os.path.join(FINETUNED_MODEL_DIR, fname(encoder_url))),
        (decoder_url, os.path.join(FINETUNED_MODEL_DIR, fname(decoder_url))),
        (joiner_url,  os.path.join(FINETUNED_MODEL_DIR, fname(joiner_url))),
    ]

    # Remove old int8 onnx files before downloading new ones
    for old in glob.glob(os.path.join(FINETUNED_MODEL_DIR, "*.int8.onnx")):
        try: os.remove(old)
        except: pass

    all_ok = True
    for url, dst in downloads:
        if not download_file(url, dst):
            all_ok = False
            break

    if all_ok:
        # Save version tag so we don't re-download next time
        with open(FINETUNED_VERSION_FILE, "w") as f:
            f.write(latest_tag)
        print(f"🎉 Fine-tuned model updated to {latest_tag}!")
    else:
        print("❌ Fine-tune model download failed. Will fall back to model_hy/.")
        # Clean up partial downloads
        for _, dst in downloads:
            if os.path.exists(dst):
                try: os.remove(dst)
                except: pass

# Run fine-tune model check
check_and_download_finetuned_model()


# --- CONFIGURATION ---
PORT = int(os.environ.get("PORT", 6006))
# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _resolve_vi_model_paths():
    """
    Returns (tokens, encoder, decoder, joiner) paths for the Vietnamese model.
    Priority:
      1. model_ft/  — latest auto-downloaded fine-tuned int8
      2. model_hy/  — fallback (legacy fine-tune)
    """
    ft_tokens  = os.path.join(FINETUNED_MODEL_DIR, "tokens.txt")
    ft_encoders = glob.glob(os.path.join(FINETUNED_MODEL_DIR, "encoder-*.int8.onnx"))
    ft_decoders = glob.glob(os.path.join(FINETUNED_MODEL_DIR, "decoder-*.int8.onnx"))
    ft_joiners  = glob.glob(os.path.join(FINETUNED_MODEL_DIR, "joiner-*.int8.onnx"))

    if (os.path.exists(ft_tokens)
            and ft_encoders and ft_decoders and ft_joiners):
        logging.info(f"🤖 Using fine-tuned model from '{FINETUNED_MODEL_DIR}/' "
                     f"(version: {open(FINETUNED_VERSION_FILE).read().strip() if os.path.exists(FINETUNED_VERSION_FILE) else 'unknown'})")
        return ft_tokens, ft_encoders[0], ft_decoders[0], ft_joiners[0]

    # Fallback
    logging.warning(f"⚠️  '{FINETUNED_MODEL_DIR}/' incomplete or missing – falling back to model_hy/")
    return (
        "model_hy/token.txt",
        "model_hy/encoder-epoch-20-avg-10.onnx",
        "model_hy/decoder-epoch-20-avg-10.onnx",
        "model_hy/joiner-epoch-20-avg-10.onnx",
    )


def create_components():
    # --- Vietnamese Recognizer ---
    model_dir_vi = "./model_vi"

    tokens_vi, encoder_vi, decoder_vi, joiner_vi = _resolve_vi_model_paths()
    
    logging.info("⏳ Loading Vietnamese Recognizer...")
    recognizer_vi = sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens_vi,
        encoder=encoder_vi,
        decoder=decoder_vi,
        joiner=joiner_vi,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="modified_beam_search",
        max_active_paths=4,
        hotwords_file="hotwords.txt" if os.path.exists("hotwords.txt") else "",
        hotwords_score=10.0,
    )

    # --- English Recognizer ---
    model_dir_en = "./model_en"
    recognizer_en = None
    if os.path.exists(model_dir_en) and glob.glob(os.path.join(model_dir_en, "encoder-*.onnx")):
        tokens_en = os.path.join(model_dir_en, "tokens.txt")
        encoder_en = glob.glob(os.path.join(model_dir_en, "encoder-*.onnx"))[0]
        decoder_en = glob.glob(os.path.join(model_dir_en, "decoder-*.onnx"))[0]
        joiner_en = glob.glob(os.path.join(model_dir_en, "joiner-*.onnx"))[0]
        
        logging.info("⏳ Loading English Recognizer...")
        recognizer_en = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=tokens_en,
            encoder=encoder_en,
            decoder=decoder_en,
            joiner=joiner_en,
            num_threads=2,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="modified_beam_search",
            max_active_paths=4,
        )

    # --- Shared VAD ---
    vad_model = os.path.join(model_dir_vi, "silero_vad.onnx")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model
    vad_config.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

    logging.info("✅ Systems Ready!")
    return recognizer_vi, recognizer_en, vad

import scipy.signal

# --- AUDIO PREPROCESSING (DSP) ---
class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # 1. Bandpass Filter (150Hz - 3400Hz)
        # Loại bỏ tiếng ù (hum) và tiếng quạt (thường < 150Hz)
        nyquist = 0.5 * sample_rate
        low = 150.0 / nyquist # Increased from 80Hz to 150Hz to cut fan noise
        high = 3400.0 / nyquist
        self.b, self.a = scipy.signal.butter(5, [low, high], btype='band')
        self.zi = scipy.signal.lfilter_zi(self.b, self.a)
        
        # 2. AGC (Automatic Gain Control)
        self.gain = 1.0
        self.target_level = 0.1  # Target RMS ~ -20dB
        self.max_gain = 10.0     # Reduced max gain from 15.0 to 10.0 to avoid boosting noise
        self.alpha = 0.01        # Smoothing factor (Attack/Decay)

    def process(self, chunk):
        # 1. Apply Bandpass Filter
        filtered_chunk, self.zi = scipy.signal.lfilter(self.b, self.a, chunk, zi=self.zi)
        
        # 2. Apply AGC
        # [NOISE GATE FOR AGC] Ignore mostly silent chunks for gain calculation
        # If RMS is too low (noise floor), don't boost gain
        rms = np.sqrt(np.mean(filtered_chunk**2)) + 1e-6
        
        if rms > 0.003: # Only adapt gain if signal is significant (above fan noise)
            current_target_gain = self.target_level / rms
            self.gain = (1 - self.alpha) * self.gain + self.alpha * current_target_gain
            
        # Clamp gain
        self.gain = min(max(self.gain, 0.1), self.max_gain)
        
        # Apply Gain
        normalized_chunk = filtered_chunk * self.gain
        
        # Soft Clipping (tanh limit) to prevent distortion
        normalized_chunk = np.tanh(normalized_chunk)
        
        return normalized_chunk

recognizer_vi, recognizer_en, vad = create_components()

async def handle_connection(websocket):
    # Parse language from URL
    from urllib.parse import urlparse, parse_qs
    query = urlparse(websocket.request.path).query
    params = parse_qs(query)
    language = params.get("language", ["vi"])[0]
    
    selected_recognizer = recognizer_en if (language == "en" and recognizer_en) else recognizer_vi
    logging.info(f"🔗 Client connected [Lang: {language}]")
    
    import uuid
    conn_id = str(uuid.uuid4())[:8]
    logging.info(f"🔗 [{conn_id}] Client connected")
    
    model_dir = "./model_vi"
    vad_model = os.path.join(model_dir, "silero_vad.onnx")
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model
    vad_config.sample_rate = 16000
    
    # ⚡ TUNING VAD PARAMETERS (NOISE REJECTION)
    # Threshold 0.55: Cứng hơn để bỏ qua tiếng quạt (Fan Noise)
    # Speech duration 0.2: Tránh bắt các xung noise ngắn
    vad_config.silero_vad.threshold = 0.55 # Increased from 0.35
    vad_config.silero_vad.min_silence_duration = 0.8 # Slightly faster cutoff
    vad_config.silero_vad.min_speech_duration = 0.25 
    
    client_vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)
    
    # --- PSEUDO-STREAMING LOGIC ---
    rolling_buffer = [] 
    last_decode_time = 0
    DECODE_INTERVAL = 0.4 
    
    current_sentence_id = 0
    current_speaker = 0 
    last_segment_end_time = 0 
    
    # [TIMESTAMP FIX]
    total_samples_processed = 0 
    vad_start_offset_samples = 0

    is_partial_decoding = False
    
    async def run_partial_decode(buffer_copy, sentence_id):
        nonlocal is_partial_decoding, last_decode_time, current_sentence_id
        try:
            if sentence_id != current_sentence_id:
                return

            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, decode_buffer_sync, selected_recognizer, buffer_copy, language)
            
            if sentence_id != current_sentence_id:
                return
            
            if text:
                 # [FILTER] Apply same filter to Partial Results (Trim Text)
                 text = re.sub(r'\b(ừ|à|ờ|um|uh|<)(\s+\1)+\b', '', text)
                 if re.fullmatch(r'^(ừ|à|ờ|um|uh)+$', text): 
                    text = ""
                    
                 if text: # Check again after filtering
                     response = {
                        "channel": {
                            "alternatives": [{
                                "transcript": text,
                                "confidence": 0.5,
                            }]
                        },
                        "is_final": False
                    }
                 try:
                    await websocket.send(json.dumps(response, ensure_ascii=False))
                 except:
                     pass 
            
            last_decode_time = asyncio.get_event_loop().time()
            
        except Exception as e:
            logging.error(f"⚠️ Partial Decode Error: {e}")
        finally:
            is_partial_decoding = False

    # --- PREPROCESSOR ---
    preprocessor = AudioPreprocessor(sample_rate=16000)

    try:
        async for message in websocket:
            samples = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            
            # [DSP] Apply Filter & AGC
            samples = preprocessor.process(samples)

            total_samples_processed += len(samples)
            
            # max_amp calculation is now on processed samples
            # max_amp = np.max(np.abs(samples)) if len(samples) > 0 else 0
            
            # [REMOVED OLD GAIN LOGIC] AGC is now handled by preprocessor class
            
            # Clip is handled by tanh in preprocessor, but safe to clamp again
            samples = np.clip(samples, -1.0, 1.0)
            
            # 1. Luôn thêm vào buffer (để Partial & Final giống hệt nhau)
            rolling_buffer.extend(samples)
            
            # 2. Vẫn feed VAD để phát hiện sự im lặng (Trigger)
            client_vad.accept_waveform(samples)
            
            # [DEBUG]
            pass

            current_time = asyncio.get_event_loop().time()
            
            # A. CHECK FOR FINAL SEGMENTS (BUFFER-BASED STRATEGY)
            # VAD chỉ đóng vai trò là "Cò" (Trigger). Khi VAD báo hết câu (có segment):
            # -> Ta lấy TOÀN BỘ rolling_buffer ra decode. Đảm bảo Final == Partial.
            if not client_vad.empty():
                logging.info(f"⚡ VAD Triggered End of Sentence")
                
                # 1. Xả hết sự kiện VAD (nhưng không dùng audio trong đó)
                while not client_vad.empty():
                    client_vad.pop()
                
                # 2. Decode FULL Buffer
                if len(rolling_buffer) > 1600: 
                    stream = selected_recognizer.create_stream()
                    stream.accept_waveform(16000, np.array(rolling_buffer, dtype=np.float32))
                    selected_recognizer.decode_stream(stream)
                    result = stream.result
                    

                    # [FILTER] Remove common hallucinations (ừ, à, ờ repeated)
                    text = result.text.strip().lower()
                    
                    # Remove multiple ừ/à/ờ (e.g., "ừ ừ ừ" -> "")
                    # Regex: \b(ừ|à|ờ)(\s+\1)+\b matches repeated sequences like "ừ ừ"
                    text = re.sub(r'\b(ừ|à|ờ|um|uh)(\s+\1)+\b', '', text)
                    # Remove isolated filler words if they are the ONLY content (noise hallucination)
                    if re.fullmatch(r'^(ừ|à|ờ|um|uh)+$', text): 
                        text = ""
                    
                    # [TIMESTAMP FIX] Tính thời gian bắt đầu dựa trên tổng sample trôi qua
                    # buffer_start_time = Hiện tại - Độ dài buffer
                    buffer_start_time = (total_samples_processed - len(rolling_buffer)) / 16000.0
                    
                    if text:
                        # [SMART PARAGRAPHING]
                        time_gap = current_time - last_segment_end_time
                        if last_segment_end_time > 0 and time_gap > 2.0:
                            current_speaker = 1 - current_speaker 
                            logging.info(f"¶ New Paragraph (Gap: {time_gap:.2f}s)")
                            
                        last_segment_end_time = current_time 

                        # [FEATURE] Word-level Timestamps
                        words = []
                        if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                            for i, token in enumerate(result.tokens):
                                 local_start = result.timestamps[i]
                                 absolute_start = buffer_start_time + local_start 
                                 
                                 start = round(absolute_start, 2)
                                 end = round(start + 0.1, 2)
                                 
                                 if i < len(result.timestamps) - 1:
                                     next_local = result.timestamps[i+1]
                                     next_absolute = buffer_start_time + next_local
                                     end = round(next_absolute, 2)
                                 
                                 clean_word = token.replace('▁', '').strip().lower()
                                 words.append({
                                     "word": clean_word,
                                     "start": start,
                                     "end": end,
                                     "confidence": 1.0,
                                     "speaker": current_speaker
                                 })
                        
                        if not words:
                            words.append({
                                "word": text,
                                "start": round(buffer_start_time, 2),
                                "end": round(buffer_start_time + (len(rolling_buffer)/16000), 2),
                                "confidence": 1.0,
                                "speaker": current_speaker
                            })
                        
                        logging.info(f"✅ [{conn_id}] Final Result [Speaker {current_speaker}]: {text}")
                        response = {
                            "channel": {
                                "alternatives": [{
                                    "transcript": text,
                                    "confidence": 1.0,
                                    "speaker": current_speaker, 
                                    "words": words 
                                }]
                            },
                            "is_final": True
                        }
                        await websocket.send(json.dumps(response, ensure_ascii=False))
                    
                # 3. Reset Buffer (Giữ lại đuôi nhỏ 0.25s để tránh cắt quá gắt nếu người dùng nói nối)
                if len(rolling_buffer) > 4000:
                    rolling_buffer = rolling_buffer[-4000:]
                else:
                    rolling_buffer = [] 
                
                current_sentence_id += 1  
                
            # [CRITICAL FEATURE] C. FORCED SEGMENTATION (Prevent Freezing on Long Speech)
            # If user speaks continuously for > 8 seconds without silence, FORCE a cut.
            # Set to 8s as a balanced limit
            if len(rolling_buffer) > 128000: # 16000 * 8s
                 logging.info(f"⚠️ Forced Segmentation (Long Speech > 8s)")
                 stream = selected_recognizer.create_stream()
                 stream.accept_waveform(16000, np.array(rolling_buffer, dtype=np.float32))
                 selected_recognizer.decode_stream(stream)
                 # [FIX] Lowercase forced segment
                 text = stream.result.text.strip().lower()
                 

                 # [FILTER] Remove common hallucinations (Same as VAD block)
                 # Regex: \b(ừ|à|ờ)(\s+\1)+\b matches repeated sequences like "ừ ừ"
                 text = re.sub(r'\b(ừ|à|ờ|um|uh)(\s+\1)+\b', '', text)
                 # Remove isolated filler words if they are the ONLY content (noise hallucination)
                 if re.fullmatch(r'^(ừ|à|ờ|um|uh)+$', text): 
                    text = ""
                 
                 # [TIMESTAMP FIX] Calculate Buffer Start Time
                 buffer_start_time = (total_samples_processed - len(rolling_buffer)) / 16000.0
                 
                 if text:
                    logging.info(f"✅ Final Result (Forced - Speaker {current_speaker}): {text}")
                    
                    # [FEATURE] Word-level Timestamps for Forced Segment
                    words_forced = []
                    result = stream.result
                    
                    if hasattr(result, 'tokens') and hasattr(result, 'timestamps'):
                        for i, token in enumerate(result.tokens):
                             local_start = result.timestamps[i]
                             absolute_start = buffer_start_time + local_start
                             
                             start = absolute_start
                             end = start + 0.1
                             
                             if i < len(result.timestamps) - 1:
                                 next_local = result.timestamps[i+1]
                                 next_absolute = buffer_start_time + next_local
                                 end = next_absolute
                             
                             clean_word = token.replace('▁', '').strip().lower()
                             words_forced.append({
                                 "word": clean_word,
                                 "start": round(start, 2),
                                 "end": round(end, 2),
                                 "confidence": 1.0,
                                 "speaker": current_speaker
                             })

                    # Fallback if no tokens
                    if not words_forced:
                        words_forced = [{
                            "word": text,
                            "start": round(buffer_start_time, 2),
                            "end": round(buffer_start_time + 8.0, 2),
                            "confidence": 1.0,
                            "speaker": current_speaker
                        }]
                    
                    await websocket.send(json.dumps({
                        "channel": {"alternatives": [{
                            "transcript": text, 
                            "confidence": 1.0,
                            "speaker": current_speaker, # Added top-level consistency
                            "words": words_forced
                        }]},
                        "is_final": True
                    }, ensure_ascii=False))
                    
                    # current_speaker = 1 - current_speaker # [Logic Update] Don't toggle on forced cut. Same speaker is talking.
                 
                 # [CRITICAL FIX] Always clear buffer after Forced Segmentation, even if text is empty!
                 rolling_buffer = []

                 current_sentence_id += 1 # New sequence 
                 client_vad.reset()
                 
                 # [TIMESTAMP FIX] Update VAD Offset because reset() zeroes internal counter
                 vad_start_offset_samples = total_samples_processed
            
            # B. PARTIAL DECODE (Visual Feedback)
            # [OPTIMIZATION] Dynamic Interval to prevent locking on long sentences
            # Base = DECODE_INTERVAL (e.g. 0.2s or 0.4s)
            # Add 0.05s for every second of audio. 10s audio -> +0.5s interval.
            buffer_duration = len(rolling_buffer) / 16000.0
            dynamic_interval = max(DECODE_INTERVAL, DECODE_INTERVAL + buffer_duration * 0.05)
            
            if len(rolling_buffer) > 4000 and (current_time - last_decode_time > dynamic_interval):
                if not is_partial_decoding:
                    # [SAFE ASYNC] Fire-and-forget task
                    # Copy buffer to ensure thread safety
                    buffer_copy = np.array(rolling_buffer, dtype=np.float32)
                    is_partial_decoding = True
                    # Pass sequence ID to ensure we don't send stale partials after final
                    asyncio.create_task(run_partial_decode(buffer_copy, current_sentence_id))
                else:
                    # Previous decode still running, skip this frame to prevent stacking
                    pass

    except websockets.exceptions.ConnectionClosed:
        logging.info("🔌 Client đã ngắt kết nối")
    except Exception as e:
        logging.error(f"❌ Lỗi connection: {e}")

# Helper for threaded decoding
def decode_buffer_sync(recognizer, buffer_array, language="vi"):
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, buffer_array)
    recognizer.decode_stream(stream)
    result = stream.result
    
    if language == "en":
        # BPE token merging for English
        raw_tokens = result.tokens
        raw_times = result.timestamps
        
        word_buffer = ""
        for j, token in enumerate(raw_tokens):
            # U+2581 check
            has_boundary = token.startswith('\u2581') or token.startswith(' ')
            content = token.replace('\u2581', '').replace(' ', '').strip()
            if not content: continue
            
            if has_boundary:
                if word_buffer: word_buffer += " "
                word_buffer += content
            else:
                word_buffer += content
        return word_buffer.strip().lower()
        
    return result.text.strip().lower()

async def main():
    server = await websockets.serve(handle_connection, "0.0.0.0", PORT, ping_interval=None)
    logging.info(f"🚀 Server UPDATED VERSION (Async Partial Decode) đang lắng nghe tại ws://0.0.0.0:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())