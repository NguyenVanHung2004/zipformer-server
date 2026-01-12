import asyncio
import websockets
import sounddevice as sd
import numpy as np
import json
import sys

async def run_client():
    uri = "ws://localhost:6006"
    print(f"🔌 Đang kết nối tới {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Kết nối thành công! Hãy nói vào microphone...")
            
            loop = asyncio.get_running_loop()
            input_queue = asyncio.Queue()

            def callback(indata, frames, time, status):
                if status:
                    print(status, file=sys.stderr)
                # Copy dữ liệu vào queue để xử lý trong loop async
                loop.call_soon_threadsafe(input_queue.put_nowait, indata.copy())

            # Start recording
            stream = sd.InputStream(channels=1, dtype="int16", samplerate=16000, callback=callback)
            stream.start()

            async def send_audio():
                while True:
                    data = await input_queue.get()
                    await ws.send(data.tobytes())

            async def receive_result():
                while True:
                    try:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # Parsing Deepgram format
                        if 'channel' in data:
                            alt = data['channel']['alternatives'][0]
                            text = alt['transcript']
                            words = alt.get('words', [])
                            
                            print(f"\n📝 Text: {text}")
                            if words:
                                print("   ⏱️ Words:")
                                for w in words:
                                    print(f"      - {w['word']} ({w['start']:.2f}s -> {w['end']:.2f}s)")
                        else:
                            # Fallback or other messages
                            print(f"\n📩 Message: {msg}")
                    except websockets.exceptions.ConnectionClosed:
                        print("\n❌ Server đã đóng kết nối.")
                        break

            # Chạy song song gửi và nhận
            await asyncio.gather(send_audio(), receive_result())
            
            stream.stop()
            stream.close()

    except Exception as e:
        print(f"❌ Không thể kết nối hoặc lỗi: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\n🛑 Dừng chương trình.")