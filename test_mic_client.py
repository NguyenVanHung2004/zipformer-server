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
                print("\nListening...")
                current_line_length = 0
                
                while True:
                    try:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        if 'channel' in data:
                            alt = data['channel']['alternatives'][0]
                            text = alt['transcript']
                            is_final = data.get('is_final', False)
                            
                            # Clear current line visually
                            sys.stdout.write('\r' + ' ' * (current_line_length + 10) + '\r')
                            
                            if is_final:
                                # Start a new line for the final result
                                sys.stdout.write(f"✅ {text}\n")
                                current_line_length = 0
                            else:
                                # Overwrite line for partial result
                                display_text = f"⏳ {text}"
                                sys.stdout.write(display_text)
                                current_line_length = len(display_text)
                                
                            sys.stdout.flush()
                            
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