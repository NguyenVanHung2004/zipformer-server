# Zipformer WebSocket ASR Server

Dự án này là một máy chủ nhận dạng giọng nói (Speech-to-Text) thời gian thực, sử dụng giao thức WebSockets và thư viện `sherpa-onnx`. Hệ thống hỗ trợ xử lý luồng âm thanh trực tiếp, tự động tải mô hình và áp dụng các kỹ thuật xử lý tín hiệu âm thanh chuyên sâu để đem lại kết quả văn bản chính xác nhất.

## Tính năng nổi bật

* **Nhận diện giọng nói thời gian thực**: Cung cấp kết quả tạm thời (Partial) liên tục trong khi nói và kết quả chốt câu (Final) khi phát hiện khoảng lặng.
* **Hỗ trợ đa ngôn ngữ**: Hỗ trợ nhận diện tiếng Việt (mô hình fine-tuned INT8) và tiếng Anh (GigaSpeech). Người dùng có thể chỉ định ngôn ngữ thông qua tham số trên URL (`?language=vi` hoặc `?language=en`).
* **Tự động quản lý mô hình (Auto-Download)**: Tự động tải xuống và cập nhật các mô hình nhận diện ASR (Zipformer) cũng như mô hình Silero VAD từ GitHub Releases và HuggingFace.
* **Xử lý tín hiệu âm thanh (DSP)**: Tích hợp bộ lọc âm (Bandpass Filter 150Hz - 3400Hz) giúp loại bỏ tiếng ồn cơ học (tiếng quạt, tiếng ù). Kèm theo đó là cơ chế tự động điều chỉnh âm lượng (AGC) để chuẩn hóa tín hiệu đầu vào.
* **Phát hiện hoạt động giọng nói (VAD)**: Tự động phân đoạn câu dựa trên khoảng lặng thực tế sử dụng thư viện Silero VAD.
* **Ngắt câu bắt buộc (Forced Segmentation)**: Tự động cắt đoạn âm thanh nếu người dùng nói liên tục quá 8 giây (tương đương 128.000 mẫu âm thanh) để tránh tình trạng treo hoặc gián đoạn hệ thống.
* **Timestamps cấp độ từ**: Cung cấp thông tin thời gian bắt đầu và kết thúc (start/end) cho từng từ cụ thể trong kết quả cuối cùng. Đi kèm khả năng theo dõi sự thay đổi người nói (speaker toggle) khi có khoảng cách thời gian giữa các đoạn.

## Yêu cầu hệ thống

Các thư viện bắt buộc được liệt kê trong file `requirements.txt`:
* `sherpa-onnx==1.12.20`
* `websockets`
* `numpy`
* `scipy`

*Lưu ý:* Để chạy script kiểm thử với microphone (`test_mic_client.py`), bạn cần cài đặt thêm thư viện `sounddevice`.

## Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt thư viện
Bạn nên sử dụng một môi trường ảo (virtual environment) và cài đặt các dependencies:
```bash
pip install -r requirements.txt
pip install sounddevice  # Bắt buộc nếu muốn dùng client thu âm trực tiếp
2. Khởi chạy Server
Mặc định, server sẽ lắng nghe ở địa chỉ ws://0.0.0.0:6006. Trong lần đầu tiên khởi chạy, server sẽ tự động tải xuống các mô hình cần thiết vào thư mục model_vi, model_en và model_ft.

Bash
python server.py
