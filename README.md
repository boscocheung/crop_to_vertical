# Crop to Vertical — chuyển 16:9 sang 9:16, tự bám theo chủ thể

Công cụ chạy trên máy tính (Windows/macOS/Linux) để chuyển hàng loạt video từ 16:9 sang 9:16, khung hình tự động bám theo khuôn mặt/chủ thể thay vì cắt cứng ở giữa.

## 1. Cài đặt

### a) Cài Python 3.8+
```bash
# macOS
brew install python3

# Ubuntu/Debian
sudo apt install python3 python3-pip

# Windows: tải từ https://www.python.org/downloads/
```

### b) Cài ffmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows: https://www.gyan.dev/ffmpeg/builds/
```

### c) Cài thư viện Python
```bash
pip install -r crop_to_vertical_requirements.txt
```

## 2. Cách sử dụng

### Chuyển đổi 1 file
```bash
python crop_to_vertical.py -i "video.mp4" -o "video_9x16.mp4"
```

### Chuyển đổi cả thư mục
```bash
python crop_to_vertical.py -i "./tap_phim" -o "./output" --batch
```

### Tùy chọn hay dùng

| Tùy chọn | Ý nghĩa | Mặc định |
|---|---|---|
| `--width` `--height` | Kích thước video output | 1080 x 1920 |
| `--smooth` | Độ mượt chuyển động của khung crop (giây) | 1.0 |
| `--detector face` | Phát hiện khuôn mặt (hợp cảnh nói chuyện) | face |
| `--detector hog` | Phát hiện dáng người toàn thân (hợp cảnh hành động) | - |
| `--sample-every` | Detect mỗi N khung hình (tăng tốc) | 3 |

### Ví dụ
```bash
# Xử lý hàng loạt với detector khuôn mặt (mặc định, nhanh)
python crop_to_vertical.py -i ./tap_phim -o ./output --batch

# Cảnh hành động: dùng HOG, độ mượt thấp
python crop_to_vertical.py -i video.mp4 -o video_9x16.mp4 --detector hog --smooth 0.5

# Độ phân giải khác
python crop_to_vertical.py -i video.mp4 -o video_9x16.mp4 --width 1080 --height 1920
```

## 3. Cách công cụ hoạt động

1. **Phát hiện cắt cảnh**: So sánh màu sắc HSV giữa các khung để phát hiện chuyển cảnh
2. **Phát hiện chủ thể**: Ở đầu mỗi cảnh, tìm khuôn mặt/người lớn nhất
3. **Tracker**: Bám theo chủ thể trong suốt cảnh bằng KCF (không detect mỗi khung → nhanh hơn)
4. **Làm mượt**: Làm mượt đường di chuyển tâm crop TRONG TỪNG CẢNH (không xuyên qua cắt cảnh)
5. **Render**: Cắt video theo cửa sổ dọc + ghép lại âm thanh gốc

## 4. Lưu ý

- Mặc định dùng `--detector face` (nhanh, hợp cảnh nói chuyện). Với cảnh hành động, thử `--detector hog`
- Nếu không detect được ai, khung crop giữ nguyên vị trí gần nhất
- Xử lý theo lô (`--batch`) rất hợp để chạy qua đêm cho cả mùa phim
- Để tạo file `.exe` chạy mà không cần Python: `pip install pyinstaller` rồi `pyinstaller --onefile crop_to_vertical.py`
