# Crop to Vertical — chuyển 16:9 sang 9:16, tự bám theo chủ thể

Công cụ chạy trên máy tính (Windows/macOS/Linux) để chuyển hàng loạt tập phim
từ 16:9 sang 9:16, khung hình tự động bám theo khuôn mặt/chủ thể thay vì cắt
cứng ở giữa.

## 1. Cài đặt (làm 1 lần)

### a) Cài Python
Tải và cài Python 3.9+ từ https://www.python.org/downloads/
(Khi cài trên Windows, nhớ tick chọn "Add Python to PATH").

### b) Cài ffmpeg
- Windows: tải bản build tại https://www.gyan.dev/ffmpeg/builds/, giải nén,
  thêm thư mục `bin` vào PATH.
- macOS: `brew install ffmpeg`
- Linux (Ubuntu/Debian): `sudo apt install ffmpeg`

Kiểm tra đã cài đúng chưa bằng cách mở Terminal/Command Prompt gõ:
```
ffmpeg -version
```

### c) Cài thư viện Python
Mở Terminal/Command Prompt tại thư mục chứa `crop_to_vertical.py`, chạy:
```
pip install -r requirements.txt
```

**Quan trọng:** nếu trước đó bạn đã lỡ cài `opencv-python`, hãy gỡ nó trước
để tránh xung đột (2 gói cùng cung cấp module `cv2`):
```
pip uninstall opencv-python opencv-python-headless -y
pip install -r requirements.txt
```

### d) Model nhận diện khuôn mặt
Công cụ dùng **YuNet** (model nhận diện khuôn mặt chính thức của OpenCV) làm
chính. Ở lần chạy đầu tiên, script sẽ **tự động tải** file model này về cùng
thư mục (cần có mạng, tải từ GitHub - khoảng 350KB). Từ lần sau sẽ dùng lại
file đã tải, không cần tải lại.

Nếu mạng của bạn chặn GitHub (một số mạng công ty/trường học), script sẽ báo
lỗi kèm đường link - bạn tải thủ công file `face_detection_yunet_2023mar.onnx`
từ đường link đó và đặt cùng thư mục với `crop_to_vertical.py`.

Ngoài ra vẫn giữ 2 file `deploy.prototxt` và
`res10_300x300_ssd_iter_140000_fp16.caffemodel` đi kèm làm phương án dự
phòng, chỉ dùng khi máy bạn cài OpenCV bản cũ (trước 5.0) và không tải được
YuNet.

## 2. Cách dùng

### Xử lý 1 tập phim
```
python crop_to_vertical.py -i "tap01.mp4" -o "tap01_9x16.mp4"
```

### Xử lý cả thư mục nhiều tập (khuyên dùng cho phim bộ)
```
python crop_to_vertical.py -i "./tap_phim" -o "./output" --batch
```
Công cụ sẽ tự quét tất cả file video trong thư mục `tap_phim` và xuất kết quả
vào thư mục `output`, giữ nguyên tên file + hậu tố `_9x16`.

### Chèn overlay (khung/tiêu đề/logo) lên video
Nếu có 1 ảnh PNG với vùng trong suốt (ví dụ khung tiêu đề, logo, watermark)
muốn phủ lên trên video SAU KHI đã crop 9:16 như bình thường:
```
python crop_to_vertical.py -i "tap01.mp4" -o "tap01_out.mp4" --background "tieu_de.png"
```
Toàn bộ pipeline crop/bám chủ thể (kể cả `--reference` nếu có) vẫn chạy y
như bình thường - `--background` chỉ thêm 1 bước phủ ảnh PNG lên trên cùng
mỗi khung hình sau khi crop xong. Vùng trong suốt của ảnh PNG sẽ để lộ video
bên dưới; vùng không trong suốt sẽ che phủ (dùng để chèn khung/tiêu đề/logo).
Ảnh nên có kích thước đúng bằng `--width x --height` (mặc định 1080x1920)
để không bị méo khi tự resize.

### Dùng ảnh mẫu để bám đúng người (khuyên dùng cho gameshow/concert)
Nếu sân khấu có màn hình LED chiếu cận mặt, hoặc nhiều người trong khung,
cung cấp 1 ảnh mẫu (ảnh chân dung rõ mặt) cho mỗi người xuất hiện — công cụ
sẽ CHỈ bám theo người khớp ảnh mẫu, bỏ qua mọi khuôn mặt khác (kể cả trên
màn hình nền):
```
python crop_to_vertical.py -i "tap01.mp4" -o "tap01_9x16.mp4" --reference "ca_si=anh_ca_si.jpg"
```
Nhiều người trong 1 video (ví dụ 2 MC thay phiên xuất hiện):
```
python crop_to_vertical.py -i "tap01.mp4" -o "tap01_9x16.mp4" ^
  --reference "mc_a=anh_mc_a.jpg" --reference "mc_b=anh_mc_b.jpg"
```
(Trên macOS/Linux dùng `\` thay cho `^` ở cuối dòng khi viết trên nhiều dòng.)

Ảnh mẫu nên là ảnh rõ mặt, chụp thẳng hoặc gần thẳng, đủ sáng. Công cụ tự
cắt khuôn mặt từ ảnh bạn cung cấp, không cần crop sẵn.

Khi dùng `--reference`, nếu một đoạn không ai khớp ảnh mẫu (ví dụ cảnh toàn
sân khấu, chưa thấy mặt), khung crop sẽ giữ nguyên vị trí gần nhất đã biết
thay vì đoán đại một khuôn mặt bất kỳ.

### Các tùy chọn hay dùng
| Tùy chọn | Ý nghĩa | Mặc định |
|---|---|---|
| `--width` `--height` | Kích thước video output | 1080 x 1920 |
| `--smooth` | Độ mượt chuyển động của khung crop (giây). Số lớn hơn = mượt hơn nhưng bám chậm hơn khi chủ thể di chuyển nhanh | 1.0 |
| `--detector` | `face` (phát hiện khuôn mặt — hợp cảnh nói chuyện/cận mặt, bắt buộc nếu dùng `--reference`) hoặc `hog` (phát hiện dáng người toàn thân — hợp cảnh hành động/toàn thân) | face |
| `--reference` | Ảnh mẫu `ten=duong_dan.jpg`, dùng nhiều lần cho nhiều người | (không có) |
| `--cut-threshold` | Độ nhạy phát hiện cắt cảnh, 0-1. Cao hơn = nhạy hơn | 0.75 |

Ví dụ tinh chỉnh cho cảnh hành động, chủ thể di chuyển nhanh (không dùng
được `--reference` vì detector đổi sang `hog`):
```
python crop_to_vertical.py -i "./tap_phim" -o "./output" --batch --detector hog --smooth 0.5
```

## 3. Cách công cụ hoạt động
1. Phát hiện điểm cắt cảnh (chuyển cảnh) bằng so sánh màu sắc giữa các khung.
2. Ở đầu mỗi cảnh: phát hiện khuôn mặt. Nếu có ảnh mẫu, chỉ chọn khuôn mặt
   khớp danh tính (so khớp bằng LBPH — thuật toán nhận diện khuôn mặt học
   trực tiếp từ ảnh mẫu bạn cung cấp, không cần tải model nặng). Nếu không
   có ảnh mẫu, chọn khuôn mặt lớn nhất trong khung.
3. Trong suốt cảnh đó: dùng tracker (KCF) bám theo đúng vùng ảnh đã chọn,
   không detect lại mỗi khung — tránh nhảy qua lại giữa 2 khuôn mặt giống
   nhau. Định kỳ xác thực lại danh tính (nếu có ảnh mẫu) để sửa nếu tracker
   trôi sang mục tiêu khác.
4. Làm mượt đường di chuyển của tâm khung crop trong từng cảnh (không làm
   mượt xuyên qua điểm cắt cảnh, để mỗi cảnh mới nhảy thẳng tới đúng vị trí).
5. Cắt video theo cửa sổ dọc bám theo tâm đã làm mượt, rồi ghép lại âm
   thanh gốc bằng ffmpeg.

## 4. Lưu ý
- **Nên dùng `--reference`** cho gameshow/concert có màn hình LED nền hoặc
  nhiều người trong khung — đây là cách đáng tin cậy nhất để tránh bám nhầm.
- Nếu một đoạn không phát hiện được ai (không khớp ảnh mẫu, hoặc không có
  khuôn mặt nào), khung crop giữ nguyên vị trí gần nhất đã biết thay vì
  nhảy về giữa.
- Xử lý theo lô (`--batch`) rất hợp để chạy qua đêm cho cả một mùa phim.
- Nếu muốn đóng gói thành file `.exe` chạy không cần cài Python, có thể dùng
  thêm `pyinstaller`: `pip install pyinstaller` rồi
  `pyinstaller --onefile crop_to_vertical.py` (khi đó cần đóng gói kèm các
  file model ở mục 1d cùng thư mục với file `.exe`).
