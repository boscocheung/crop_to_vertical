#!/usr/bin/env python3
"""
crop_to_vertical.py
--------------------
Chuyển video 16:9 -> 9:16, khung crop tự động bám theo chủ thể
(khuôn mặt/người) trong từng khung hình, thay vì cắt cố định ở giữa.

Cách hoạt động:
  1) Quét video, phát hiện khuôn mặt ở mỗi khung hình (lấy mẫu cách N frame
     để tăng tốc), ghi lại vị trí tâm khuôn mặt theo thời gian. Đồng thời
     phát hiện các điểm CẮT CẢNH (chuyển cảnh đột ngột, hay gặp trong trailer).
  2) Nội suy + làm mượt (smoothing) đường đi của tâm crop TRONG TỪNG CẢNH
     để camera "ảo" di chuyển nhẹ nhàng, không giật - nhưng KHÔNG làm mượt
     xuyên qua các điểm cắt cảnh, để mỗi cảnh mới nhảy thẳng tới đúng vị trí
     thay vì bị lia máy từ từ qua lại.
  3) Cắt video theo cửa sổ 9:16 bám theo tâm đã làm mượt, dùng OpenCV để
     ghi hình, rồi dùng ffmpeg để ghép lại âm thanh gốc + nén H.264.

Yêu cầu:
  - Python 3.8+
  - pip install opencv-python numpy
  - ffmpeg cài sẵn và có trong PATH (dùng để ghép audio + nén cuối cùng)

Cách dùng:
  Xử lý 1 file:
    python crop_to_vertical.py -i tap01.mp4 -o out/tap01_9x16.mp4

  Xử lý cả thư mục (nhiều tập phim cùng lúc):
    python crop_to_vertical.py -i ./tap_phim -o ./output --batch

  Tùy chọn thường dùng:
    --width 1080 --height 1920      kích thước output (mặc định 1080x1920)
    --smooth 1.0                    độ mượt chuyển động camera (giây), số
                                     càng lớn càng mượt nhưng bám chậm hơn
    --sample-every 3                cứ mỗi 3 khung hình mới detect 1 lần
                                     (tăng tốc, giảm độ chính xác theo thời gian thực)
    --detector face|hog             'face' = phát hiện khuôn mặt (nhanh, hợp cảnh
                                     nói chuyện/phỏng vấn/phim có mặt rõ);
                                     'hog' = phát hiện dáng người toàn thân
                                     (chậm hơn, hợp cảnh hành động/toàn thân)
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def log(msg):
    print(f"[crop_to_vertical] {msg}", flush=True)


YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]


class YuNetDetector:
    """Bọc cv2.FaceDetectorYN để chuẩn hóa cách gọi giống các detector khác
    trong script (input_size phải khớp kích thước ảnh thật ở mỗi lần detect)."""

    def __init__(self, model_path):
        self._detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320), score_threshold=0.5, nms_threshold=0.3, top_k=50
        )
        self._current_size = (320, 320)

    def detect(self, frame):
        h, w = frame.shape[:2]
        if (w, h) != self._current_size:
            self._detector.setInputSize((w, h))
            self._current_size = (w, h)
        ok, faces = self._detector.detect(frame)
        return faces if faces is not None else []


def _download_yunet_model(dest_path):
    import urllib.request
    for url in YUNET_MODEL_URLS:
        try:
            log(f"  Đang tải model nhận diện khuôn mặt (YuNet) từ {url} ...")
            urllib.request.urlretrieve(url, dest_path)
            if os.path.getsize(dest_path) > 100_000:  # file thật phải vài trăm KB
                log("  Tải xong.")
                return True
            os.remove(dest_path)
        except Exception as e:
            log(f"  Không tải được từ {url}: {e}")
    return False


def get_face_detector():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ưu tiên YuNet - API riêng của OpenCV cho nhận diện khuôn mặt, hoạt động
    # trên cả OpenCV 4.x và 5.x (không phụ thuộc readNetFromCaffe, vốn đã bị
    # OpenCV 5.0 loại bỏ), đồng thời nhận diện khuôn mặt nhỏ/mờ tốt hơn.
    if hasattr(cv2, "FaceDetectorYN_create"):
        yunet_path = os.path.join(script_dir, YUNET_MODEL_FILENAME)
        if not os.path.isfile(yunet_path):
            ok = _download_yunet_model(yunet_path)
            if not ok:
                log(f"  Cảnh báo: không tự tải được model YuNet. Bạn có thể tải thủ công "
                    f"file '{YUNET_MODEL_FILENAME}' từ "
                    f"{YUNET_MODEL_URLS[0]} và đặt cùng thư mục với script.")
                yunet_path = None
        if yunet_path and os.path.isfile(yunet_path):
            try:
                return YuNetDetector(yunet_path)
            except Exception as e:
                log(f"  Cảnh báo: khởi tạo YuNet thất bại ({e}), sẽ thử cách khác.")

    # Dự phòng: model Caffe cũ, chỉ hoạt động với OpenCV < 5.0
    if hasattr(cv2.dnn, "readNetFromCaffe"):
        prototxt = os.path.join(script_dir, "deploy.prototxt")
        model = os.path.join(script_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
        if os.path.isfile(prototxt) and os.path.isfile(model):
            return cv2.dnn.readNetFromCaffe(prototxt, model)

    raise RuntimeError(
        "Không thể khởi tạo bộ nhận diện khuôn mặt. Không tải được model YuNet "
        "(có thể do mạng chặn GitHub) và cũng không có model Caffe dự phòng "
        "khả dụng trên phiên bản OpenCV này. Hãy thử tải thủ công file "
        f"'{YUNET_MODEL_FILENAME}' từ {YUNET_MODEL_URLS[0]} và đặt cùng thư "
        "mục với crop_to_vertical.py."
    )


SFACE_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"
SFACE_MODEL_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]
# Ngưỡng độ tương đồng cosine do OpenCV Zoo khuyến nghị cho SFace: >= ngưỡng
# này coi là cùng 1 người. Đáng tin cậy hơn nhiều so với LBPH (vốn không đủ
# khả năng phân biệt người khác nhau trong video nén/ánh sáng thay đổi).
SFACE_MATCH_THRESHOLD = 0.363


def get_face_recognizer():
    """Tải (nếu cần) và khởi tạo SFace - mô hình nhận diện danh tính khuôn
    mặt (deep embedding), chính xác hơn LBPH rất nhiều. Trả về None nếu
    không tải/khởi tạo được (khi đó code gọi sẽ tự rơi về dùng LBPH)."""
    if not hasattr(cv2, "FaceRecognizerSF_create"):
        return None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, SFACE_MODEL_FILENAME)
    if not os.path.isfile(model_path):
        ok = False
        for url in SFACE_MODEL_URLS:
            try:
                import urllib.request
                log(f"  Đang tải model nhận diện danh tính (SFace) từ {url} ...")
                urllib.request.urlretrieve(url, model_path)
                if os.path.getsize(model_path) > 100_000:
                    log("  Tải xong.")
                    ok = True
                    break
                os.remove(model_path)
            except Exception as e:
                log(f"  Không tải được từ {url}: {e}")
        if not ok:
            log(f"  Cảnh báo: không tự tải được model SFace. Sẽ dùng phương pháp "
                f"nhận diện danh tính dự phòng (kém chính xác hơn). Bạn có thể tải "
                f"thủ công file '{SFACE_MODEL_FILENAME}' từ {SFACE_MODEL_URLS[0]} "
                f"và đặt cùng thư mục với script.")
            return None
    try:
        return cv2.FaceRecognizerSF_create(model_path, "")
    except Exception as e:
        log(f"  Cảnh báo: khởi tạo SFace thất bại ({e}), sẽ dùng phương pháp dự phòng.")
        return None


def _get_face_embedding(face_recognizer, frame, box, landmarks):
    """Trích embedding danh tính (SFace) cho 1 khuôn mặt. Nếu có landmarks
    (5 điểm mắt/mũi/miệng từ YuNet) sẽ căn chỉnh khuôn mặt trước khi trích
    đặc trưng - chính xác hơn nhiều so với chỉ crop thô."""
    x, y, w, h = box
    if landmarks is not None:
        face_info = np.array([x, y, w, h] + list(landmarks.flatten()), dtype=np.float32)
        aligned = face_recognizer.alignCrop(frame, face_info)
    else:
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            return None
        aligned = cv2.resize(crop, (112, 112))
    return face_recognizer.feature(aligned)


def _sface_similarity(face_recognizer, emb1, emb2):
    return face_recognizer.match(emb1, emb2, cv2.FaceRecognizerSF_FR_COSINE)


def get_hog_detector():
    """Trả về HOG people-detector, hoặc None nếu OpenCV bản này không hỗ trợ
    (một số bản OpenCV 5.x dường như đã bỏ HOGDescriptor). Khi None, các
    tính năng phụ thuộc HOG (--detector hog, fallback nhận trang phục ở cảnh
    toàn) sẽ tự động bị tắt thay vì làm crash toàn bộ script."""
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    except Exception:
        return None


FACE_CONFIDENCE_THRESHOLD = 0.5
IDENTITY_FACE_SIZE = 200  # kích thước chuẩn hóa khuôn mặt để so khớp danh tính


def _preprocess_face_for_identity(frame, box):
    """Cắt + chuẩn hóa 1 vùng khuôn mặt (grayscale, resize, cân bằng sáng)
    để so khớp danh tính bằng LBPH - ổn định hơn khi ảnh mẫu và video có
    ánh sáng/độ phân giải khác nhau."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    face = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IDENTITY_FACE_SIZE, IDENTITY_FACE_SIZE))
    gray = cv2.equalizeHist(gray)
    return gray


def load_reference_identities(reference_paths, face_detector, face_recognizer=None):
    """reference_paths: dict {ten_nguoi: duong_dan_anh}.
    Nếu face_recognizer (SFace) khả dụng, trả về ('sface', {ten: embedding}).
    Nếu không, dùng LBPH dự phòng, trả về ('lbph', (recognizer, names)).
    Trả về (None, None) nếu không học được ai."""
    if not reference_paths:
        return None, None

    face_imgs_by_name = {}
    for name, path in reference_paths.items():
        img = cv2.imread(path)
        if img is None:
            log(f"  Cảnh báo: không đọc được ảnh mẫu '{path}' cho '{name}', bỏ qua.")
            continue
        candidates = detect_candidates(img, None, face_detector, "face")
        if not candidates:
            log(f"  Cảnh báo: không tìm thấy khuôn mặt nào trong ảnh mẫu '{path}' "
                f"cho '{name}', bỏ qua.")
            continue
        # Nếu ảnh mẫu có nhiều mặt, lấy mặt lớn nhất (thường là mặt chính giữa ảnh)
        best = max(candidates, key=lambda c: c[1])
        face_imgs_by_name[name] = (img, best)

    if not face_imgs_by_name:
        return None, None

    if face_recognizer is not None:
        embeddings = {}
        for name, (img, cand) in face_imgs_by_name.items():
            emb = _get_face_embedding(face_recognizer, img, cand[2], cand[3])
            if emb is not None:
                embeddings[name] = emb
        if embeddings:
            log(f"  Đã học {len(embeddings)} người từ ảnh mẫu (SFace): "
                f"{', '.join(embeddings.keys())}")
            return "sface", embeddings
        log("  Cảnh báo: SFace không trích được đặc trưng từ ảnh mẫu, chuyển sang LBPH.")

    # Dự phòng: LBPH (kém chính xác hơn nhiều, chỉ dùng khi không có SFace)
    images, labels, names = [], [], []
    for name, (img, cand) in face_imgs_by_name.items():
        bx, by, bw, bh = cand[2]
        face_img = _preprocess_face_for_identity(img, (bx, by, bx + bw, by + bh))
        if face_img is None:
            continue
        labels.append(len(names))
        names.append(name)
        images.append(face_img)

    if not images:
        return None, None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    log(f"  Đã học {len(names)} người từ ảnh mẫu (LBPH - dự phòng, kém chính xác "
        f"hơn SFace): {', '.join(names)}")
    return "lbph", (recognizer, names)


def detect_candidates(frame, gray, detector, kind):
    """Trả về list các (center_x, area, box, landmarks) của mọi chủ thể phát
    hiện được, trong đó box=(x,y,w,h) dùng để khởi tạo tracker bám theo, và
    landmarks (5 điểm mắt/mũi/miệng, hoặc None nếu detector không hỗ trợ)
    dùng để căn chỉnh khuôn mặt cho SFace nhận diện danh tính chính xác hơn."""
    h, w = frame.shape[:2]
    if kind == "face":
        if isinstance(detector, YuNetDetector):
            faces = detector.detect(frame)
            candidates = []
            for f in faces:
                x, y, fw, fh = f[0], f[1], f[2], f[3]
                x = max(0.0, x); y = max(0.0, y)
                fw = max(min(fw, w - x), 1.0)
                fh = max(min(fh, h - y), 1.0)
                cx = x + fw / 2.0
                landmarks = f[4:14].reshape(5, 2) if len(f) >= 14 else None
                candidates.append((cx, fw * fh, (int(x), int(y), int(fw), int(fh)), landmarks))
            return candidates
        else:
            # Model Caffe dự phòng (OpenCV < 5.0) - không có landmarks
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            detector.setInput(blob)
            detections = detector.forward()
            candidates = []
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < FACE_CONFIDENCE_THRESHOLD:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                fw, fh = max(x2 - x1, 1), max(y2 - y1, 1)
                cx = (x1 + x2) / 2.0
                candidates.append((cx, fw * fh, (int(x1), int(y1), int(fw), int(fh)), None))
            return candidates
    else:  # hog - phát hiện người toàn thân
        rects, weights = detector.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        return [(rx + rw / 2.0, rw * rh, (int(rx), int(ry), int(rw), int(rh)), None)
                for (rx, ry, rw, rh) in rects]


IDENTITY_MAX_DISTANCE = 85.0  # ngưỡng LBPH (dự phòng): càng thấp càng giống
IDENTITY_RECHECK_SECONDS = 1.0  # cứ sau khoảng thời gian này xác thực lại danh tính đang bám, phòng bị trôi dần


def _valid_identity_matches(frame, candidates, identity_method, identity_data, face_recognizer):
    """Trả về list (candidate, ten) của những ứng viên KHỚP danh tính với ảnh
    mẫu, dùng chung cho cả SFace (embedding, chính xác cao) và LBPH (dự
    phòng, kém chính xác hơn)."""
    valid = []
    if identity_method == "sface":
        embeddings = identity_data  # dict {ten: embedding}
        for cand in candidates:
            emb = _get_face_embedding(face_recognizer, frame, cand[2], cand[3])
            if emb is None:
                continue
            best_name, best_score = None, -1.0
            for name, ref_emb in embeddings.items():
                score = _sface_similarity(face_recognizer, emb, ref_emb)
                if score > best_score:
                    best_name, best_score = name, score
            if best_score >= SFACE_MATCH_THRESHOLD:
                valid.append((cand, best_name))
    elif identity_method == "lbph":
        recognizer, names = identity_data
        for cand in candidates:
            face_img = _preprocess_face_for_identity(frame, (
                cand[2][0], cand[2][1], cand[2][0] + cand[2][2], cand[2][1] + cand[2][3]))
            if face_img is None:
                continue
            label, dist = recognizer.predict(face_img)
            if dist <= IDENTITY_MAX_DISTANCE:
                valid.append((cand, names[label]))
    return valid


def _best_identity_match(frame, candidates, identity_method, identity_data, face_recognizer):
    """Trong các ứng viên phát hiện được, lọc ra những ai KHỚP danh tính với
    ảnh mẫu (loại bỏ khuôn mặt lạ - ví dụ trên màn hình LED/phông nền).
    Nếu có nhiều người cùng khớp ảnh mẫu trong khung (cảnh toàn nhiều người),
    ưu tiên người có khuôn mặt LỚN NHẤT - thường là người đang nói/gần camera
    nhất - làm người chính, bỏ qua những người còn lại. Trả về
    (candidate, ten) hoặc None nếu không ai khớp ảnh mẫu."""
    valid = _valid_identity_matches(frame, candidates, identity_method, identity_data, face_recognizer)
    if not valid:
        return None
    return max(valid, key=lambda m: m[0][1])  # m[0][1] = dien tich khuon mat


def _nearest_identity_match(frame, candidates, identity_method, identity_data, face_recognizer, ref_center):
    """Giống _best_identity_match nhưng dùng để XÁC THỰC tracker đang bám
    (không phải chọn người chính lúc đầu cảnh) - nên ưu tiên ứng viên khớp
    ảnh mẫu GẦN vị trí đang bám nhất, tránh việc tự động đổi sang người khác
    chỉ vì họ tạm thời có khuôn mặt to hơn trong 1-2 khung hình."""
    valid = _valid_identity_matches(frame, candidates, identity_method, identity_data, face_recognizer)
    if not valid:
        return None
    return min(valid, key=lambda m: abs(m[0][0] - ref_center))


APPEARANCE_MATCH_THRESHOLD = 0.55  # ngưỡng tương đồng màu trang phục (0-1, càng cao càng giống)


def _appearance_signature_from_face_box(frame, face_box):
    """Trích 'chữ ký màu trang phục' (histogram màu HSV) từ vùng NGAY DƯỚI
    khuôn mặt đã xác thực - dùng để nhận ra lại người này ở cảnh toàn, khi
    mặt quá nhỏ để detect nhưng trang phục vẫn còn phân biệt được."""
    x, y, fw, fh = face_box
    h, w = frame.shape[:2]
    body_y1 = min(h, y + fh)
    body_y2 = min(h, y + fh + int(fh * 2.5))
    body_x1 = max(0, x - int(fw * 0.3))
    body_x2 = min(w, x + fw + int(fw * 0.3))
    if body_y2 <= body_y1 or body_x2 <= body_x1:
        return None
    region = frame[body_y1:body_y2, body_x1:body_x2]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _appearance_signature_from_body_box(frame, body_box):
    """Trích histogram màu từ khoảng giữa của 1 box dáng người (HOG) - tránh
    lấy phần đầu/tóc và chân/giày, tập trung vào phần thân/trang phục."""
    x, y, bw, bh = body_box
    h, w = frame.shape[:2]
    y1 = max(0, y + int(bh * 0.25))
    y2 = min(h, y + int(bh * 0.75))
    x1, x2 = max(0, x), min(w, x + bw)
    if y2 <= y1 or x2 <= x1:
        return None
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


HOG_DETECT_WIDTH = 480  # thu nhỏ khung hình về độ rộng này trước khi chạy HOG - tăng tốc ~15-17 lần


def _best_body_match_by_appearance(frame, hog_detector, known_signatures):
    """Ở cảnh toàn (mặt quá nhỏ để detect), dùng HOG để tìm dáng người, rồi
    so màu trang phục với chữ ký đã lưu của người trong ảnh mẫu. Trả về
    (center_x, box, ten) của kết quả khớp tốt nhất, hoặc None.

    HOG rất chậm ở độ phân giải gốc (~0.7 khung/giây với ảnh 1920x1080), nên
    luôn thu nhỏ khung hình trước khi detect (tăng tốc ~15-17 lần), rồi quy
    đổi toạ độ box tìm được về lại kích thước gốc."""
    if not known_signatures:
        return None
    h, w = frame.shape[:2]
    scale = HOG_DETECT_WIDTH / w if w > HOG_DETECT_WIDTH else 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame

    rects, weights = hog_detector.detectMultiScale(
        small, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    best = None
    for (rx, ry, rw, rh) in rects:
        # quy đổi box về toạ độ khung hình gốc
        box = (int(rx / scale), int(ry / scale), int(rw / scale), int(rh / scale))
        sig = _appearance_signature_from_body_box(frame, box)
        if sig is None:
            continue
        for name, known_sig in known_signatures.items():
            score = cv2.compareHist(sig, known_sig, cv2.HISTCMP_CORREL)
            if score < APPEARANCE_MATCH_THRESHOLD:
                continue
            if best is None or score > best[3]:
                cx = box[0] + box[2] / 2.0
                best = (cx, box, name, score)
    if best is None:
        return None
    return best[0], best[1], best[2]


def analyze_video(path, sample_every, detector_kind, cut_threshold=0.75,
                   identity_method=None, identity_data=None, face_recognizer=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = get_face_detector() if detector_kind == "face" else get_hog_detector()
    if detector_kind == "hog" and detector is None:
        raise RuntimeError(
            "Phiên bản OpenCV này không hỗ trợ HOGDescriptor (dùng cho --detector hog). "
            "Hãy dùng --detector face thay thế."
        )
    use_identity = identity_method is not None and detector_kind == "face"
    # HOG dùng làm phương án dự phòng khi cảnh toàn khiến mặt quá nhỏ để
    # detect - nhận lại người qua màu trang phục đã ghi nhận lúc cảnh cận.
    # Nếu OpenCV bản này không hỗ trợ HOG, bỏ qua tính năng dự phòng này
    # (không ảnh hưởng đến phần còn lại - chỉ là 1 lớp bảo hiểm thêm).
    body_detector = get_hog_detector() if use_identity else None
    known_signatures = {}

    centers = np.full(total_frames, width / 2.0, dtype=np.float64)
    has_detection = np.zeros(total_frames, dtype=bool)
    is_cut = np.zeros(total_frames, dtype=bool)

    log(f"Đang phân tích {os.path.basename(path)}: {total_frames} khung, {width}x{height}, {fps:.2f} fps")
    if use_identity:
        if body_detector is not None:
            log("  (dùng ảnh mẫu để xác thực đúng người + tracker bám theo đặc trưng hình ảnh; "
                "khi cảnh toàn không thấy mặt, thử nhận lại qua màu trang phục)")
        else:
            log("  (dùng ảnh mẫu để xác thực đúng người + tracker bám theo đặc trưng hình ảnh; "
                "LƯU Ý: OpenCV bản này không hỗ trợ HOG nên tắt fallback nhận trang phục ở cảnh toàn)")
    else:
        log("  (dùng detect để khóa mục tiêu + tracker bám theo đặc trưng hình ảnh, "
            "tránh nhầm giữa 2 khuôn mặt giống nhau)")

    idx = 0
    last_center = width / 2.0
    prev_hist = None
    tracker = None
    tracker_fail_streak = 0
    current_identity = None
    frames_since_recheck = 0
    recheck_interval = max(1, int(fps * IDENTITY_RECHECK_SECONDS))
    # Cho phép tracker mất dấu tạm thời vài khung (che khuất, quay lưng...)
    # trước khi coi là thật sự mất và phải detect lại.
    max_tracker_fail = max(3, int(fps * 0.3))
    # HOG (fallback nhận diện qua trang phục) khá chậm - chỉ thử lại mỗi
    # vài khung thay vì mỗi khung hình, để không làm chậm toàn bộ tiến trình.
    body_fallback_interval = max(1, int(fps * 0.3))
    frames_since_body_fallback = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện cắt cảnh bằng histogram màu HSV - ổn định hơn nhiều so với
        # so sánh pixel trực tiếp khi nền có ánh đèn sân khấu/hiệu ứng động
        # (vẫn cùng 1 cảnh nhưng pixel đổi liên tục do ánh sáng nhấp nháy).
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        scene_changed = False
        if prev_hist is not None:
            corr = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
            if corr < cut_threshold:
                is_cut[idx] = True
                scene_changed = True
        prev_hist = hist

        if scene_changed:
            tracker = None  # cảnh mới -> quên mục tiêu cũ, sẽ chọn lại từ đầu
            current_identity = None
            frames_since_body_fallback = body_fallback_interval  # cho thử ngay khung đầu cảnh mới
            # Không giữ lại vị trí của cảnh TRƯỚC (rất có thể sai hoàn toàn
            # với bố cục cảnh mới) - tạm về giữa khung hình trong lúc chờ
            # detect ở ngay khung này bắt kịp chủ thể mới.
            last_center = width / 2.0

        if tracker is not None:
            ok, box = tracker.update(frame)
            if ok:
                x, y, bw, bh = box
                last_center = x + bw / 2.0
                has_detection[idx] = True
                tracker_fail_streak = 0

                # Định kỳ xác thực lại xem tracker có còn bám đúng khuôn mặt
                # thật hay không - phòng trường hợp KCF tự tin báo "vẫn bám
                # tốt" (ok=True) trong khi thực chất đã trôi sang vùng không
                # phải khuôn mặt (nền, vật thể tĩnh...). Áp dụng cho cả 2 chế
                # độ, có hay không có ảnh mẫu.
                frames_since_recheck += 1
                if frames_since_recheck >= recheck_interval:
                    frames_since_recheck = 0
                    candidates = detect_candidates(frame, None, detector, detector_kind)

                    if use_identity:
                        match = _nearest_identity_match(
                            frame, candidates, identity_method, identity_data, face_recognizer, last_center)
                        if match is not None:
                            cand, name = match
                            # Nếu người khớp ảnh mẫu GẦN vị trí hiện tại lại ở khá
                            # xa toạ độ tracker đang báo -> tracker đã trôi/bám
                            # nhầm sang vùng khác, khóa lại đúng người.
                            if abs(cand[0] - last_center) > width * 0.08:
                                last_center = cand[0]
                                current_identity = name
                                tracker = create_tracker(frame, _clamp_box_for_tracker(cand[2]))
                            # Dù không cần khoá lại, vẫn cập nhật chữ ký trang
                            # phục mới nhất của người này (ánh sáng/góc máy có
                            # thể đổi theo cảnh) để dùng khi sang cảnh toàn.
                            sig = _appearance_signature_from_face_box(frame, cand[2])
                            if sig is not None:
                                known_signatures[name] = sig
                        # Không ai khớp ảnh mẫu gần vị trí đang bám trong lần
                        # kiểm tra này - có thể chỉ là detect tạm thời không
                        # thấy (góc mặt, mờ...), không vội reset tracker; sẽ
                        # kiểm tra lại ở lần recheck kế tiếp.
                    elif candidates:
                        # Không có ảnh mẫu: chỉ kiểm tra xem tracker hiện có
                        # còn khớp với BẤT KỲ khuôn mặt thật nào gần đó không.
                        # Nếu vị trí tracker cách xa mọi khuôn mặt phát hiện
                        # được -> tracker đã trôi sang vùng không phải mặt,
                        # sửa lại về khuôn mặt GẦN NHẤT (không phải lớn nhất,
                        # để không tự ý đổi sang người khác đang bám đúng).
                        nearest = min(candidates, key=lambda c: abs(c[0] - last_center))
                        if abs(nearest[0] - last_center) > width * 0.08:
                            last_center = nearest[0]
                            tracker = create_tracker(frame, _clamp_box_for_tracker(nearest[2]))
            else:
                tracker_fail_streak += 1
                if tracker_fail_streak > max_tracker_fail:
                    tracker = None  # mất dấu quá lâu -> phải detect lại từ đầu

        if tracker is None:
            candidates = detect_candidates(frame, None, detector, detector_kind)
            chosen = None
            if use_identity:
                match = _best_identity_match(frame, candidates, identity_method, identity_data, face_recognizer) \
                    if candidates else None
                if match is not None:
                    cand, name = match
                    chosen = cand
                    current_identity = name
                    sig = _appearance_signature_from_face_box(frame, cand[2])
                    if sig is not None:
                        known_signatures[name] = sig
                elif known_signatures and body_detector is not None:
                    # Không detect được mặt nào khớp ảnh mẫu (thường là cảnh
                    # toàn, mặt quá nhỏ) - thử nhận lại người qua màu trang
                    # phục đã ghi nhận lúc cảnh cận. HOG khá chậm nên chỉ thử
                    # lại mỗi vài khung (body_fallback_interval), các khung
                    # giữa 2 lần thử giữ nguyên vị trí gần nhất đã biết.
                    frames_since_body_fallback += 1
                    if frames_since_body_fallback >= body_fallback_interval:
                        frames_since_body_fallback = 0
                        body_match = _best_body_match_by_appearance(frame, body_detector, known_signatures)
                        if body_match is not None:
                            cx, box, name = body_match
                            chosen = (cx, box[2] * box[3], box)
                            current_identity = name

                if chosen is None and candidates:
                    # Không khớp được ai qua mặt lẫn trang phục (đoạn này
                    # không có ai trong ảnh mẫu, ví dụ cảnh toàn cảnh/chuyển
                    # cảnh) - phương án cuối: lấy khuôn mặt lớn nhất phát
                    # hiện được, còn hơn đứng yên cả đoạn dài.
                    chosen = max(candidates, key=lambda c: c[1])
                    current_identity = None
                # Nếu candidates rỗng (không detect được gì cả) -> giữ
                # nguyên vị trí gần nhất đã biết, chờ khung sau.
            elif candidates:
                chosen = max(candidates, key=lambda c: c[1])

            if chosen is not None:
                last_center = chosen[0]
                has_detection[idx] = True
                tracker_fail_streak = 0
                frames_since_recheck = 0
                tracker = create_tracker(frame, _clamp_box_for_tracker(chosen[2]))
            # Không chọn được ai -> giữ nguyên vị trí gần nhất đã biết

        centers[idx] = last_center
        idx += 1
        if idx % 500 == 0:
            log(f"  ...đã xử lý {idx}/{total_frames} khung")

    cap.release()

    # Nội suy lại cho những đoạn đầu video chưa có detection nào
    detected_idx = np.where(has_detection)[0]
    if len(detected_idx) > 0:
        first = detected_idx[0]
        centers[:first] = centers[first]
    else:
        log("  Cảnh báo: không phát hiện được chủ thể nào, sẽ crop giữa khung hình.")

    n_cuts = int(is_cut.sum())
    log(f"  Phát hiện {n_cuts} điểm cắt cảnh.")

    return centers, fps, total_frames, width, height, is_cut, has_detection


def rolling_median(values, window):
    if window <= 1:
        return values.copy()
    n = len(values)
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(padded[i:i + window])
    return out


KCF_MAX_BOX_SIZE = 150  # OpenCV KCF chậm hẳn (~30-50 lần) với box lớn hơn mức này


def _make_kcf_tracker():
    """Tạo tracker KCF, thử lần lượt các API khác nhau tùy phiên bản OpenCV
    (KCF_create bị đổi tên/di chuyển vị trí qua nhiều bản OpenCV, đặc biệt
    OpenCV 5.0 dường như tổ chức lại rất nhiều so với 4.x). Trả về None nếu
    không cách nào tạo được - khi đó code gọi sẽ tự chuyển sang chế độ detect
    lại mỗi khung hình (chậm hơn nhưng không crash)."""
    candidates = []
    if hasattr(cv2, "TrackerKCF_create"):
        candidates.append(lambda: cv2.TrackerKCF_create())
    if hasattr(cv2, "TrackerKCF") and hasattr(cv2.TrackerKCF, "create"):
        candidates.append(lambda: cv2.TrackerKCF.create())
    if hasattr(cv2, "legacy"):
        if hasattr(cv2.legacy, "TrackerKCF_create"):
            candidates.append(lambda: cv2.legacy.TrackerKCF_create())
        if hasattr(cv2.legacy, "TrackerMOSSE_create"):
            candidates.append(lambda: cv2.legacy.TrackerMOSSE_create())
    if hasattr(cv2, "TrackerMIL_create"):
        candidates.append(lambda: cv2.TrackerMIL_create())

    for make in candidates:
        try:
            return make()
        except Exception:
            continue
    return None


_tracker_unavailable_warned = [False]


def create_tracker(frame, box):
    """Tạo và khởi tạo (init) 1 tracker bám theo box đã cho. Trả về tracker
    hoặc None nếu phiên bản OpenCV này không hỗ trợ tracker nào cả - khi đó
    vòng lặp chính sẽ tự chuyển sang detect lại mỗi khung hình thay vì dùng
    tracker để bám giữa các lần detect (chậm hơn nhưng vẫn chạy được, không
    crash)."""
    tracker = _make_kcf_tracker()
    if tracker is None:
        if not _tracker_unavailable_warned[0]:
            log("  Cảnh báo: phiên bản OpenCV này không hỗ trợ tracker nào (KCF/MIL/MOSSE) - "
                "sẽ detect lại mỗi khung hình thay vì dùng tracker để bám giữa các lần detect "
                "(chậm hơn nhưng vẫn chạy được).")
            _tracker_unavailable_warned[0] = True
        return None
    try:
        tracker.init(frame, box)
        return tracker
    except Exception:
        return None


def _clamp_box_for_tracker(box):
    """Giới hạn box về tối đa KCF_MAX_BOX_SIZE x KCF_MAX_BOX_SIZE (giữ nguyên
    tâm) trước khi đưa vào KCF - OpenCV's KCF có 1 'vực chậm' nghiêm trọng
    (chậm hơn 30-50 lần) khi box theo dõi lớn, có lẽ do kích thước FFT nội bộ
    rơi vào trường hợp không tối ưu. Ta chỉ cần tracker bám đúng TÂM của đối
    tượng để tính vị trí crop, không cần theo dõi đúng kích thước gốc, nên
    thu nhỏ box lại hoàn toàn an toàn."""
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = min(w, KCF_MAX_BOX_SIZE), min(h, KCF_MAX_BOX_SIZE)
    nx, ny = int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0))
    return (max(0, nx), max(0, ny), int(nw), int(nh))


def clamp_speed(values, max_shift_per_frame):
    """Giới hạn mức thay đổi tối đa giữa 2 khung liên tiếp để tránh giật đột ngột."""
    out = values.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if delta > max_shift_per_frame:
            out[i] = out[i - 1] + max_shift_per_frame
        elif delta < -max_shift_per_frame:
            out[i] = out[i - 1] - max_shift_per_frame
    return out


def smooth_segment(centers, fps, smooth_seconds, frame_width, has_detection=None):
    """Áp dụng median filter + giới hạn tốc độ + trung bình trượt cho 1 đoạn liên tục (không có cắt cảnh)."""
    if len(centers) <= 2:
        return centers.copy()

    centers = centers.copy()
    if has_detection is not None and has_detection.any():
        # Các khung ĐẦU đoạn trước khi detect bắt được chủ thể (ví dụ 1-2
        # khung đầu cảnh mới) chỉ đang giữ giá trị tạm (giữa khung hình) -
        # thay thế bằng đúng giá trị detect được ngay khi có, để bước làm
        # mượt/giới hạn tốc độ bên dưới không hiểu nhầm đây là 1 chuyển động
        # cần "bò" dần tới, mà nhảy thẳng (snap) ngay từ đầu cảnh.
        first_det = np.argmax(has_detection)
        if first_det > 0:
            centers[:first_det] = centers[first_det]

    # 1) Lọc trung vị để loại bỏ các điểm nhảy do detect sai (outlier)
    median_window = min(len(centers), max(1, int(fps * 0.5)) | 1)
    centers = rolling_median(centers, median_window)

    # 2) Giới hạn tốc độ di chuyển tối đa mỗi khung hình, tránh giật đột ngột
    max_shift_per_frame = (frame_width * 0.04) / max(fps / 25.0, 0.5)
    centers = clamp_speed(centers, max_shift_per_frame)

    # 3) Trung bình trượt để làm mượt tổng thể
    window = min(len(centers), max(1, int(fps * smooth_seconds)))
    if window > 1:
        kernel = np.ones(window) / window
        pad = window // 2
        padded = np.pad(centers, (pad, pad), mode="edge")
        centers = np.convolve(padded, kernel, mode="same")[pad: pad + len(centers)]

    return centers


def smooth_centers(centers, fps, smooth_seconds, frame_width, is_cut, has_detection=None):
    """Làm mượt toàn bộ video, nhưng KHÔNG làm mượt xuyên qua các điểm cắt cảnh -
    mỗi cảnh mới sẽ nhảy thẳng tới vị trí chủ thể mới thay vì lia máy từ từ."""
    result = np.empty_like(centers)
    cut_indices = np.where(is_cut)[0]
    boundaries = [0] + list(cut_indices) + [len(centers)]
    boundaries = sorted(set(boundaries))

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end <= start:
            continue
        segment = centers[start:end]
        seg_has_detection = has_detection[start:end] if has_detection is not None else None
        result[start:end] = smooth_segment(segment, fps, smooth_seconds, frame_width, seg_has_detection)

    return result


def load_overlay_image(overlay_path, out_w, out_h):
    """Đọc ảnh overlay (PNG có kênh alpha), resize khớp kích thước output nếu
    cần, trả về (rgb_float, alpha_float) đã chuẩn hoá 0-1 để blend nhanh."""
    img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Không đọc được ảnh overlay: {overlay_path}")
    if img.shape[1] != out_w or img.shape[0] != out_h:
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if img.shape[2] == 4:
        rgb = img[:, :, :3].astype(np.float32)
        alpha = (img[:, :, 3:4].astype(np.float32)) / 255.0
    else:
        # Ảnh không có kênh alpha -> coi như phủ kín hoàn toàn (ít gặp)
        rgb = img.astype(np.float32)
        alpha = np.ones((out_h, out_w, 1), dtype=np.float32)
    return rgb, alpha


def render_cropped(path, out_path, centers, fps, width, height, out_w, out_h, overlay=None):
    crop_w = int(round(height * out_w / out_h))  # giữ nguyên chiều cao, tính bề rộng crop theo tỉ lệ đích
    if crop_w > width:
        crop_w = width  # video quá "hẹp" thì đành lấy full width
    half = crop_w / 2.0

    overlay_rgb, overlay_alpha = (None, None)
    if overlay is not None:
        overlay_rgb, overlay_alpha = load_overlay_image(overlay, out_w, out_h)

    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    tmp_video = out_path + ".tmp_noaudio.mp4"
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))

    idx = 0
    log(f"Đang render {os.path.basename(out_path)} (crop {crop_w}x{height} -> {out_w}x{out_h})"
        + (" + overlay" if overlay is not None else ""))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cx = centers[idx] if idx < len(centers) else centers[-1]
        x0 = int(round(cx - half))
        x0 = max(0, min(width - crop_w, x0))
        cropped = frame[:, x0:x0 + crop_w]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

        if overlay_rgb is not None:
            # Alpha-blend overlay lên trên khung đã crop: vùng trong suốt
            # (alpha=0) giữ nguyên video, vùng có alpha hiện overlay đè lên.
            resized = (resized.astype(np.float32) * (1 - overlay_alpha)
                       + overlay_rgb * overlay_alpha).astype(np.uint8)

        writer.write(resized)
        idx += 1
        if idx % 500 == 0:
            log(f"  ...đã render {idx} khung")

    cap.release()
    writer.release()

    mux_audio_and_encode(tmp_video, path, out_path)


def mux_audio_and_encode(tmp_video_noaudio, original_path, out_path):
    """Ghép âm thanh gốc (từ original_path) vào video câm tmp_video_noaudio,
    nén H.264, xuất ra out_path. Dùng chung cho mọi chế độ render."""
    log("Đang ghép âm thanh gốc bằng ffmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video_noaudio,
        "-i", original_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp_video_noaudio)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg lỗi:\n{result.stderr}")
    log(f"Xong: {out_path}")


def parse_reference_args(reference_list):
    """Chuyển list ['ten=duong_dan', ...] thành dict {ten: duong_dan}."""
    result = {}
    if not reference_list:
        return result
    for item in reference_list:
        if "=" not in item:
            raise SystemExit(
                f"Lỗi: --reference phải có dạng ten=duong_dan_anh.jpg, nhận được: '{item}'"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not os.path.isfile(path):
            raise SystemExit(f"Lỗi: không tìm thấy file ảnh mẫu '{path}' cho '{name}'")
        result[name] = path
    return result


def process_file(in_path, out_path, args, identity_method=None, identity_data=None, face_recognizer=None):
    centers, fps, total_frames, width, height, is_cut, has_detection = analyze_video(
        in_path, args.sample_every, args.detector, args.cut_threshold,
        identity_method=identity_method, identity_data=identity_data, face_recognizer=face_recognizer,
    )
    centers = smooth_centers(centers, fps, args.smooth, width, is_cut, has_detection)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    render_cropped(in_path, out_path, centers, fps, width, height, args.width, args.height,
                   overlay=args.background)


def main():
    p = argparse.ArgumentParser(description="Chuyển video 16:9 sang 9:16, crop bám theo chủ thể.")
    p.add_argument("-i", "--input", required=True, help="File video, hoặc thư mục nếu dùng --batch")
    p.add_argument("-o", "--output", required=True, help="File output, hoặc thư mục output nếu dùng --batch")
    p.add_argument("--batch", action="store_true", help="Xử lý toàn bộ video trong thư mục --input")
    p.add_argument("--width", type=int, default=1080, help="Chiều rộng output (mặc định 1080)")
    p.add_argument("--height", type=int, default=1920, help="Chiều cao output (mặc định 1920)")
    p.add_argument("--smooth", type=float, default=1.0, help="Độ mượt camera tính bằng giây (mặc định 1.0)")
    p.add_argument("--sample-every", type=int, default=3, help="Detect mỗi N khung hình (mặc định 3)")
    p.add_argument("--detector", choices=["face", "hog"], default="face", help="Kiểu phát hiện chủ thể")
    p.add_argument("--cut-threshold", type=float, default=0.75,
                    help="Độ nhạy phát hiện cắt cảnh, từ 0-1 (mặc định 0.75). Số CÀNG THẤP "
                         "= càng khó bị coi là cắt cảnh (ít nhạy hơn); số càng cao = càng dễ "
                         "bị coi là cắt cảnh (nhạy hơn, dễ bắt nhầm cả cảnh sáng/tối thay đổi).")
    p.add_argument("--reference", action="append", metavar="TEN=DUONG_DAN_ANH",
                    help="Ảnh mẫu khuôn mặt của 1 người, dạng ten=duong_dan.jpg. Có thể "
                         "dùng nhiều lần cho nhiều người (--reference an=an.jpg --reference "
                         "binh=binh.jpg). Khi có ảnh mẫu, công cụ CHỈ bám theo người khớp ảnh "
                         "mẫu, giúp tránh bám nhầm khuôn mặt trên màn hình LED/phông nền.")
    p.add_argument("--background", metavar="DUONG_DAN_ANH_PNG",
                    help="Ảnh overlay (PNG có kênh alpha/vùng trong suốt) phủ đè lên trên "
                         "video sau khi đã crop 9:16 như bình thường - dùng để chèn khung/"
                         "tiêu đề/logo. Vùng trong suốt của ảnh sẽ để lộ video bên dưới, "
                         "vùng không trong suốt sẽ che phủ. Ảnh nên có kích thước đúng bằng "
                         "--width x --height (mặc định 1080x1920), nếu khác sẽ tự resize.")
    args = p.parse_args()

    if args.background and not os.path.isfile(args.background):
        raise SystemExit(f"Lỗi: không tìm thấy file ảnh nền '{args.background}'")

    reference_paths = parse_reference_args(args.reference)
    identity_method, identity_data, face_recognizer = (None, None, None)
    if reference_paths:
        if args.detector != "face":
            log("Cảnh báo: --reference chỉ hoạt động với --detector face, sẽ bỏ qua ảnh mẫu.")
        else:
            face_detector_for_ref = get_face_detector()
            face_recognizer = get_face_recognizer()
            identity_method, identity_data = load_reference_identities(
                reference_paths, face_detector_for_ref, face_recognizer
            )
            if identity_method is None:
                log("Cảnh báo: không học được người nào từ ảnh mẫu, sẽ chạy như không có ảnh mẫu.")

    if args.batch:
        os.makedirs(args.output, exist_ok=True)
        files = sorted(
            f for f in glob.glob(os.path.join(args.input, "*"))
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        )
        if not files:
            log(f"Không tìm thấy video nào trong {args.input}")
            sys.exit(1)
        log(f"Tìm thấy {len(files)} video. Bắt đầu xử lý hàng loạt...")
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(args.output, f"{name}_9x16.mp4")
            try:
                process_file(f, out_path, args, identity_method, identity_data, face_recognizer)
            except Exception as e:
                log(f"LỖI khi xử lý {f}: {e}")
        log("Hoàn tất toàn bộ thư mục.")
    else:
        process_file(args.input, args.output, args, identity_method, identity_data, face_recognizer)


if __name__ == "__main__":
    main()