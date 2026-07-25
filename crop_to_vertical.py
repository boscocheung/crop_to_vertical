#!/usr/bin/env python3
"""
crop_to_vertical.py - Chuyển video 16:9 → 9:16, tự bám theo chủ thể
Quét video phát hiện khuôn mặt, bám theo, làm mượt, cắt video theo dọc
"""

import argparse
import glob
import os
import subprocess
import sys

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

def log(msg):
    print(f"[crop_to_vertical] {msg}", flush=True)

def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

def get_hog_detector():
    if not hasattr(cv2, "HOGDescriptor"):
        return None
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog
    except Exception:
        return None

def detect_candidates(frame, detector, kind):
    h, w = frame.shape[:2]
    if kind == "face":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [(x + fw / 2.0, fw * fh, (x, y, fw, fh)) for (x, y, fw, fh) in faces]
    else:
        rects, weights = detector.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return [(rx + rw / 2.0, rw * rh, (int(rx), int(ry), int(rw), int(rh))) for (rx, ry, rw, rh) in rects]

def create_tracker(frame, box):
    if not hasattr(cv2, "TrackerKCF_create"):
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            tracker = cv2.legacy.TrackerKCF_create()
        else:
            return None
    else:
        tracker = cv2.TrackerKCF_create()
    try:
        tracker.init(frame, box)
        return tracker
    except Exception:
        return None

def analyze_video(path, detector_kind):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = get_face_detector() if detector_kind == "face" else get_hog_detector()
    if detector_kind == "hog" and detector is None:
        raise RuntimeError("HOG detector không hỗ trợ.")

    centers = np.full(total_frames, width / 2.0, dtype=np.float64)
    has_detection = np.zeros(total_frames, dtype=bool)
    is_cut = np.zeros(total_frames, dtype=bool)

    log(f"Phân tích {os.path.basename(path)}: {total_frames} khung, {width}x{height}, {fps:.2f} fps")

    idx = 0
    last_center = width / 2.0
    prev_hist = None
    tracker = None
    tracker_fail_streak = 0
    max_tracker_fail = max(3, int(fps * 0.3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        scene_changed = False
        if prev_hist is not None:
            corr = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
            if corr < 0.75:
                is_cut[idx] = True
                scene_changed = True
        prev_hist = hist

        if scene_changed:
            tracker = None
            last_center = width / 2.0

        if tracker is not None:
            ok, box = tracker.update(frame)
            if ok:
                x, y, bw, bh = box
                last_center = x + bw / 2.0
                has_detection[idx] = True
                tracker_fail_streak = 0
            else:
                tracker_fail_streak += 1
                if tracker_fail_streak > max_tracker_fail:
                    tracker = None

        if tracker is None:
            candidates = detect_candidates(frame, detector, detector_kind)
            if candidates:
                chosen = max(candidates, key=lambda c: c[1])
                last_center = chosen[0]
                has_detection[idx] = True
                tracker_fail_streak = 0
                tracker = create_tracker(frame, chosen[2])

        centers[idx] = last_center
        idx += 1
        if idx % 500 == 0:
            log(f"  ...đã xử lý {idx}/{total_frames} khung")

    cap.release()

    detected_idx = np.where(has_detection)[0]
    if len(detected_idx) > 0:
        first = detected_idx[0]
        centers[:first] = centers[first]

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
        out[i] = np.median(padded[i : i + window])
    return out

def clamp_speed(values, max_shift_per_frame):
    out = values.copy()
    for i in range(1, len(out)):
        delta = out[i] - out[i - 1]
        if delta > max_shift_per_frame:
            out[i] = out[i - 1] + max_shift_per_frame
        elif delta < -max_shift_per_frame:
            out[i] = out[i - 1] - max_shift_per_frame
    return out

def smooth_segment(centers, fps, smooth_seconds, frame_width, has_detection=None):
    if len(centers) <= 2:
        return centers.copy()

    centers = centers.copy()
    if has_detection is not None and has_detection.any():
        first_det = np.argmax(has_detection)
        if first_det > 0:
            centers[:first_det] = centers[first_det]

    median_window = min(len(centers), max(1, int(fps * 0.5)) | 1)
    centers = rolling_median(centers, median_window)

    max_shift_per_frame = (frame_width * 0.04) / max(fps / 25.0, 0.5)
    centers = clamp_speed(centers, max_shift_per_frame)

    window = min(len(centers), max(1, int(fps * smooth_seconds)))
    if window > 1:
        kernel = np.ones(window) / window
        pad = window // 2
        padded = np.pad(centers, (pad, pad), mode="edge")
        centers = np.convolve(padded, kernel, mode="same")[pad : pad + len(centers)]

    return centers

def smooth_centers(centers, fps, smooth_seconds, frame_width, is_cut, has_detection=None):
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

def render_cropped(path, out_path, centers, fps, width, height, out_w, out_h):
    crop_w = int(round(height * out_w / out_h))
    if crop_w > width:
        crop_w = width
    half = crop_w / 2.0

    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_video = out_path + ".tmp_noaudio.mp4"
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))

    idx = 0
    log(f"Render {os.path.basename(out_path)} (crop {crop_w}x{height} -> {out_w}x{out_h})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cx = centers[idx] if idx < len(centers) else centers[-1]
        x0 = int(round(cx - half))
        x0 = max(0, min(width - crop_w, x0))
        cropped = frame[:, x0 : x0 + crop_w]
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        idx += 1
        if idx % 500 == 0:
            log(f"  ...đã render {idx} khung")

    cap.release()
    writer.release()

    log("Ghép âm thanh gốc bằng ffmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(tmp_video)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg lỗi:\n{result.stderr}")
    log(f"Xong: {out_path}")

def process_file(in_path, out_path, args):
    centers, fps, total_frames, width, height, is_cut, has_detection = analyze_video(
        in_path, args.detector
    )
    centers = smooth_centers(centers, fps, args.smooth, width, is_cut, has_detection)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    render_cropped(in_path, out_path, centers, fps, width, height, args.width, args.height)

def main():
    p = argparse.ArgumentParser(description="Chuyển video 16:9 sang 9:16, crop bám theo chủ thể.")
    p.add_argument("-i", "--input", required=True, help="File video hoặc thư mục nếu dùng --batch")
    p.add_argument("-o", "--output", required=True, help="File output hoặc thư mục output")
    p.add_argument("--batch", action="store_true", help="Xử lý toàn bộ video trong thư mục")
    p.add_argument("--width", type=int, default=1080, help="Chiều rộng output (mặc định 1080)")
    p.add_argument("--height", type=int, default=1920, help="Chiều cao output (mặc định 1920)")
    p.add_argument("--smooth", type=float, default=1.0, help="Độ mượt chuyển động (giây, mặc định 1.0)")
    p.add_argument("--detector", choices=["face", "hog"], default="face", help="Kiểu phát hiện")
    args = p.parse_args()

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
                process_file(f, out_path, args)
            except Exception as e:
                log(f"LỖI khi xử lý {f}: {e}")
        log("Hoàn tất toàn bộ thư mục.")
    else:
        process_file(args.input, args.output, args)

if __name__ == "__main__":
    main()
