import math
import os
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


def _read_video(path: Path, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError(f"Video has no readable frames: {path}")

    return frames, fps, (width, height)


def _write_video(frames: List[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not create video writer: {path}")

    for frame in frames:
        writer.write(frame)

    writer.release()


def _center_box(width: int, height: int) -> Box:
    box_w = int(width * 0.42)
    box_h = int(height * 0.72)
    x = (width - box_w) // 2
    y = (height - box_h) // 2
    return x, y, box_w, box_h


def _motion_box(frame: np.ndarray, previous: Optional[np.ndarray]) -> Box:
    height, width = frame.shape[:2]
    fallback = _center_box(width, height)
    if previous is None:
        return fallback

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > width * height * 0.004]
    if not contours:
        return fallback

    x1, y1, x2, y2 = width, height, 0, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x + w)
        y2 = max(y2, y + h)

    pad_x = int((x2 - x1) * 0.25)
    pad_y = int((y2 - y1) * 0.35)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return fallback

    return x1, y1, x2 - x1, y2 - y1


def _box_center(box: Box) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _box_score(a: Box, b: Box, frame_size: Tuple[int, int]) -> float:
    width, height = frame_size
    ax, ay = _box_center(a)
    bx, by = _box_center(b)

    center_dist = math.hypot((ax - bx) / width, (ay - by) / height)
    area_a = max(1.0, a[2] * a[3])
    area_b = max(1.0, b[2] * b[3])
    area_ratio = abs(math.log(area_a / area_b))
    aspect_ratio = abs(math.log((a[2] / max(1, a[3])) / (b[2] / max(1, b[3]))))

    return center_dist * 0.58 + area_ratio * 0.27 + aspect_ratio * 0.15


def _match_frame(frames_a: List[np.ndarray], frames_b: List[np.ndarray], fps: float) -> Tuple[int, Box, Box]:
    reference_index = max(1, len(frames_a) - 1)
    reference_box = _motion_box(frames_a[reference_index], frames_a[reference_index - 1])

    search_limit = min(len(frames_b), max(1, int(fps * 2.5)))
    best_index = 0
    best_box = _motion_box(frames_b[0], None)
    best_score = float("inf")
    frame_size = (frames_a[0].shape[1], frames_a[0].shape[0])

    for i in range(search_limit):
        previous = frames_b[i - 1] if i > 0 else None
        candidate_box = _motion_box(frames_b[i], previous)
        score = _box_score(reference_box, candidate_box, frame_size)
        if score < best_score:
            best_score = score
            best_index = i
            best_box = candidate_box

    return best_index, reference_box, best_box


def _align_frame(frame: np.ndarray, source_box: Box, target_box: Box, output_size: Tuple[int, int]) -> np.ndarray:
    out_w, out_h = output_size
    sx, sy, sw, sh = source_box
    tx, ty, tw, th = target_box

    source_scale = max(sw, sh, 1)
    target_scale = max(tw, th, 1)
    scale = target_scale / source_scale

    scx, scy = _box_center(source_box)
    tcx, tcy = _box_center(target_box)

    matrix = np.array([
        [scale, 0.0, tcx - scx * scale],
        [0.0, scale, tcy - scy * scale],
    ], dtype=np.float32)

    return cv2.warpAffine(
        frame,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _ease_in_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def _slow_motion_tail(frames: List[np.ndarray], count: int, factor: int) -> List[np.ndarray]:
    if count <= 0 or factor <= 1:
        return frames[-max(0, count):]
    tail = frames[-count:]
    slowed: List[np.ndarray] = []
    for frame in tail:
        slowed.extend([frame] * factor)
    return slowed


def _speed_ramp_head(frames: List[np.ndarray], start_index: int, count: int) -> List[np.ndarray]:
    head = frames[start_index : min(len(frames), start_index + count)]
    ramped: List[np.ndarray] = []
    for i, frame in enumerate(head):
        hold = max(1, 3 - int((i / max(1, count - 1)) * 2))
        ramped.extend([frame] * hold)
    return ramped


def render_match_morph_transition(
    clip_a: Path,
    clip_b: Path,
    output_dir: Path,
    slowmo_seconds: float = 0.8,
    morph_seconds: float = 0.33,
    restore_seconds: float = 0.5,
    no_blur: bool = True,
) -> Path:
    if not no_blur:
        raise ValueError("This effect is configured as no-blur only")

    output_dir.mkdir(parents=True, exist_ok=True)

    frames_a, fps_a, size_a = _read_video(clip_a)
    frames_b, fps_b, size_b = _read_video(clip_b)

    fps = fps_a or fps_b or 30.0
    out_w, out_h = size_a
    frames_b = [cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR) for frame in frames_b]

    slow_count = min(len(frames_a), max(1, int(fps * slowmo_seconds)))
    morph_count = max(2, int(fps * morph_seconds))
    restore_count = max(1, int(fps * restore_seconds))

    match_index_b, box_a, box_b = _match_frame(frames_a, frames_b, fps)

    normal_a = frames_a[: max(0, len(frames_a) - slow_count)]
    slowed_a = _slow_motion_tail(frames_a, slow_count, factor=3)

    hold_a = frames_a[-1]
    aligned_b = _align_frame(frames_b[match_index_b], box_b, box_a, (out_w, out_h))

    morph_frames: List[np.ndarray] = []
    for i in range(morph_count):
        alpha = _ease_in_out(i / max(1, morph_count - 1))
        # Crisp opacity blend only: no Gaussian blur, no motion blur, no softened mask.
        blended = cv2.addWeighted(hold_a, 1.0 - alpha, aligned_b, alpha, 0.0)
        morph_frames.append(blended)

    ramp_b = _speed_ramp_head(frames_b, match_index_b, restore_count)
    normal_b = frames_b[min(len(frames_b), match_index_b + restore_count) :]

    output_frames = normal_a + slowed_a + morph_frames + ramp_b + normal_b
    output_path = output_dir / f"match_morph_{uuid.uuid4().hex}.mp4"
    _write_video(output_frames, output_path, fps)

    return output_path


def mux_audio_from_clip(video_path: Path, audio_source: Path, output_dir: Path) -> Path:
    output_path = output_dir / f"{video_path.stem}_audio.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_path if output_path.exists() else video_path
