#!/usr/bin/env python3
"""
yolov8n-face 모델로 영상에서 두 눈이 처음 등장하는 시점(face_timestamp.start)을 탐지해
기존 face_prediction JSON 파일에 face_timestamp 필드를 추가하는 스크립트.

조건:
  - face bounding box confidence >= 0.7
  - 두 눈(left_eye, right_eye) 키포인트가 모두 confidence >= 0.7 로 감지
  - 눈 간격이 face box 너비의 15% 이상 (뒤돌아있으면 두 눈이 x축상 겹쳐 보임)
  - 위 세 조건을 모두 만족하는 첫 프레임을 face_timestamp.start 로 기록
  - 사람이 없는 영상(person_timestamp=null) → face_timestamp=null (추론 생략)
  - face_timestamp 키가 이미 있는 JSON → 스킵 (재실행 가능)

사용법:
  python run_face_timestamp.py predict
  python run_face_timestamp.py predict --model yolov8n-face.pt --frame-skip 1 --device 0
"""

import json
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

SEQUENCES_ROOT = Path("/nas03/homeai_dataset/root/sequences")
FACE_PRED_ROOT = Path("/nas03/homeai_dataset/root/face_prediction")
SKIP_FILES     = {"face_all.json", "face_unknown.json"}
MODEL_PATH     = "yolov8n-face.pt"
VIDEO_EXTS     = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}

# yolov8-face 키포인트: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
LEFT_EYE_IDX       = 0
RIGHT_EYE_IDX      = 1
NOSE_IDX           = 2
EYE_CONF_THRESHOLD = 0.7   # 눈 키포인트 confidence (0.5 → 0.7)
FACE_CONF_THRESHOLD = 0.7  # 얼굴 bounding box confidence
# 눈 간격이 face box 너비 대비 최소 비율 (너무 붙어있으면 false positive)
MIN_EYE_DIST_RATIO = 0.15


def frame_to_timestamp(frame_num: int, fps: float) -> str:
    total_seconds = int(frame_num / fps)
    hours   = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def both_eyes_visible(keypoints_data, box_conf: float,
                      eye_conf_thresh: float, face_conf_thresh: float,
                      box_w: float) -> bool:
    """
    두 눈이 정면으로 보이는지 다중 조건으로 확인.
      1) face bounding box confidence >= face_conf_thresh
      2) 두 눈 키포인트 confidence >= eye_conf_thresh
      3) 눈 간격이 face box 너비의 MIN_EYE_DIST_RATIO 이상 (뒤돌아있으면 눈이 겹쳐보임)
    """
    if keypoints_data.shape[0] < 3:
        return False

    # 1) face box confidence
    if float(box_conf) < face_conf_thresh:
        return False

    # 2) 눈 키포인트 confidence
    left_conf  = float(keypoints_data[LEFT_EYE_IDX,  2])
    right_conf = float(keypoints_data[RIGHT_EYE_IDX, 2])
    if left_conf < eye_conf_thresh or right_conf < eye_conf_thresh:
        return False

    # 3) 눈 간격 검사: 뒤돌아있으면 두 눈이 거의 같은 x좌표에 몰림
    lx = float(keypoints_data[LEFT_EYE_IDX,  0])
    rx = float(keypoints_data[RIGHT_EYE_IDX, 0])
    eye_dist = abs(lx - rx)
    if box_w > 0 and (eye_dist / box_w) < MIN_EYE_DIST_RATIO:
        return False

    return True


def find_first_both_eyes(video_path: Path, model, frame_skip: int, device: str):
    """영상에서 두 눈이 처음 등장하는 프레임의 timestamp 반환. 없으면 None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"         ⚠️  영상 열기 실패: {video_path}")
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"         총 {total_frames} 프레임, {fps:.1f} FPS")

    result_ts = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        results = model(frame, verbose=False, device=device)

        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            kps   = result.keypoints.data  # (n_faces, 5, 3)
            boxes = result.boxes           # xyxy + conf + cls
            for i, face_kps in enumerate(kps):
                box_conf = float(boxes.conf[i]) if i < len(boxes.conf) else 0.0
                box_w    = float(boxes.xywh[i][2]) if i < len(boxes.xywh) else 0.0
                if both_eyes_visible(face_kps, box_conf,
                                     EYE_CONF_THRESHOLD, FACE_CONF_THRESHOLD,
                                     box_w):
                    result_ts = frame_to_timestamp(frame_idx, fps)
                    break
            if result_ts is not None:
                break

        if result_ts is not None:
            break

        if total_frames > 0 and frame_idx % (total_frames // 10 + 1) == 0:
            print(f"         진행률: {frame_idx / total_frames * 100:.0f}%")

        frame_idx += 1

    cap.release()
    return result_ts


def run_reset(json_root: Path):
    """모든 JSON에서 face_timestamp 키 제거."""
    json_files = sorted(
        f for f in json_root.rglob("*.json")
        if f.name not in SKIP_FILES
    )
    removed = 0
    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if "face_timestamp" in data:
            del data["face_timestamp"]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            removed += 1
    print(f"[reset] 완료: {removed}개 파일에서 face_timestamp 제거")


def run_predict(video_root: Path, json_root: Path, model_path: str,
                frame_skip: int, device: str):
    json_files = sorted(
        f for f in json_root.rglob("*.json")
        if f.name not in SKIP_FILES
    )
    total = len(json_files)

    print(f"[predict] 모델 로딩: {model_path}")
    model = YOLO(model_path)
    print(f"[predict] 모델 로드 완료!")
    print(f"[predict] 총 {total}개 JSON 발견\n")

    skipped = 0; null_person = 0; detected = 0; not_detected = 0

    for idx, json_path in enumerate(json_files, 1):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        video_name = data.get("video_name", "")
        human_id   = data.get("human_id",   "")
        room_id    = data.get("room_id",    "")
        sequence   = data.get("sequence",   "")
        person_ts  = data.get("person_timestamp")

        # 이미 처리된 파일 스킵
        if "face_timestamp" in data:
            print(f"[{idx}/{total}] 스킵: {sequence}/{human_id}/{room_id}/{video_name}")
            skipped += 1
            continue

        print(f"[{idx}/{total}] 처리: {sequence}/{human_id}/{room_id}/{video_name}")

        # person_timestamp=null → 사람 없음, 추론 생략
        if person_ts is None:
            print(f"         사람 없음 → face_timestamp=null")
            data["face_timestamp"] = None
            null_person += 1

        else:
            video_path = video_root / sequence / human_id / room_id / video_name
            if not video_path.exists():
                print(f"         ⚠️  영상 파일 없음: {video_path}")
                data["face_timestamp"] = None
                not_detected += 1
            else:
                first_ts = find_first_both_eyes(video_path, model, frame_skip, device)
                if first_ts is not None:
                    print(f"         👀 두 눈 첫 등장: {first_ts}")
                    data["face_timestamp"] = {"start": first_ts}
                    detected += 1
                else:
                    print(f"         ❌ 두 눈 미감지 → face_timestamp=null")
                    data["face_timestamp"] = None
                    not_detected += 1

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"         ✅ 저장 완료\n")

    print(f"\n[predict] 완료! → {json_root}")
    print(f"          전체={total}  스킵={skipped}  사람없음={null_person}  "
          f"얼굴감지={detected}  미감지={not_detected}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_r = sub.add_parser("reset")
    p_r.add_argument("--json-root", type=Path, default=FACE_PRED_ROOT)

    p_p = sub.add_parser("predict")
    p_p.add_argument("--model",      type=str,  default=MODEL_PATH)
    p_p.add_argument("--video-root", type=Path, default=SEQUENCES_ROOT)
    p_p.add_argument("--json-root",  type=Path, default=FACE_PRED_ROOT)
    p_p.add_argument("--frame-skip", type=int,  default=1)
    p_p.add_argument("--device",     type=str,  default="0")

    args = parser.parse_args()

    if args.mode == "reset":
        run_reset(args.json_root)
    elif args.mode == "predict":
        run_predict(args.video_root, args.json_root, args.model,
                    args.frame_skip, args.device)


if __name__ == "__main__":
    main()
