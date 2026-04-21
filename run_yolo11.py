#!/usr/bin/env python3
import os
import json
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

SEQUENCES_ROOT = Path("/nas03/homeai_dataset/root/sequences")
PREDICT_OUTPUT = Path("/nas03/homeai_dataset/root/face_prediction")
MODEL_PATH = "yolo11n.pt"
VIDEO_EXTS = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}

def iter_videos(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in VIDEO_EXTS:
            yield p

def frame_to_timestamp(frame_num: int, fps: float) -> str:
    total_seconds = int(frame_num / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_metadata_from_path(video_path: Path, root: Path) -> dict:
    try:
        rel_parts = video_path.relative_to(root).parts
        if len(rel_parts) >= 4:
            return {"sequence": rel_parts[0], "human_id": rel_parts[1], "room_id": rel_parts[2]}
    except ValueError:
        pass
    return {"sequence": "", "human_id": "", "room_id": ""}

def detect_human_timestamps(video_path: Path, model, threshold_frames: int = 5) -> list:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = []
    person_present = False
    start_frame = None
    consecutive_detect = 0
    consecutive_absent = 0
    frame_idx = 0
    print(f"         총 {total_frames} 프레임, {fps:.1f} FPS")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        persons_detected = False
        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0:
                    persons_detected = True
                    break
            if persons_detected:
                break
        if persons_detected:
            consecutive_detect += 1
            consecutive_absent = 0
            if not person_present and consecutive_detect >= threshold_frames:
                person_present = True
                start_frame = frame_idx - threshold_frames + 1
                print(f"         👤 등장: {frame_to_timestamp(start_frame, fps)}")
        else:
            consecutive_absent += 1
            consecutive_detect = 0
            if person_present and consecutive_absent >= threshold_frames:
                person_present = False
                end_frame = frame_idx - threshold_frames + 1
                timestamps.append({"start": frame_to_timestamp(start_frame, fps), "end": frame_to_timestamp(end_frame, fps)})
                print(f"         👋 퇴장: {frame_to_timestamp(end_frame, fps)}")
        frame_idx += 1
        if total_frames > 0 and frame_idx % (total_frames // 10 + 1) == 0:
            print(f"         진행률: {frame_idx / total_frames * 100:.0f}%")
    if person_present and start_frame is not None:
        timestamps.append({"start": frame_to_timestamp(start_frame, fps), "end": frame_to_timestamp(frame_idx - 1, fps)})
    cap.release()
    return timestamps

def run_predict(video_root: Path, output_dir: Path, model_path: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[predict] 모델 로딩: {model_path}")
    model = YOLO(model_path)
    print("[predict] 모델 로드 완료!")
    video_list = list(iter_videos(video_root))
    print(f"[predict] 총 {len(video_list)}개 동영상 발견\n")
    for idx, video_path in enumerate(video_list, 1):
        meta = extract_metadata_from_path(video_path, video_root)
        sequence, human_id, room_id = meta["sequence"], meta["human_id"], meta["room_id"]
        video_name = video_path.name
        json_path = output_dir / sequence / human_id / room_id / (video_path.stem + ".json")
        if json_path.exists():
            print(f"[{idx}/{len(video_list)}] 스킵: {json_path.relative_to(output_dir)}")
            continue
        print(f"[{idx}/{len(video_list)}] 처리: {sequence}/{human_id}/{room_id}/{video_name}")
        timestamps = detect_human_timestamps(video_path, model)
        result = {"video_name": video_name, "human_id": human_id, "room_id": room_id, "sequence": sequence, "timestamp": timestamps}
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"         ✅ 저장 완료 ({len(timestamps)}개 구간)\n")
    print(f"\n[predict] 완료! → {output_dir}")

def run_merge(json_root: Path, output_file: Path):
    json_files = sorted(json_root.rglob("*.json"))
    if not json_files:
        print(f"[merge] JSON 없음")
        return
    merged = [json.load(open(jf, encoding="utf-8")) for jf in json_files]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[merge] {len(merged)}개 합침 → {output_file}")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p_p = sub.add_parser("predict")
    p_p.add_argument("--video-root", type=Path, default=SEQUENCES_ROOT)
    p_p.add_argument("--output-dir", type=Path, default=PREDICT_OUTPUT)
    p_p.add_argument("--model", type=str, default=MODEL_PATH)
    p_m = sub.add_parser("merge")
    p_m.add_argument("--json-root", type=Path, default=PREDICT_OUTPUT)
    p_m.add_argument("--output", type=Path, default=Path("merged.json"))
    args = parser.parse_args()
    if args.mode == "predict":
        run_predict(args.video_root, args.output_dir, args.model)
    elif args.mode == "merge":
        run_merge(args.json_root, args.output)

if __name__ == "__main__":
    main()