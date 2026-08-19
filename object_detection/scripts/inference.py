#!/usr/bin/env python3
"""
Run the fine-tuned ADAS detector on images, a video file, or a live camera.

Class names are read from the loaded checkpoint (model.names) rather than
hardcoded, so this works correctly with any of the shipped models.

Usage:
    python inference.py --model ../models/yolo26s_best.pt --source ../test_images --type image
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def inference_images(model_path: str, image_dir: str, conf: float, iou: float, save_dir: str) -> None:
    model = YOLO(model_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    print(f"\nRunning inference on {len(image_files)} images (conf={conf}, iou={iou})\n")

    for idx, img_file in enumerate(image_files):
        start = time.time()
        results = model.predict(source=str(img_file), conf=conf, iou=iou, verbose=False)
        latency_ms = (time.time() - start) * 1000

        annotated_img = results[0].plot()
        output_path = save_dir / f"pred_{img_file.name}"
        cv2.imwrite(str(output_path), annotated_img)

        num_detections = len(results[0].boxes)
        print(f"  [{idx + 1}/{len(image_files)}] {img_file.name:<30} "
              f"detections: {num_detections:<3} | latency: {latency_ms:.1f}ms")

    print(f"\nPredictions saved to: {save_dir}")


def inference_video(model_path: str, video_path: str, conf: float, iou: float, output_path: str | None) -> None:
    model = YOLO(model_path)
    class_names = model.names

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Can't open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing video: {video_path} ({width}x{height} @ {fps:.1f}fps, {total_frames} frames)\n")

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_count = 0
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        start = time.time()
        results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        latency = time.time() - start
        current_fps = 1 / latency if latency > 0 else 0.0
        fps_history.append(current_fps)

        annotated = results[0].plot()
        cv2.putText(annotated, f'FPS: {current_fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        detected = {class_names[int(b.cls[0])] for b in results[0].boxes}
        cv2.putText(annotated, f"Classes: {', '.join(list(detected)[:5])}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        cv2.imshow('ADAS inference', annotated)
        if out:
            out.write(annotated)

        if frame_count % 30 == 0:
            avg_fps = sum(fps_history[-30:]) / 30
            print(f"  frame {frame_count}/{total_frames} | avg FPS: {avg_fps:.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    if fps_history:
        print(f"\nDone — {frame_count} frames, avg FPS: {sum(fps_history) / len(fps_history):.1f}")
    if output_path:
        print(f"Output video: {output_path}")


def inference_camera(model_path: str, conf: float, iou: float, camera_id: int) -> None:
    model = YOLO(model_path)
    class_names = model.names

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Can't open camera {camera_id}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"\nLive inference from camera {camera_id} — press 'q' to quit, 's' to save a frame\n")
    frame_count, save_count = 0, 0
    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        start = time.time()
        results = model.predict(source=frame, conf=conf, iou=iou, verbose=False)
        latency = time.time() - start
        current_fps = 1 / latency if latency > 0 else 0.0
        fps_history.append(current_fps)

        annotated = results[0].plot()
        cv2.putText(annotated, f'FPS: {current_fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        y_offset = 70
        for b in list(results[0].boxes)[:5]:
            label = f"{class_names[int(b.cls[0])]}({float(b.conf[0]):.2f})"
            cv2.putText(annotated, label, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            y_offset += 30

        cv2.imshow('ADAS live inference (q to quit, s to save)', annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f'adas_capture_{save_count}.jpg'
            cv2.imwrite(save_path, annotated)
            print(f"  saved: {save_path}")
            save_count += 1

        if frame_count % 30 == 0:
            avg_fps = sum(fps_history[-30:]) / 30
            print(f"  frame {frame_count} | avg FPS: {avg_fps:.1f}")

    cap.release()
    cv2.destroyAllWindows()
    if fps_history:
        print(f"\nDone — {frame_count} frames, avg FPS: {sum(fps_history) / len(fps_history):.1f}, "
              f"{save_count} frames saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ADAS detector on images/video/camera')
    parser.add_argument('--model', required=True, help='Path to a trained checkpoint (.pt)')
    parser.add_argument('--source', required=True, help='Image directory, video file, or camera ID')
    parser.add_argument('--type', default='auto', choices=['image', 'video', 'camera', 'auto'])
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--output', help='Output video path (video mode only)')
    parser.add_argument('--save-dir', default='./predictions', help='Save directory (image mode only)')
    args = parser.parse_args()

    source_type = args.type
    if source_type == 'auto':
        if args.source.isdigit():
            source_type = 'camera'
        elif Path(args.source).is_dir():
            source_type = 'image'
        else:
            source_type = 'video'

    if source_type == 'image':
        inference_images(args.model, args.source, args.conf, args.iou, args.save_dir)
    elif source_type == 'video':
        inference_video(args.model, args.source, args.conf, args.iou, args.output)
    else:
        inference_camera(args.model, args.conf, args.iou, int(args.source))
