import torch
from ultralytics import YOLO
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import time
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
import logging
import warnings

warnings.filterwarnings("ignore")


class HumanDetector:
    def __init__(self):
        self.console = Console()
        self.setup_model()
        self.setup_visual_params()

    def setup_model(self):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = YOLO("yolov8x.pt")
            self.model.to(self.device)
            self.model.conf = 0.35
            self.model.iou = 0.45

            with self.console.status("[bold green]Loading AI model..."):

                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model(dummy, verbose=False)
        except Exception as e:
            self.console.print(f"[bold red]Model setup error: {str(e)}")
            raise

    def setup_visual_params(self):

        self.box_color = (75, 199, 255)
        self.text_color = (255, 255, 255)
        self.box_thickness = 2
        self.text_size = 0.7
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_detection(self, frame, box, conf):
        x1, y1, x2, y2 = map(int, box)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.box_color, -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.box_thickness)

        label = f"Person {conf:.1%}"
        (label_w, label_h), _ = cv2.getTextSize(label, self.font, self.text_size, 1)
        margin = 5

        cv2.rectangle(
            frame,
            (x1 - margin, y1 - label_h - 2 * margin),
            (x1 + label_w + margin, y1),
            self.box_color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - margin),
            self.font,
            self.text_size,
            self.text_color,
            2,
        )

        return frame

    def process_frame(self, frame):
        try:
            results = self.model(frame, classes=0, verbose=False)[0]
            detections = []

            for box in results.boxes:
                if box.conf[0] > self.model.conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    frame = self.draw_detection(frame, (x1, y1, x2, y2), conf)
                    detections.append((x1, y1, x2, y2, conf))

            return frame, len(detections) > 0

        except Exception as e:
            self.console.print(f"[bold red]Frame processing error: {str(e)}")
            return frame, False


class VideoProcessor:
    def __init__(self):
        self.detector = HumanDetector()
        self.console = Console()

    def process_video(self, input_path: str, output_path: str):
        try:
            if not Path(input_path).exists():
                raise FileNotFoundError(f"Input video not found: {input_path}")

            with self.console.status("[bold green]Loading video...") as status:
                video = VideoFileClip(input_path)
                total_frames = int(video.duration * video.fps)

                output = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video.fps,
                    (int(video.size[0]), int(video.size[1])),
                )

            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Processing video...", total=total_frames
                )

                frame_count = 0
                human_frames = 0
                start_time = time.time()

                while frame_count < total_frames:
                    frame = video.get_frame(frame_count / video.fps)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    processed_frame, has_humans = self.detector.process_frame(frame_bgr)

                    if has_humans:
                        output.write(processed_frame)
                        human_frames += 1

                    frame_count += 1
                    progress.update(task, advance=1)

                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                stats = f"""
                Processed {frame_count} frames in {elapsed_time:.1f} seconds
                Average FPS: {fps:.1f}
                Found humans in {human_frames} frames
                Output saved to: {output_path}
                """

                self.console.print(
                    Panel(stats, title="Processing Complete", border_style="green")
                )

            video.close()
            output.release()
            cv2.destroyAllWindows()

        except Exception as e:
            self.console.print(
                Panel(f"Error: {str(e)}", title="Error", border_style="red")
            )
            raise


def main():
    INPUT_VIDEO = "input.mp4"
    OUTPUT_VIDEO = "output_detected.mp4"

    processor = VideoProcessor()
    processor.process_video(INPUT_VIDEO, OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
