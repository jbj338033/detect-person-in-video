from ultralytics import YOLO
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
import torch
from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass
class Segment:
    start: float
    end: float

class VideoProcessor:
    def __init__(self, min_duration: float = 1.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt')
        self.min_duration = min_duration
        self._setup_model()

    def _setup_model(self):
        self.model.to(self.device)
        self.model.conf = 0.5
        
    def detect_humans(self, frame: np.ndarray) -> bool:
        results = self.model(frame, classes=0)
        return len(results[0].boxes) > 0

    def process_video(self, input_path: str, output_path: str):
        video = VideoFileClip(input_path)
        fps = video.fps
        total_frames = int(video.duration * fps)
        
        segments: List[Segment] = []
        start_time = None
        human_visible = False

        for t in range(0, total_frames, 2):
            current_time = t / fps
            frame = video.get_frame(current_time)
            
            if self.detect_humans(frame):
                if not human_visible:
                    start_time = current_time
                    human_visible = True
            elif human_visible:
                if start_time is not None:
                    duration = current_time - start_time
                    if duration >= self.min_duration:
                        segments.append(Segment(start_time, current_time))
                human_visible = False
                start_time = None

        if human_visible and start_time is not None:
            duration = video.duration - start_time
            if duration >= self.min_duration:
                segments.append(Segment(start_time, video.duration))

        self._create_output(video, segments, output_path)
        video.close()

    def _create_output(self, video, segments, output_path):
        if not segments:
            return
            
        clips = [video.subclip(seg.start, seg.end) for seg in segments]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        final_clip.close()

def main():
    INPUT_VIDEO = "input.mp4"
    OUTPUT_VIDEO = "output.mp4"
    
    processor = VideoProcessor(min_duration=1.0)
    processor.process_video(INPUT_VIDEO, OUTPUT_VIDEO)

if __name__ == "__main__":
    main()