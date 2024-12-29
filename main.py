from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import warnings
from threading import Thread
from dataclasses import dataclass
from typing import List
warnings.filterwarnings('ignore')

@dataclass
class Segment:
    start_idx: int
    end_idx: int
    frames: List[np.ndarray]

class FrameGrabber:
    def __init__(self):
        self.cap = None
        self.grabbed_frame = None
        self.running = False
        self.thread = None
        self.actual_fps = 0
    
    def start(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        self.actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"Camera FPS: {self.actual_fps}")
        
        self.running = True
        self.thread = Thread(target=self._grab_frames, daemon=True)
        self.thread.start()
        
    def _grab_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.grabbed_frame = frame.copy()
            time.sleep(1/self.actual_fps)
            
    def get_frame(self):
        return self.grabbed_frame.copy() if self.grabbed_frame is not None else None
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()

class HumanDetector:
    def __init__(self):
        print("Initializing AI model...")
        self.setup_model()
        self.setup_visual_params()
        
    def setup_model(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using {self.device} for AI processing")
        
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)
        self.model.conf = 0.35
        self.model.iou = 0.45
        
        if self.device == 'cuda':
            self.model.fuse()
        
        print("AI model ready")
        
    def setup_visual_params(self):
        self.box_color = (75, 199, 255)
        self.text_color = (255, 255, 255)
        self.box_thickness = 2
        self.text_size = 0.7
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def process_frame(self, frame):
        if frame is None:
            return frame, False
            
        results = self.model(frame, classes=0, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            if box.conf[0] > self.model.conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                            self.box_color, self.box_thickness)
                
                label = f"Person {conf:.1%}"
                (label_w, label_h), _ = cv2.getTextSize(label, self.font, 
                                                      self.text_size, 1)
                cv2.rectangle(frame, 
                            (x1, y1 - label_h - 10),
                            (x1 + label_w, y1),
                            self.box_color, -1)
                
                cv2.putText(frame, label,
                          (x1, y1 - 5),
                          self.font,
                          self.text_size,
                          self.text_color, 2)
                
                detections.append((x1, y1, x2, y2, conf))
        
        return frame, len(detections) > 0

class VideoProcessor:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path("recordings") / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRecordings will be saved to: {self.output_dir}")
        
        self.detector = HumanDetector()
        self.frame_times = []
        
        self.original_frames = []
        self.human_segments = []
        self.current_segment = None
        
    def calculate_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        while self.frame_times and current_time - self.frame_times[0] > 1.0:
            self.frame_times.pop(0)
            
        return len(self.frame_times)

    def add_to_segments(self, frame_idx, processed_frame, has_humans):
        if has_humans:
            if self.current_segment is None:
                self.current_segment = Segment(frame_idx, frame_idx, [processed_frame])
            else:
                self.current_segment.end_idx = frame_idx
                self.current_segment.frames.append(processed_frame)
        elif self.current_segment is not None:
            if len(self.current_segment.frames) > 5:  # 최소 5프레임 이상일 때만 저장
                self.human_segments.append(self.current_segment)
            self.current_segment = None

    def save_videos(self, fps):
        if not self.original_frames:
            print("No frames to save")
            return
            
        print("\nSaving recordings...")
        
        height, width = self.original_frames[0].shape[:2]
        
        # 원본 비디오 저장
        raw_path = self.output_dir / "original.mp4"
        raw_writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        for frame in self.original_frames:
            raw_writer.write(frame)
        raw_writer.release()
        
        # 처리된 비디오 저장 (사람이 감지된 부분만)
        if self.human_segments:
            processed_path = self.output_dir / "detected.mp4"
            processed_writer = cv2.VideoWriter(
                str(processed_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
            # 각 세그먼트의 프레임들을 저장
            for segment in self.human_segments:
                for frame in segment.frames:
                    processed_writer.write(frame)
            
            processed_writer.release()
            
            print(f"Original video saved: {raw_path}")
            print(f"Processed video (humans only) saved: {processed_path}")
            print(f"Found {len(self.human_segments)} segments with humans")
        else:
            print("No humans detected in the video")

    def run(self):
        print("\n=== AI Human Detection System ===")
        print("Recording automatically...")
        print("Press 'Q' to stop and save")
        print("===============================\n")

        frame_grabber = FrameGrabber()
        frame_grabber.start()
        
        try:
            frame_idx = 0
            while True:
                original_frame = frame_grabber.get_frame()
                if original_frame is None:
                    continue

                self.original_frames.append(original_frame.copy())
                
                # AI 처리
                processed_frame, has_humans = self.detector.process_frame(original_frame.copy())
                
                # 세그먼트 관리
                self.add_to_segments(frame_idx, processed_frame, has_humans)
                
                # FPS 계산 및 상태 표시
                fps = self.calculate_fps()
                status = "Human Detected" if has_humans else "Monitoring"
                status_color = (0, 255, 0) if has_humans else (255, 255, 255)
                
                cv2.putText(processed_frame, 
                           f"FPS: {fps} | {status}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           status_color, 
                           2)
                
                # 화면 표시
                cv2.imshow('AI Human Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                frame_idx += 1

        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            
        finally:
            # 마지막 세그먼트 처리
            if self.current_segment and len(self.current_segment.frames) > 5:
                self.human_segments.append(self.current_segment)
            
            print("\nSaving recordings...")
            self.save_videos(frame_grabber.actual_fps)
            frame_grabber.stop()
            cv2.destroyAllWindows()
            print("All recordings saved successfully")

def main():
    try:
        processor = VideoProcessor()
        processor.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        print("Program terminated")

if __name__ == "__main__":
    import torch
    main()