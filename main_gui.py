import os
import cv2
import threading
import queue
import time
import datetime
import requests
import subprocess
import pandas as pd
from collections import Counter
import customtkinter as ctk
from PIL import Image, ImageTk
import faiss
import pickle
import torch
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from requests.auth import HTTPBasicAuth

# ==========================================
# CONFIGURATION
# ==========================================
USERNAME = "admin"
PASSWORD = "admin"
CAMERA_IP_1 = "10.34.0.17"
CAMERA_IP_2 = "10.34.0.16"

CAM1_PRESETS = [1, 2, 3, 4, 5, 6, 7, 8]
CAM2_PRESETS = [1, 2, 3, 4, 5, 6]

PROTOCOLS = {
    1: {"name": "5-Minute", "cam1": 32, "cam2": 45},
    2: {"name": "15-Minute", "cam1": 107, "cam2": 145},
    3: {"name": "45-Minute", "cam1": 332, "cam2": 445}
}

CONFIDENCE_THRESHOLD = 0.78
FRAME_SKIP = 2
FRAMES_PER_VOTE = 5

BASE_OUTPUT_DIR = "VIDEOS"
RESULTS_DIR = os.path.abspath('ATTENDENCE RESULTS/MINE')

# Setup Guardrails
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize Device & Models globally or within a manager
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def format_timestamp(frame_count, fps):
    fps = fps if fps > 0 else 30
    total_seconds = frame_count // fps
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def crop_standard(img, box):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img.shape[1], x2 + margin_x)
    y2 = min(img.shape[0], y2 + margin_y)
    return img[y1:y2, x1:x2]

class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time YOLO Tracking & PTZ Control")
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.is_running = False
        self.selected_protocol = ctk.IntVar(value=1)
        self.frame_queue = queue.Queue(maxsize=10)

        self.setup_ui()
        self.load_models()

    def load_models(self):
        try:
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            self.yolo_model = YOLO('yolov8n-face.pt', task='detect')
            
            faiss_index_path = './face_attendance_faiss.bin'
            if os.path.exists(faiss_index_path):
                self.index = faiss.read_index(faiss_index_path)
                self.index.nprobe = 20
            else:
                print("[-] FAISS Index missing.")
                self.index = None
                
            meta_path = './face_attendance_meta.pkl'
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    saved_data = pickle.load(f)
                self.target_names = saved_data['target_names']
                self.y_real = saved_data['y_real']
            else:
                print("[-] Meta pickle missing.")
                self.target_names, self.y_real = [], []
                
            self.to_tensor = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
        except Exception as e:
            print(f"[-] Error loading models: {e}")

    def setup_ui(self):
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Attendance System", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)
        
        self.protocol_label = ctk.CTkLabel(self.sidebar_frame, text="Select Protocol:")
        self.protocol_label.pack(pady=10, padx=20, anchor="w")
        
        for p, info in PROTOCOLS.items():
            rb = ctk.CTkRadioButton(self.sidebar_frame, text=info['name'], variable=self.selected_protocol, value=p)
            rb.pack(pady=5, padx=20, anchor="w")
            
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Start Real-Time Tracking", command=self.start_tracking)
        self.start_btn.pack(pady=20, padx=20)
        
        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Stop/End Protocol", command=self.stop_tracking, fg_color="red", hover_color="darkred")
        self.stop_btn.pack(pady=10, padx=20)
        self.stop_btn.configure(state="disabled")

        # Main video frame
        self.video_frame = ctk.CTkLabel(self, text="Camera Feed Offline", bg_color="gray", width=900, height=700)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def start_tracking(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        sel = self.selected_protocol.get()
        protocol_info = PROTOCOLS[sel]
        print(f"Starting Protocol {sel}: {protocol_info['name']}")
        
        # We process Camera 1 in the main RTSP/YOLO pipeline
        self.rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP_1}:554/stream1"
        self.timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.active_track_memory = {}
        self.archived_tracks = {}
        self.track_identities = {}
        
        self.video_thread = threading.Thread(target=self.process_video_stream, daemon=True)
        self.video_thread.start()
        
        self.ptz_thread_1 = threading.Thread(target=self.run_camera_1, args=(protocol_info['cam1'],), daemon=True)
        self.ptz_thread_2 = threading.Thread(target=self.run_camera_2, args=(protocol_info['cam2'],), daemon=True)
        
        self.ptz_thread_1.start()
        self.ptz_thread_2.start()
        
        self.update_gui_frame()

    def stop_tracking(self):
        if not self.is_running:
            return
        print("[*] Stopping Protocol...")
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def call_preset(self, ip, preset_number):
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset_number}"
        try:
            r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
            if r.status_code != 200:
                print(f"[-] {ip}: Failed preset {preset_number}")
        except Exception as e:
            print(f"[-] {ip}: Error connecting {e}")

    def run_camera_1(self, duration):
        for preset in CAM1_PRESETS:
            if not self.is_running: break
            print(f"[*] Camera 1 Preset {preset}")
            self.call_preset(CAMERA_IP_1, preset)
            time.sleep(5)
            # Sleep logic to simulate the duration logic of ffmpeg
            t0 = time.time()
            while self.is_running and (time.time() - t0) < duration:
                time.sleep(1)

    def run_camera_2(self, duration):
        for preset in CAM2_PRESETS:
            if not self.is_running: break
            print(f"[*] Camera 2 Preset {preset}")
            self.call_preset(CAMERA_IP_2, preset)
            time.sleep(5)
            t0 = time.time()
            while self.is_running and (time.time() - t0) < duration:
                time.sleep(1)

    def process_video_stream(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print("[-] Unable to open RTSP stream.")
            self.is_running = False
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0: fps = 25 
        
        raw_output_path = os.path.join(BASE_OUTPUT_DIR, f"{self.timestamp_str}_raw.mp4")
        ann_output_path = os.path.join(RESULTS_DIR, f"{self.timestamp_str}_annotated.mp4")
        
        writer_raw = cv2.VideoWriter(raw_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer_ann = cv2.VideoWriter(ann_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_count = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # 1. Write the raw frame
            writer_raw.write(frame)
            
            display_frame = frame.copy()
            
            # 2. YOLO processing (skip logic could be applied here if needed)
            if self.index is not None and len(self.target_names) > 0:
                results = self.yolo_model.track(display_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                has_detections = results[0].boxes.id is not None
                
                if has_detections:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.int().cpu().numpy()
                    
                    for t_id in ids:
                        if t_id not in self.active_track_memory:
                            self.active_track_memory[t_id] = {
                                'start_time': format_timestamp(frame_count, fps),
                                'frames_alive': 0, 'buffer': [], 'all_preds': [], 'missing_frames': 0
                            }
                        self.active_track_memory[t_id]['frames_alive'] += 1

                    if frame_count % FRAME_SKIP == 0:
                        batch_tensors, batch_track_ids = [], []
                        for i, t_id in enumerate(ids):
                            crop = crop_standard(frame, boxes[i])
                            if crop.size > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() > 4.0:
                                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                t = (self.to_tensor(pil_img).unsqueeze(0).to(device) - 0.5) * 2
                                batch_tensors.append(t)
                                batch_track_ids.append(t_id)
                        
                        if batch_tensors:
                            with torch.no_grad():
                                embeddings = self.resnet(torch.cat(batch_tensors, dim=0)).cpu().numpy().astype('float32')
                            faiss.normalize_L2(embeddings)
                            sims, indices = self.index.search(embeddings, k=1)
                            for i, t_id in enumerate(batch_track_ids):
                                name = self.target_names[self.y_real[indices[i][0]]] if sims[i][0] > CONFIDENCE_THRESHOLD else "Unknown"
                                self.active_track_memory[t_id]['buffer'].append(name)
                                self.active_track_memory[t_id]['all_preds'].append(name)
                                
                                if len(self.active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                    valid_history = [v for v in self.active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                    winner = Counter(valid_history).most_common(1)[0][0] if valid_history else "Unknown"
                                    self.track_identities[t_id] = winner
                                    self.active_track_memory[t_id]['buffer'] = []
                                
                    # Draw boxes on display frame
                    for i in range(len(ids)):
                        t_id = ids[i]
                        box = boxes[i]
                        name = self.track_identities.get(t_id, "Analyzing...")
                        color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                        cv2.rectangle(display_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                        cv2.putText(display_frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                    # Memory cleanup
                    alive_ids = set(ids)
                    for t_id in list(self.active_track_memory.keys()):
                        if t_id not in alive_ids:
                            self.active_track_memory[t_id]['missing_frames'] += 1
                            if self.active_track_memory[t_id]['missing_frames'] > 15:
                                self.archived_tracks[t_id] = self.active_track_memory.pop(t_id)
                        else:
                            self.active_track_memory[t_id]['missing_frames'] = 0

            # 3. Write annotated frame
            writer_ann.write(display_frame)
            
            # 4. Push to GUI
            try:
                # Limit queue size to avoid staleness
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(display_frame)
            except queue.Full:
                pass
                
            frame_count += 1
            
        print("[*] Releasing resources and generating attendance logs...")
        cap.release()
        writer_raw.release()
        writer_ann.release()
        
        self.save_attendance_results(self.timestamp_str)
        print("[+] Finished saving. Returning to idle state.")

    def update_gui_frame(self):
        if not self.is_running:
            self.video_frame.configure(image=None, text="Camera Feed Offline")
            return
            
        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((900, 700))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(900, 700))
            self.video_frame.configure(image=imgtk, text="")
            self.video_frame.image = imgtk
        except queue.Empty:
            pass
            
        self.after(30, self.update_gui_frame)

    def save_attendance_results(self, timestamp_str):
        final_mem = {**self.archived_tracks, **self.active_track_memory}
        debug_data = []
        student_presence = {name: False for name in self.target_names}
        student_detection_count = {name: 0 for name in self.target_names}

        for t_id, data in final_mem.items():
            total_frames = data.get('frames_alive', 0)
            all_preds = data['all_preds']
            valid_preds = [p for p in all_preds if p != "Unknown"]
            valid_votes_count = len(valid_preds)
            
            if valid_votes_count > 0:
                counts = Counter(valid_preds)
                winner = counts.most_common(1)[0][0]
                win_ratio = counts.get(winner, 0) / valid_votes_count
                total_samples = len(all_preds)
                sample_ratio = counts.get(winner, 0) / total_samples if total_samples > 0 else 0
                
                status = "Passed" if (total_frames >= 45 and 
                                      total_samples >= 15 and 
                                      sample_ratio >= 0.25 and 
                                      win_ratio >= 0.60) else "Failed"
            else:
                winner = "Unknown"
                status = "Failed"
                counts = Counter(all_preds) 

            if status == "Passed" and winner != "Unknown":
                student_presence[winner] = True
                student_detection_count[winner] += counts.get(winner, 0)

            debug_data.append({
                'Track ID': t_id,
                'Start Time': data.get('start_time', ''),
                'Total Frames': total_frames,
                'Valid Votes': valid_votes_count,
                'Total Preds (inc. Unknown)': len(all_preds),
                'Predicted Identity': winner,
                'Gate Status': status,
                'Breakdown': dict(Counter(all_preds))
            })

        pd.DataFrame(debug_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_DEBUG_Tracks.csv"))
        output_data = [{'Name': s, 'Status': 'Present' if student_presence[s] else 'Absent', 'Detection Count': student_detection_count[s]} for s in self.target_names]
        pd.DataFrame(output_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_output.csv"), index=False)

if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()
