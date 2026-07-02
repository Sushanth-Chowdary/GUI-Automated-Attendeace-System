import os
import cv2
import threading
import queue
import time
import datetime
import requests
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
import multiprocessing as mp

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

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def format_timestamp(frame_count, fps):
    fps = fps if fps > 0 else 30
    total_seconds = int(frame_count // fps)
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

# ==========================================
# MULTIPROCESSING INFERENCE PROCESS (THROTTLED CPU)
# ==========================================
def inference_worker(inf_queue, ann_queue, cmd_queue, timestamp_str, cam_name):
    """
    Evaluates ML on throttled inbound frames inherently preventing deadlocks.
    Writes native annotations directly against duplicate outputs maintaining 1-1 strict bounds flawlessly.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[ML PROCESS | {cam_name}] Initializing strictly on {device}...")
    
    try:
        yolo_model = YOLO('yolov8n-face.pt', task='detect')
        yolo_model.to(device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
        faiss_index_path = './face_attendance_faiss.bin'
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            index.nprobe = 20
        else:
            index = None
            
        meta_path = './face_attendance_meta.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                saved_data = pickle.load(f)
            target_names = saved_data['target_names']
            y_real = saved_data['y_real']
        else:
            target_names, y_real = [], []
            
        to_tensor = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor()])
    except Exception as e:
        print(f"[ML PROCESS ERROR | {cam_name}] Failed to load models: {e}")
        return

    fps = 30
    frame_count = 0
    active_track_memory = {}
    archived_tracks = {}
    track_identities = {}

    print(f"[ML PROCESS | {cam_name}] Ready and listening for inbound throttled streams...")

    while True:
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == 'STOP': break
        except queue.Empty: pass

        try:
            item = inf_queue.get(timeout=0.2)
        except queue.Empty:
            continue
            
        frame, input_fps, skipped_frames = item
        
        frame_count += (skipped_frames + 1)
        fps = input_fps if input_fps > 0 else 30
        
        # Deadlock Elimination: Frame is drawn explicitly without sync-queue loops globally limiting logic
        display_frame = frame.copy()
        current_active_faces = 0

        if index is not None and len(target_names) > 0:
            results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            has_detections = results[0].boxes.id is not None
            
            if has_detections:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                current_active_faces = len(ids)
                
                for t_id in ids:
                    if t_id not in active_track_memory:
                        active_track_memory[t_id] = {
                            'start_time': format_timestamp(frame_count, fps),
                            'frames_alive': 0, 'buffer': [], 'all_preds': [], 'missing_frames': 0
                        }
                    active_track_memory[t_id]['frames_alive'] += (skipped_frames + 1)

                if frame_count % FRAME_SKIP == 0 or (frame_count - skipped_frames) % FRAME_SKIP == 0:
                    batch_tensors, batch_track_ids = [], []
                    for i, t_id in enumerate(ids):
                        crop = crop_standard(frame, boxes[i])
                        if crop.size > 0 and cv2.Laplacian(crop, cv2.CV_64F).var() > 4.0:
                            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            t = (to_tensor(pil_img).unsqueeze(0).to(device) - 0.5) * 2
                            batch_tensors.append(t)
                            batch_track_ids.append(t_id)
                    
                    if batch_tensors:
                        with torch.no_grad():
                            embeddings = resnet(torch.cat(batch_tensors, dim=0)).cpu().numpy().astype('float32')
                        faiss.normalize_L2(embeddings)
                        sims, indices = index.search(embeddings, k=1)
                        for i, t_id in enumerate(batch_track_ids):
                            name = target_names[y_real[indices[i][0]]] if sims[i][0] > CONFIDENCE_THRESHOLD else "Unknown"
                            active_track_memory[t_id]['buffer'].append(name)
                            active_track_memory[t_id]['all_preds'].append(name)
                            
                            if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                valid_history = [v for v in active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                winner = Counter(valid_history).most_common(1)[0][0] if valid_history else "Unknown"
                                track_identities[t_id] = winner
                                active_track_memory[t_id]['buffer'] = []

                # Draw Visual Output Layers Directly Inside GPU Map Boundaries preventing deadlocks
                for i in range(len(ids)):
                    t_id = ids[i]
                    box = boxes[i]
                    name = track_identities.get(t_id, "Analyzing...")
                    color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                    cv2.rectangle(display_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(display_frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                alive_ids = set(ids)
                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        active_track_memory[t_id]['missing_frames'] += (skipped_frames + 1)
                        if active_track_memory[t_id]['missing_frames'] > 15:
                            archived_tracks[t_id] = active_track_memory.pop(t_id)
                    else:
                        active_track_memory[t_id]['missing_frames'] = 0

        # Enforce highly reliable logic outputs directly to single bound queue strictly avoiding cross queue ID locks!
        if not ann_queue.full():
            ann_queue.put_nowait((display_frame, skipped_frames, current_active_faces, fps))

    # ==========================
    # FINAL ATTENDANCE SECURE DUMP
    # ==========================
    print(f"[ML PROCESS | {cam_name}] Shutting down natively... Generating Log Checkboxes.")
    final_mem = {**archived_tracks, **active_track_memory}
    debug_data = []
    student_presence = {name: False for name in target_names}
    student_detection_count = {name: 0 for name in target_names}

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
            status = "Passed" if (total_frames >= 45 and total_samples >= 15 and sample_ratio >= 0.25 and win_ratio >= 0.60) else "Failed"
        else:
            winner = "Unknown"
            status = "Failed"
            counts = Counter(all_preds)

        if status == "Passed" and winner != "Unknown":
            student_presence[winner] = True
            student_detection_count[winner] += counts.get(winner, 0)

        debug_data.append({
            'Track ID': t_id, 'Start Time': data.get('start_time', ''),
            'Total Frames': total_frames, 'Valid Votes': valid_votes_count,
            'Total Preds (inc. Unknown)': len(all_preds), 'Predicted Identity': winner,
            'Gate Status': status, 'Breakdown': dict(Counter(all_preds))
        })

    pd.DataFrame(debug_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_{cam_name}_DEBUG_Tracks.csv"))
    output_data = [{'Name': s, 'Status': 'Present' if student_presence[s] else 'Absent', 'Detection Count': student_detection_count[s]} for s in target_names]
    pd.DataFrame(output_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_{cam_name}_output.csv"), index=False)


# ==========================================
# BACKGROUND OUPUT AND INGESTION THREADS
# ==========================================
def rstp_reader(ip, running_event, raw_queue, inf_queue):
    """ 
    Absolute Watchdog. 
    Implements CPU optimization via Input Throttling caching explicit dropped frames natively. 
    """
    url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/stream1"
    cap = None
    
    skipped_frames = 0
    target_fps = 10.0
    throttle_interval = 1.0 / target_fps
    last_dispatch_time = 0
    
    while running_event.is_set():
        if cap is None or not cap.isOpened():
            print(f"[*] Native Reader Booting IP {ip}...")
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                time.sleep(5)
                continue
            
        ret, frame = cap.read()
        if not ret:
            cap.release(); cap = None; continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0: fps = 30
        
        # Raw Writer explicitly gets all original non-throttled data
        if not raw_queue.full():
            raw_queue.put_nowait((frame, fps, 1))
            
        # Throttled Dispatch evaluating constraints preventing CPU Serialization overload arrays entirely
        now = time.perf_counter()
        if (now - last_dispatch_time) >= throttle_interval and not inf_queue.full():
            # Send cleanly directly towards processing layer explicitly counting dropped logic 
            inf_queue.put_nowait((frame, fps, skipped_frames))
            last_dispatch_time = now
            skipped_frames = 0
        else:
            skipped_frames += 1
                
    if cap: cap.release()

def raw_writer_worker(queue_obj, output_path, running_event):
    writer = None
    while True:
        try:
            item = queue_obj.get(timeout=0.2)
            if item is None: break
            frame, input_fps, duplicates = item
            
            if writer is None:
                h, w = frame.shape[:2]
                fps = input_fps if input_fps > 0 else 30
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
            for _ in range(duplicates):
                writer.write(frame)
        except queue.Empty:
            if not running_event.is_set() and queue_obj.empty(): break
    if writer: writer.release()

def ann_writer_worker(ann_queue, ui_queue, output_path, running_event):
    """
    Highly robust generic bounds mapping directly towards duplicating skipped loops.
    Strictly deadlock-free! 
    """
    writer = None
    while True:
        try:
            item = ann_queue.get(timeout=0.2)
            if item is None: break
            frame, skipped_frames, active_faces, input_fps = item
                
            if writer is None:
                h, w = frame.shape[:2]
                fps = input_fps if input_fps > 0 else 30
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
            # EXECUTE FRAME PADDING AVOIDING SPED UP PHENOMENON unconditionally avoiding dropped timing errors natively
            for _ in range(skipped_frames + 1):
                writer.write(frame)
                
            if not ui_queue.full():
                ui_queue.put_nowait((frame, active_faces))
                
        except queue.Empty:
            if not running_event.is_set() and ann_queue.empty(): break
            
    if writer: writer.release()


# ==========================================
# MAIN DESKTOP GUI
# ==========================================
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Dual-Process Optimized Architecture")
        self.geometry("1400x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.running_event = mp.Event()
        self.selected_protocol = ctk.IntVar(value=1)
        self.view_target = ctk.StringVar(value="Camera 1") 
        
        self.t_end = 0
        self.sys_fps = 0
        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.cam1_active_faces = 0
        self.cam2_active_faces = 0

        self.setup_ui()

    def setup_ui(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Attendance Core ML", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)
        
        self.view_label = ctk.CTkLabel(self.sidebar_frame, text="Active Telemetry View:", font=ctk.CTkFont(size=14, underline=True))
        self.view_label.pack(pady=(10,0), padx=20, anchor="w")
        self.view_toggle = ctk.CTkSegmentedButton(self.sidebar_frame, values=["Camera 1", "Camera 2"], variable=self.view_target)
        self.view_toggle.pack(pady=10, padx=20, fill="x")
        
        self.protocol_label = ctk.CTkLabel(self.sidebar_frame, text="Execution Iteration:")
        self.protocol_label.pack(pady=(15,0), padx=20, anchor="w")
        for p, info in PROTOCOLS.items():
            rb = ctk.CTkRadioButton(self.sidebar_frame, text=f"{info['name']} Protocol", variable=self.selected_protocol, value=p)
            rb.pack(pady=5, padx=20, anchor="w")
            
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Execute Bound Loop", command=self.start_tracking)
        self.start_btn.pack(pady=20, padx=20)
        
        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Force Stop System", command=self.stop_tracking, fg_color="red", hover_color="darkred")
        self.stop_btn.pack(pady=10, padx=20)
        self.stop_btn.configure(state="disabled")
        
        self.telemetry_lbl = ctk.CTkLabel(self.sidebar_frame, text="Live Telemetry Overlay", font=ctk.CTkFont(size=18, weight="bold"))
        self.telemetry_lbl.pack(pady=(40,10), padx=20, anchor="w")

        self.time_lbl = ctk.CTkLabel(self.sidebar_frame, text="Remaining Timer: 00:00:00", font=ctk.CTkFont(size=14))
        self.time_lbl.pack(pady=5, padx=20, anchor="w")
        self.fps_lbl = ctk.CTkLabel(self.sidebar_frame, text="System Render FPS: 0.0", font=ctk.CTkFont(size=14))
        self.fps_lbl.pack(pady=5, padx=20, anchor="w")
        self.faces_lbl = ctk.CTkLabel(self.sidebar_frame, text="Target Faces Detected: 0", font=ctk.CTkFont(size=14))
        self.faces_lbl.pack(pady=5, padx=20, anchor="w")

        self.video_frame = ctk.CTkLabel(self, text="Camera ML Feed Offline", bg_color="gray", width=950, height=750)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def start_tracking(self):
        if self.running_event.is_set(): return
            
        self.running_event.set()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        sel = self.selected_protocol.get()
        p_info = PROTOCOLS[sel]
        
        total_time_cam1 = len(CAM1_PRESETS) * (p_info['cam1'] + 5)
        total_time_cam2 = len(CAM2_PRESETS) * (p_info['cam2'] + 5)
        self.t_end = time.perf_counter() + max(total_time_cam1, total_time_cam2)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.cam1_inf_q = mp.Queue(maxsize=30)
        self.cam1_ann_q = mp.Queue(maxsize=150)
        self.cam1_cmd_q = mp.Queue()
        
        self.cam2_inf_q = mp.Queue(maxsize=30)
        self.cam2_ann_q = mp.Queue(maxsize=150)
        self.cam2_cmd_q = mp.Queue()
        
        self.cam1_raw_q = queue.Queue(maxsize=60)
        self.cam1_ui_q = queue.Queue(maxsize=30)
        
        self.cam2_raw_q = queue.Queue(maxsize=60)
        self.cam2_ui_q = queue.Queue(maxsize=30)

        # 1. Fire Multiprocessing Dual Inference Core Threads safely bound
        self.ml_p1 = mp.Process(target=inference_worker, args=(self.cam1_inf_q, self.cam1_ann_q, self.cam1_cmd_q, timestamp_str, "Cam1"))
        self.ml_p2 = mp.Process(target=inference_worker, args=(self.cam2_inf_q, self.cam2_ann_q, self.cam2_cmd_q, timestamp_str, "Cam2"))
        self.ml_p1.daemon = True; self.ml_p1.start()
        self.ml_p2.daemon = True; self.ml_p2.start()

        # 2. Fire Independent Sync Disconnected Writers 
        self.threads = [
            threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event)),
            threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event)),
            threading.Thread(target=ann_writer_worker, args=(self.cam1_ann_q, self.cam1_ui_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam1_annotated.mp4"), self.running_event)),
            threading.Thread(target=ann_writer_worker, args=(self.cam2_ann_q, self.cam2_ui_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam2_annotated.mp4"), self.running_event)),
            threading.Thread(target=rstp_reader, args=(CAMERA_IP_1, self.running_event, self.cam1_raw_q, self.cam1_inf_q)),
            threading.Thread(target=rstp_reader, args=(CAMERA_IP_2, self.running_event, self.cam2_raw_q, self.cam2_inf_q)),
            threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_1, CAM1_PRESETS, p_info['cam1'])),
            threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_2, CAM2_PRESETS, p_info['cam2']))
        ]
        for t in self.threads: t.start()

        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.update_gui_frame()

    def stop_tracking(self):
        if not self.running_event.is_set(): return
        self.running_event.clear()
        
        self.cam1_cmd_q.put('STOP')
        self.cam2_cmd_q.put('STOP')
        
        self.cam1_ann_q.put(None); self.cam2_ann_q.put(None)
        self.cam1_raw_q.put(None); self.cam2_raw_q.put(None)
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def ptz_runner(self, ip, presets, duration):
        for preset in presets:
            if not self.running_event.is_set(): break
            url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset}"
            try: requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
            except: pass
            
            t_mechanical = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_mechanical) < 5.0:
                time.sleep(0.1)
                
            t_start = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_start) < duration:
                time.sleep(0.1)

    def update_gui_frame(self):
        if not self.running_event.is_set():
            self.video_frame.configure(image=None, text="Camera ML Feed Offline")
            return
            
        try:
            now = time.perf_counter()
            remaining = max(0, int(self.t_end - now))
            
            if remaining <= 0:
                self.stop_tracking()
                return
                
            h, rem = divmod(remaining, 3600)
            m, s = divmod(rem, 60)
            self.time_lbl.configure(text=f"Time Remaining: {h:02d}:{m:02d}:{s:02d}")
            
            frame1 = None
            while not self.cam1_ui_q.empty():
                try: 
                    res1 = self.cam1_ui_q.get_nowait()
                    frame1, self.cam1_active_faces = res1
                except queue.Empty: break
                    
            frame2 = None
            while not self.cam2_ui_q.empty():
                try: 
                    res2 = self.cam2_ui_q.get_nowait()
                    frame2, self.cam2_active_faces = res2
                except queue.Empty: break
            
            if (now - self.last_fps_time) >= 1.0:
                self.sys_fps = self.frames_rendered / (now - self.last_fps_time)
                self.fps_lbl.configure(text=f"System Render FPS: {self.sys_fps:.1f}")
                disp_faces = self.cam1_active_faces if self.view_target.get() == "Camera 1" else self.cam2_active_faces
                self.faces_lbl.configure(text=f"Target Faces Detected: {disp_faces}")
                self.frames_rendered = 0
                self.last_fps_time = now

            target_frame = frame1 if self.view_target.get() == "Camera 1" else frame2
            if target_frame is not None:
                self.frames_rendered += 1
                frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(1000, 780))
                self.video_frame.configure(image=imgtk, text="")
                self.video_frame.image = imgtk
                
        except Exception as e: pass
        finally: self.after(33, self.update_gui_frame)

if __name__ == "__main__":
    mp.freeze_support()
    app = AttendanceApp()
    app.mainloop()
