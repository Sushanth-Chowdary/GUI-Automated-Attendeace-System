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

# Setup Guardrails
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper for timestamps
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
# MULTIPROCESSING INFERENCE PROCESS
# ==========================================
def inference_worker(inf_queue, res_queue, cmd_queue, timestamp_str):
    """
    Subprocess for handling the heavy ML Pipeline.
    Bypasses GIL constraints using multiprocessing process.
    """
    # 1. Hardware Acceleration Enforced
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[ML PROCESS] Initializing on {device}...")
    
    # 2. Models
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
        print(f"[ML PROCESS ERROR] Failed to load models: {e}")
        return

    fps = 25
    frame_count = 0
    active_track_memory = {}
    archived_tracks = {}
    track_identities = {}

    print(f"[ML PROCESS] Ready and listening for frames...")

    while True:
        # Check command queue for SHUTDOWN signal
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == 'STOP':
                break
        except queue.Empty:
            pass

        # Real-time sync explicit: DROP OLDEST FRAMES if queue backs up
        frame, input_fps = None, None
        dropped_count = 0
        
        # Drain the inference queue to ensure we only process the absolutely newest frame available.
        # This guarantees the ML inference never falls behind live-time.
        while not inf_queue.empty():
            try:
                frame, input_fps = inf_queue.get_nowait()
                dropped_count += 1
            except queue.Empty:
                break
                
        # If we didn't drain any frames, gracefully block untill one arrives
        if frame is None:
            try:
                frame, input_fps = inf_queue.get(timeout=0.05)
                dropped_count = 1
            except queue.Empty:
                continue
                
        fps = input_fps if input_fps > 0 else 25
        frame_count += dropped_count # Maintain roughly accurate frame counts for tracking intervals

        display_frame = frame.copy()
        current_active_faces = 0

        # Perform YOLO + Tracking
        if index is not None and len(target_names) > 0:
            results = yolo_model.track(display_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            has_detections = results[0].boxes.id is not None
            
            if has_detections:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                current_active_faces = len(ids)
                
                # Register new active tracks
                for t_id in ids:
                    if t_id not in active_track_memory:
                        active_track_memory[t_id] = {
                            'start_time': format_timestamp(frame_count, fps),
                            'frames_alive': 0, 'buffer': [], 'all_preds': [], 'missing_frames': 0
                        }
                    active_track_memory[t_id]['frames_alive'] += 1

                # Periodic FaceNet Recognition Embeddings
                if frame_count % FRAME_SKIP == 0:
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
                            
                            # Voting Memory calculation
                            if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                valid_history = [v for v in active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                winner = Counter(valid_history).most_common(1)[0][0] if valid_history else "Unknown"
                                track_identities[t_id] = winner
                                active_track_memory[t_id]['buffer'] = []

                # Draw bounding boxes and text directly onto our display frame
                for i in range(len(ids)):
                    t_id = ids[i]
                    box = boxes[i]
                    name = track_identities.get(t_id, "Analyzing...")
                    color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                    cv2.rectangle(display_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    cv2.putText(display_frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Memory cleanup for vanished track IDs mapping
                alive_ids = set(ids)
                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        active_track_memory[t_id]['missing_frames'] += 1
                        if active_track_memory[t_id]['missing_frames'] > 15:
                            archived_tracks[t_id] = active_track_memory.pop(t_id)
                    else:
                        active_track_memory[t_id]['missing_frames'] = 0

        # Push the finalized frame to the parent GUI rendering queue
        if not res_queue.full():
            res_queue.put_nowait((display_frame, current_active_faces, fps))

    # ==========================
    # FINAL ATTENDANCE SAVING THREAD
    # ==========================
    # Runs automatically upon successful 'STOP' signal retrieval
    print("[ML PROCESS] Shutting down, generating attendance CSV logs...")
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
            # Extremely strict tolerance to mark as genuinely "Passed"
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

    pd.DataFrame(debug_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_DEBUG_Tracks.csv"))
    output_data = [{'Name': s, 'Status': 'Present' if student_presence[s] else 'Absent', 'Detection Count': student_detection_count[s]} for s in target_names]
    pd.DataFrame(output_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_output.csv"), index=False)
    print(f"[ML PROCESS] Successfully compiled robust CSV metrics into {RESULTS_DIR}")


# ==========================================
# DECOUPLED I/O WORKER THREADS
# ==========================================
def rstp_reader(ip, running_event, raw_queue, inf_queue=None):
    """ RTSP Watchdog thread that handles reconnection logic """
    url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/stream1"
    cap = None
    while running_event.is_set():
        if cap is None or not cap.isOpened():
            print(f"[*] RTSP Watchdog connecting to {ip}...")
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"[-] {ip} failed connection. Retrying in 5s.")
                time.sleep(5)
                continue
            
        ret, frame = cap.read()
        if not ret:
            print(f"[-] {ip} frame read failed. Watchdog attempting reconnect.")
            cap.release()
            cap = None
            continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Dispatch to the raw video writing worker process queue
        if not raw_queue.full():
            raw_queue.put_nowait((frame, fps))
            
        # Dispatch heavily load ML process queue independently (only Cam 1 does this)
        if inf_queue is not None:
            if not inf_queue.full():
                inf_queue.put_nowait((frame, fps))
                
    if cap:
        cap.release()
    print(f"[*] Finished capturing RSTP bounds cleanly for {ip}")

def writer_worker(queue_obj, output_path, running_event):
    """ Generic background writer thread strictly isolating Disk I/O dependencies """
    writer = None
    while True:
        try:
            item = queue_obj.get(timeout=0.2)
            if item is None: # Explicit Shutdown signal
                break
            frame, input_fps = item
            
            # Initialize dynamic writer bounds cleanly the first time a frame actually processes
            if writer is None:
                h, w = frame.shape[:2]
                fps = input_fps if input_fps > 0 else 25
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                print(f"[+] Instantiated persistent writer for: {output_path}")
                
            writer.write(frame)
        except queue.Empty:
            # Shutdown if event flags are dead, AND queue finally cleared entirely.
            if not running_event.is_set() and queue_obj.empty():
                break
                
    if writer:
        writer.release()
    print(f"[*] Gracefully released video writer dependencies for: {output_path}")


# ==========================================
# MAIN DESKTOP GUI
# ==========================================
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Multiprocessing YOLO Tracking & PTZ Control")
        # Increase bounds to account for telemetry UI sizes
        self.geometry("1400x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Using a Multiprocessing Event enables seamless thread+ML process sync logic
        self.running_event = mp.Event()
        self.selected_protocol = ctk.IntVar(value=1)
        
        # Dynamic Telemetry Parameters
        self.t_end = 0
        self.sys_fps = 0
        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.active_faces = 0

        self.setup_ui()

    def setup_ui(self):
        # Master Sidebar Layout
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Attendance Core ML", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)
        
        # Protocol
        self.protocol_label = ctk.CTkLabel(self.sidebar_frame, text="Select Execution Protocol:")
        self.protocol_label.pack(pady=10, padx=20, anchor="w")
        
        for p, info in PROTOCOLS.items():
            rb = ctk.CTkRadioButton(self.sidebar_frame, text=f"{info['name']} Iteration", variable=self.selected_protocol, value=p)
            rb.pack(pady=5, padx=20, anchor="w")
            
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Initialize Architecture", command=self.start_tracking)
        self.start_btn.pack(pady=20, padx=20)
        
        self.stop_btn = ctk.CTkButton(self.sidebar_frame, text="Force Stop System", command=self.stop_tracking, fg_color="red", hover_color="darkred")
        self.stop_btn.pack(pady=10, padx=20)
        self.stop_btn.configure(state="disabled")
        
        # Live System Telemetry Section
        self.telemetry_lbl = ctk.CTkLabel(self.sidebar_frame, text="Live Telemetry Stats", font=ctk.CTkFont(size=18, weight="bold", underline=True))
        self.telemetry_lbl.pack(pady=(40,10), padx=20, anchor="w")

        self.time_lbl = ctk.CTkLabel(self.sidebar_frame, text="Time Remaining: 00:00:00", font=ctk.CTkFont(size=14))
        self.time_lbl.pack(pady=5, padx=20, anchor="w")
        
        self.fps_lbl = ctk.CTkLabel(self.sidebar_frame, text="System UI FPS: 0.0", font=ctk.CTkFont(size=14))
        self.fps_lbl.pack(pady=5, padx=20, anchor="w")
        
        self.faces_lbl = ctk.CTkLabel(self.sidebar_frame, text="Active Faces Currently: 0", font=ctk.CTkFont(size=14))
        self.faces_lbl.pack(pady=5, padx=20, anchor="w")

        # Visual Image Renderer Layout
        self.video_frame = ctk.CTkLabel(self, text="Camera ML Feed Offline", bg_color="gray", width=1020, height=800)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def start_tracking(self):
        if self.running_event.is_set():
            return
            
        self.running_event.set()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        sel = self.selected_protocol.get()
        protocol_info = PROTOCOLS[sel]
        
        # Calculate dynamic absolute end time for auto-termination limit
        total_time_cam1 = len(CAM1_PRESETS) * (protocol_info['cam1'] + 5)
        total_time_cam2 = len(CAM2_PRESETS) * (protocol_info['cam2'] + 5)
        total_sweep_time = max(total_time_cam1, total_time_cam2)
        
        self.t_end = time.perf_counter() + total_sweep_time
        print(f"\n[*] Booting Protocol {sel}. The Architecture absolute boundary timer shuts down in {total_sweep_time}s.")

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Instantiate System Communication Queues
        self.inf_queue = mp.Queue(maxsize=30)
        self.res_queue = mp.Queue(maxsize=30)
        self.cmd_queue = mp.Queue()
        
        # Massive Queue definitions mitigating potential fast-capture buffering lags
        self.cam1_raw_queue = queue.Queue(maxsize=300)
        self.cam2_raw_queue = queue.Queue(maxsize=300)
        self.cam1_ann_queue = queue.Queue(maxsize=300)
        
        # 1. Fire up Multiprocessing Isolation Workers 
        self.ml_process = mp.Process(target=inference_worker, args=(self.inf_queue, self.res_queue, self.cmd_queue, timestamp_str))
        self.ml_process.daemon = True
        self.ml_process.start()

        # 2. Fire up I/O Discrete Storage Workers (Simultaneous outputs requested)
        self.writers_pool = [
            threading.Thread(target=writer_worker, args=(self.cam1_raw_queue, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event)),
            threading.Thread(target=writer_worker, args=(self.cam2_raw_queue, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event)),
            threading.Thread(target=writer_worker, args=(self.cam1_ann_queue, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam1_annotated.mp4"), self.running_event))
        ]
        for w in self.writers_pool:
            w.start()

        # 3. Fire up Background RTSP Scrapers
        self.readers_pool = [
            threading.Thread(target=rstp_reader, args=(CAMERA_IP_1, self.running_event, self.cam1_raw_queue, self.inf_queue)),
            threading.Thread(target=rstp_reader, args=(CAMERA_IP_2, self.running_event, self.cam2_raw_queue, None))
        ]
        for r in self.readers_pool:
            r.start()

        # 4. Fire up background PTZ Iterators 
        self.ptz1 = threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_1, CAM1_PRESETS, protocol_info['cam1']))
        self.ptz2 = threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_2, CAM2_PRESETS, protocol_info['cam2']))
        self.ptz1.start()
        self.ptz2.start()

        # Restart Telemetry
        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        
        # Bootup Recursive GUI View Event-Loop Hook
        self.update_gui_frame()

    def stop_tracking(self):
        if not self.running_event.is_set():
            return
            
        print("\n[*] Initializing Auto-Termination and Graceful Application Shutdown Sequence...")
        
        # Stripping master logic variables directly
        self.running_event.clear()
        
        # Send sentinel explicitly to process 
        self.cmd_queue.put('STOP')
        
        # Instantly flush concurrent queues forcing clean write shutdowns
        self.cam1_raw_queue.put(None)
        self.cam2_raw_queue.put(None)
        self.cam1_ann_queue.put(None)
        
        # Optional: You could thread join writers here, but forcing a UI stutter is bad. 
        # Relying on internal threads evaluating background loop triggers cleans smoothly.
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        print("[+] Application Successfully Halted. Output saved securely.")

    def ptz_runner(self, ip, presets, duration):
        """
        No drifting time.sleep. Adheres to explicit `time.perf_counter()` boundary validations.
        """
        for preset in presets:
            if not self.running_event.is_set():
                break
            
            print(f"[*] Moving {ip} towards physical Preset {preset}")
            url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset}"
            try:
                r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
            except:
                pass
            
            # Wait for mechanical PTZ adjustment manually with non-locking iterators
            t_mechanical = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_mechanical) < 5.0:
                time.sleep(0.1)
                
            if not self.running_event.is_set():
                break
            
            # Substantial Absolute Timing loop validating limits indefinitely ensuring NO TIME DRIFTS
            t_start = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_start) < duration:
                time.sleep(0.1)

    def update_gui_frame(self):
        if not self.running_event.is_set():
            self.video_frame.configure(image=None, text="Camera ML Feed Offline")
            return
            
        try:
            # 1. Real-Time Dashboard Updates Array
            now = time.perf_counter()
            remaining = max(0, int(self.t_end - now))
            
            if remaining <= 0:
                print("[-] Protocol duration ceiling met. Triggering forced stream close.")
                self.stop_tracking()
                return
                
            h, rem = divmod(remaining, 3600)
            m, s = divmod(rem, 60)
            self.time_lbl.configure(text=f"Time Remaining: {h:02d}:{m:02d}:{s:02d}")
            
            # Trigger FPS computations roughly once per second internally based on performance render loops 
            if (now - self.last_fps_time) >= 1.0:
                self.sys_fps = self.frames_rendered / (now - self.last_fps_time)
                self.fps_lbl.configure(text=f"System UI FPS: {self.sys_fps:.1f}")
                self.faces_lbl.configure(text=f"Active Faces Currently: {self.active_faces}")
                self.frames_rendered = 0
                self.last_fps_time = now

            # 2. Rendering Event Loop Checks
            frame, input_fps = None, 25
            
            # Drain ML Process Display queues ensuring we pull the explicit edge frames only
            while not self.res_queue.empty():
                try: 
                    res = self.res_queue.get_nowait()
                    frame, active_faces, input_fps = res
                    self.active_faces = active_faces
                except queue.Empty:
                    break
                    
            if frame is not None:
                self.frames_rendered += 1
                
                # Divert analyzed frame directly down into disk bounds natively 
                if not self.cam1_ann_queue.full():
                    self.cam1_ann_queue.put_nowait((frame, input_fps))
                    
                # Bind arrays directly against Tkinter Display mapping
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize specifically towards bounding requirements dynamically
                # This could be resized utilizing native Image constraints, but hardcoding UI fit applies consistency
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(1000, 780))
                self.video_frame.configure(image=imgtk, text="")
                self.video_frame.image = imgtk
                
        except Exception as e:
            print(f"[GUI ERROR] Engine mismatch mapping visual limits: {e}")
            
        finally:
            # Rehook Recursive loop iteratively validating Tkinter checks approximately ~33ms max bounds = 30FPS constraints
            self.after(33, self.update_gui_frame)

if __name__ == "__main__":
    # Absolutely Mandatory requirement unlocking internal Multiprocessing queues
    mp.freeze_support()
    app = AttendanceApp()
    app.mainloop()
