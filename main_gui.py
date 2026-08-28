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
from multiprocessing import shared_memory
import multiprocessing as mp
import torchvision.ops as ops
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
    2: {"name": "10-Minute", "cam1": 70, "cam2": 95},
    3: {"name": "15-Minute", "cam1": 107, "cam2": 145},
    4: {"name": "20-Minute", "cam1": 145, "cam2": 195},
    5: {"name": "25-Minute", "cam1": 182, "cam2": 245},
    6: {"name": "30-Minute", "cam1": 220, "cam2": 295},
    7: {"name": "35-Minute", "cam1": 257, "cam2": 345},
    8: {"name": "40-Minute", "cam1": 295, "cam2": 395}
}

CONFIDENCE_THRESHOLD = 0.79
FRAME_SKIP = 1
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


# ==========================================
# MULTIPROCESSING INFERENCE PROCESS (CUDA OPTIMIZED)
# ==========================================
def inference_worker(inf_queue, ann_queue, cmd_queue, timestamp_str, cam_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_half = torch.cuda.is_available()
    print(f"[ML PROCESS | {cam_name}] Initializing strictly on {device}...")
    
    try:
        yolo_model = YOLO('yolov8n-face.pt', task='detect')
        yolo_model.to(device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        if use_half:
            resnet = resnet.half()
        
        faiss_index_path = './face_attendance_faiss.bin'
        index = None
        db_tensor = None
        
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            try:
                if hasattr(index, 'make_direct_map'):
                    index.make_direct_map()
                db_embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
                for i in range(index.ntotal):
                    db_embeddings[i] = index.reconstruct(i)
                db_tensor = torch.from_numpy(db_embeddings).to(device)
                db_tensor = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
                if use_half:
                    db_tensor = db_tensor.half()
            except Exception:
                db_tensor = None
            
        meta_path = './face_attendance_meta.pkl'
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                saved_data = pickle.load(f)
            target_names = saved_data.get('target_names', [])
            y_real = saved_data.get('y_real', [])
        else:
            target_names, y_real = [], []
            
    except Exception as e:
        print(f"[ML PROCESS ERROR | {cam_name}] Failed to load models: {e}")
        return

    fps = 30
    frame_count = 0
    active_track_memory = {}
    archived_tracks = {}
    track_identities = {}

    print(f"[ML PROCESS | {cam_name}] Ready and listening for inbound streams...")

    while True:
        try:
            cmd = cmd_queue.get_nowait()
            if cmd == 'STOP':
                break
        except queue.Empty:
            pass

        try:
            item = inf_queue.get(timeout=0.1)
        except queue.Empty:
            continue
            
        meta, input_fps, skipped_frames = item
        shm_inf = None
        try:
            shm_inf = shared_memory.SharedMemory(name=meta['name'])
            frame = np.ndarray(meta['shape'], dtype=meta['dtype'], buffer=shm_inf.buf).copy()
        except FileNotFoundError:
            continue
        finally:
            if shm_inf is not None:
                try:
                    shm_inf.close()
                    shm_inf.unlink()
                except Exception:
                    pass
        
        frame_count += (skipped_frames + 1)
        fps = input_fps if input_fps > 0 else 30
        
        metadata = {'boxes': [], 'ids': [], 'names': []}
        current_active_faces = 0

        try:
            if (db_tensor is not None or index is not None) and len(target_names) > 0:
                results = yolo_model.track(
                    frame, 
                    persist=True, 
                    tracker="bytetrack.yaml", 
                    verbose=False, 
                    half=use_half, 
                    imgsz=640
                )
                
                boxes_obj = results[0].boxes
                has_detections = boxes_obj is not None and boxes_obj.id is not None
                
                if has_detections:
                    boxes = boxes_obj.xyxy
                    ids = boxes_obj.id.int()
                    current_active_faces = len(ids)
                    
                    alive_ids = ids.cpu().numpy().tolist()
                    for t_id in alive_ids:
                        if t_id not in active_track_memory:
                            active_track_memory[t_id] = {
                                'start_time': format_timestamp(frame_count, fps),
                                'frames_alive': 0, 'buffer': [], 'all_preds': [], 'missing_frames': 0
                            }
                        active_track_memory[t_id]['frames_alive'] += (skipped_frames + 1)

                    if frame_count % FRAME_SKIP == 0 or (frame_count - skipped_frames) % FRAME_SKIP == 0:
                        w = boxes[:, 2] - boxes[:, 0]
                        h = boxes[:, 3] - boxes[:, 1]
                        aspect_ratio = w / torch.clamp(h, min=1e-6)
                        valid_mask = (w >= 65) & (h >= 65) & (aspect_ratio >= 0.55) & (aspect_ratio <= 1.55)
                        
                        valid_indices = torch.where(valid_mask)[0]
                        if len(valid_indices) > 0:
                            v_boxes = boxes[valid_indices]
                            v_ids = [alive_ids[idx] for idx in valid_indices.cpu().numpy()]
                            
                            vw = v_boxes[:, 2] - v_boxes[:, 0]
                            vh = v_boxes[:, 3] - v_boxes[:, 1]
                            mx, my = vw * 0.15, vh * 0.15
                            
                            b_adj = v_boxes.clone()
                            b_adj[:, 0] = torch.clamp(v_boxes[:, 0] - mx, min=0)
                            b_adj[:, 1] = torch.clamp(v_boxes[:, 1] - my, min=0)
                            b_adj[:, 2] = torch.clamp(v_boxes[:, 2] + mx, max=frame.shape[1])
                            b_adj[:, 3] = torch.clamp(v_boxes[:, 3] + my, max=frame.shape[0])
                            
                            batch_idx = torch.zeros((v_boxes.size(0), 1), device=device, dtype=v_boxes.dtype)
                            b_batch = torch.cat([batch_idx, b_adj], dim=1)
                            
                            frame_tensor = torch.from_numpy(frame).to(device, non_blocking=True).float().permute(2, 0, 1).unsqueeze(0)
                            crops = ops.roi_align(frame_tensor, b_batch, output_size=(160, 160))
                            crops_rgb = crops[:, [2, 1, 0], :, :]
                            
                            gray = 0.2989 * crops_rgb[:, 0:1, :, :] + 0.5870 * crops_rgb[:, 1:2, :, :] + 0.1140 * crops_rgb[:, 2:3, :, :]
                            lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=device, dtype=gray.dtype)
                            lap_out = torch.nn.functional.conv2d(gray, lap_kernel, padding=1)
                            lap_var = torch.var(lap_out, dim=(1, 2, 3))
                            
                            sharp_mask = lap_var > 5.0
                            sharp_indices = torch.where(sharp_mask)[0]
                            
                            if len(sharp_indices) > 0:
                                valid_crops = crops_rgb[sharp_indices]
                                sharp_ids = [v_ids[i] for i in sharp_indices.cpu().numpy()]
                                
                                valid_crops = (valid_crops / 127.5) - 1.0
                                valid_crops = valid_crops.half() if use_half else valid_crops.float()
                                
                                with torch.inference_mode():
                                    emb = resnet(valid_crops)
                                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                                    
                                    if db_tensor is not None:
                                        db_tensor_cast = db_tensor.to(emb.dtype)
                                        sims = torch.mm(emb, db_tensor_cast.t())
                                        max_sims, max_indices = torch.max(sims, dim=1)
                                        m_sims_vals = max_sims.float().cpu().tolist()
                                        m_idx_vals = max_indices.cpu().tolist()
                                    elif index is not None:
                                        emb_cpu = emb.float().cpu().numpy()
                                        sims_res, indices_res = index.search(emb_cpu, k=1)
                                        m_sims_vals = [float(s[0]) for s in sims_res]
                                        m_idx_vals = [int(idx[0]) for idx in indices_res]
                                    else:
                                        m_sims_vals, m_idx_vals = [], []
                                        
                                    for i, t_id in enumerate(sharp_ids):
                                        if i < len(m_sims_vals):
                                            idx_match = m_idx_vals[i]
                                            sim_val = m_sims_vals[i]
                                            name = target_names[y_real[idx_match]] if (sim_val > CONFIDENCE_THRESHOLD and idx_match < len(y_real)) else "Unknown"
                                            
                                            active_track_memory[t_id]['buffer'].append(name)
                                            active_track_memory[t_id]['all_preds'].append(name)
                                            
                                            if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                                v_hist = [v for v in active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                                winner = Counter(v_hist).most_common(1)[0][0] if v_hist else "Unknown"
                                                track_identities[t_id] = winner
                                                active_track_memory[t_id]['buffer'] = []

                    meta_boxes = boxes.cpu().numpy().tolist()
                    for i in range(len(alive_ids)):
                        t_id = alive_ids[i]
                        metadata['boxes'].append(meta_boxes[i])
                        metadata['ids'].append(t_id)
                        metadata['names'].append(track_identities.get(t_id, "Analyzing..."))

                    alive_set = set(alive_ids)
                    for t_id in list(active_track_memory.keys()):
                        if t_id not in alive_set:
                            active_track_memory[t_id]['missing_frames'] += (skipped_frames + 1)
                            if active_track_memory[t_id]['missing_frames'] > 50:
                                archived_tracks[t_id] = active_track_memory.pop(t_id)
                        else:
                            active_track_memory[t_id]['missing_frames'] = 0

            try:
                ann_queue.put((metadata['boxes'], metadata['ids'], metadata['names'], skipped_frames, current_active_faces, fps), timeout=0.1)
            except queue.Full:
                pass
                
        except Exception as e:
            print(f"[ML PROCESS | {cam_name}] Runtime Error: {e}")

    # ==========================
    # FINAL ATTENDANCE SECURE DUMP
    # ==========================
    print(f"[ML PROCESS | {cam_name}] Shutting down natively... Generating CSV logs.")
    final_mem = {**archived_tracks, **active_track_memory}
    debug_data = []
    student_presence = {name: False for name in target_names}
    student_detection_count = {name: 0 for name in target_names}

    for t_id, data in final_mem.items():
        total_frames = data.get('frames_alive', 0)
        all_preds = data.get('all_preds', [])
        valid_preds = [p for p in all_preds if p != "Unknown"]
        valid_votes_count = len(valid_preds)
        
        if valid_votes_count > 0:
            counts = Counter(valid_preds)
            winner = counts.most_common(1)[0][0]
            win_ratio = counts.get(winner, 0) / valid_votes_count
            total_samples = len(all_preds)
            sample_ratio = counts.get(winner, 0) / total_samples if total_samples > 0 else 0
            status = "Passed" if (total_frames >= 45 and total_samples >= 15 and sample_ratio >= 0.33 and win_ratio >= 0.52) else "Failed"
        else:
            winner = "Unknown"
            status = "Failed"
            counts = Counter(all_preds)

        if status == "Passed" and winner != "Unknown" and winner in student_presence:
            student_presence[winner] = True
            student_detection_count[winner] += counts.get(winner, 0)

        debug_data.append({
            'Track ID': t_id, 'Start Time': data.get('start_time', ''),
            'Total Frames': total_frames, 'Valid Votes': valid_votes_count,
            'Total Preds (inc. Unknown)': len(all_preds), 'Predicted Identity': winner,
            'Gate Status': status, 'Breakdown': dict(Counter(all_preds))
        })

    if debug_data:
        pd.DataFrame(debug_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_{cam_name}_DEBUG_Tracks.csv"), index=False)
    output_data = [{'Name': s, 'Status': 'Present' if student_presence[s] else 'Absent', 'Detection Count': student_detection_count[s]} for s in target_names]
    pd.DataFrame(output_data).to_csv(os.path.join(RESULTS_DIR, f"{timestamp_str}_{cam_name}_output.csv"), index=False)


# ==========================================
# BACKGROUND THREADS (I/O & STREAMING)
# ==========================================
def rstp_reader(ip, running_event, raw_queue, inf_queue, ann_frame_queue=None, enable_inference=True):
    url = f"rtsp://{USERNAME}:{PASSWORD}@{ip}:554/stream1"
    cap = None
    last_frame_time = time.perf_counter()
    
    while running_event.is_set():
        if cap is None or not cap.isOpened():
            print(f"[*] Native Reader Booting IP {ip}...")
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                time.sleep(3)
                continue
            last_frame_time = time.perf_counter()
            
        ret, frame = cap.read()
        now = time.perf_counter()
        
        if not ret:
            cap.release()
            cap = None
            continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30
        
        target_interval = 1.0 / fps
        delta = now - last_frame_time
        skipped_frames = max(0, round(delta / target_interval) - 1)
        last_frame_time = now
        
        # In-process raw recording (Thread-safe memory reference)
        if raw_queue is not None and not raw_queue.full():
            raw_queue.put_nowait((frame, fps, skipped_frames + 1))
            
        # Real-time UI and Annotation Feed (Main process queue)
        if ann_frame_queue is not None:
            if ann_frame_queue.full():
                try:
                    ann_frame_queue.get_nowait()
                except queue.Empty:
                    pass
            ann_frame_queue.put_nowait((frame, fps, skipped_frames))

        # Inter-Process Shared Memory for ML Background Process
        if enable_inference and inf_queue is not None and not inf_queue.full():
            try:
                shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                shm_arr = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                shm_arr[:] = frame[:]
                inf_queue.put_nowait(({'name': shm.name, 'shape': frame.shape, 'dtype': str(frame.dtype)}, fps, skipped_frames))
                shm.close()
            except Exception:
                pass
                
    if cap:
        cap.release()

def raw_writer_worker(queue_obj, ui_queue, output_path, running_event, populate_ui=False):
    writer = None
    while True:
        try:
            item = queue_obj.get(timeout=0.2)
            if item is None:
                break
            frame, input_fps, duplicates = item
            
            if writer is None:
                h, w = frame.shape[:2]
                fps = input_fps if input_fps > 0 else 30
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
            for _ in range(duplicates):
                writer.write(frame)
                
            if populate_ui and ui_queue is not None and not ui_queue.full():
                ui_frame = cv2.resize(frame, (1024, 576))
                ui_queue.put_nowait((ui_frame, 0))
                
        except queue.Empty:
            if not running_event.is_set() and queue_obj.empty():
                break
    if writer:
        writer.release()

def async_video_writer_worker(writer_queue, output_path, running_event):
    writer = None
    while True:
        try:
            item = writer_queue.get(timeout=0.2)
            if item is None:
                break
            frame, input_fps, duplicates = item
            
            if writer is None:
                h, w = frame.shape[:2]
                fps = input_fps if input_fps > 0 else 30
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
            for _ in range(duplicates):
                writer.write(frame)
                
        except queue.Empty:
            if not running_event.is_set() and writer_queue.empty():
                break
    if writer:
        writer.release()

def ann_writer_worker(ann_queue, ann_frame_queue, ui_queue, writer_queue, running_event):
    while True:
        try:
            item = ann_queue.get(timeout=0.2)
            if item is None:
                break
            boxes, ids, names, skipped_frames, active_faces, input_fps = item
                
            frame = None
            try:
                frame_item = ann_frame_queue.get(timeout=0.2)
                raw_frame, _, _ = frame_item
                frame = raw_frame.copy()
            except queue.Empty:
                continue

            for i in range(len(ids)):
                box = boxes[i]
                t_id = ids[i]
                name = names[i]
                color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            if writer_queue is not None and not writer_queue.full():
                writer_queue.put_nowait((frame, input_fps, skipped_frames + 1))
                
            if ui_queue is not None and not ui_queue.full():
                ui_frame = cv2.resize(frame, (1024, 576))
                ui_queue.put_nowait((ui_frame, active_faces))
                
        except queue.Empty:
            if not running_event.is_set() and ann_queue.empty():
                break


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
        self.gui_loop_active = False

        self.setup_ui()

    def setup_ui(self):
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Attendance Core ML", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)
        
        self.view_label = ctk.CTkLabel(self.sidebar_frame, text="Active Telemetry View:", font=ctk.CTkFont(size=14, underline=True))
        self.view_label.pack(pady=(10,0), padx=20, anchor="w")
        self.view_toggle = ctk.CTkSegmentedButton(self.sidebar_frame, values=["Camera 1", "Camera 2"], variable=self.view_target)
        self.view_toggle.pack(pady=10, padx=20, fill="x")

        self.cam1_enable_var = ctk.BooleanVar(value=True)
        self.cam1_enable_switch = ctk.CTkSwitch(self.sidebar_frame, text="Enable Camera 1", variable=self.cam1_enable_var)
        self.cam1_enable_switch.pack(pady=(10, 0), padx=20, anchor="w")
        
        self.cam2_enable_var = ctk.BooleanVar(value=True)
        self.cam2_enable_switch = ctk.CTkSwitch(self.sidebar_frame, text="Enable Camera 2", variable=self.cam2_enable_var)
        self.cam2_enable_switch.pack(pady=10, padx=20, anchor="w")

        self.global_live_tracking_var = ctk.BooleanVar(value=True)
        self.global_live_tracking_switch = ctk.CTkSwitch(self.sidebar_frame, text="Live Tracking", variable=self.global_live_tracking_var)
        self.global_live_tracking_switch.pack(pady=(15,0), padx=20, anchor="w")
        
        self.protocol_label = ctk.CTkLabel(self.sidebar_frame, text="Execution Protocol:")
        self.protocol_label.pack(pady=(15,0), padx=20, anchor="w")
        self.protocol_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.protocol_frame.pack(pady=5, padx=20, fill="x")
        
        for idx, (p, info) in enumerate(PROTOCOLS.items()):
            r = idx // 2
            c = idx % 2
            rb = ctk.CTkRadioButton(self.protocol_frame, text=info['name'], variable=self.selected_protocol, value=p)
            rb.grid(row=r, column=c, pady=5, padx=5, sticky="w")
            
        self.ptz_label1 = ctk.CTkLabel(self.sidebar_frame, text="Cam1 Presets:")
        self.ptz_label1.pack(pady=(10,0), padx=20, anchor="w")
        self.preset_frame1 = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.preset_frame1.pack(pady=5, padx=20, fill="x")
        self.cam1_preset_vars = {}
        for p in CAM1_PRESETS:
            var = ctk.BooleanVar(value=True)
            self.cam1_preset_vars[p] = var
            cb = ctk.CTkCheckBox(self.preset_frame1, text=str(p), variable=var, width=40)
            cb.grid(row=(p-1)//4, column=(p-1)%4, padx=5, pady=5)

        self.ptz_label2 = ctk.CTkLabel(self.sidebar_frame, text="Cam2 Presets:")
        self.ptz_label2.pack(pady=(10,0), padx=20, anchor="w")
        self.preset_frame2 = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.preset_frame2.pack(pady=5, padx=20, fill="x")
        self.cam2_preset_vars = {}
        for p in CAM2_PRESETS:
            var = ctk.BooleanVar(value=True)
            self.cam2_preset_vars[p] = var
            cb = ctk.CTkCheckBox(self.preset_frame2, text=str(p), variable=var, width=40)
            cb.grid(row=(p-1)//4, column=(p-1)%4, padx=5, pady=5)

        self.ptz_manual_label = ctk.CTkLabel(self.sidebar_frame, text="Manual PTZ Controls:")
        self.ptz_manual_label.pack(pady=(10,0), padx=20, anchor="w")
        self.manual_ptz_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.manual_ptz_frame.pack(pady=5, padx=20, fill="x")
        
        btn_up = ctk.CTkButton(self.manual_ptz_frame, text="Up", width=50)
        btn_down = ctk.CTkButton(self.manual_ptz_frame, text="Down", width=50)
        btn_left = ctk.CTkButton(self.manual_ptz_frame, text="Left", width=50)
        btn_right = ctk.CTkButton(self.manual_ptz_frame, text="Right", width=50)
        btn_zi = ctk.CTkButton(self.manual_ptz_frame, text="Zoom In", width=50)
        btn_zo = ctk.CTkButton(self.manual_ptz_frame, text="Zoom Out", width=50)
        btn_fi = ctk.CTkButton(self.manual_ptz_frame, text="Focus In", width=50)
        btn_fo = ctk.CTkButton(self.manual_ptz_frame, text="Focus Out", width=50)

        for btn, cmd in [
            (btn_up, "up"), (btn_down, "down"), (btn_left, "left"), 
            (btn_right, "right"), (btn_zi, "zoomin"), (btn_zo, "zoomout"), 
            (btn_fi, "focusin"), (btn_fo, "focusout")
        ]:
            for widget in [btn, btn._canvas, btn._text_label]:
                widget.bind("<ButtonPress-1>", lambda e, c=cmd: self.manual_ptz(c))
                widget.bind("<ButtonRelease-1>", lambda e: self.manual_ptz("ptzstop"))

        btn_up.grid(row=0, column=1, padx=2, pady=2)
        btn_down.grid(row=2, column=1, padx=2, pady=2)
        btn_left.grid(row=1, column=0, padx=2, pady=2)
        btn_right.grid(row=1, column=2, padx=2, pady=2)
        btn_zi.grid(row=0, column=3, padx=2, pady=2)
        btn_zo.grid(row=2, column=3, padx=2, pady=2)
        btn_fi.grid(row=0, column=4, padx=2, pady=2)
        btn_fo.grid(row=2, column=4, padx=2, pady=2)
            
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

        self.video_frame = ctk.CTkLabel(self, text="Camera ML Feed Offline", bg_color="gray", width=1024, height=576)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def start_tracking(self):
        if self.running_event.is_set():
            return
            
        self.running_event.set()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        sel = self.selected_protocol.get()
        p_info = PROTOCOLS[sel]

        selected_cam1_presets = [p for p, var in self.cam1_preset_vars.items() if var.get()]
        selected_cam2_presets = [p for p, var in self.cam2_preset_vars.items() if var.get()]
        
        use_cam1 = self.cam1_enable_var.get()
        use_cam2 = self.cam2_enable_var.get()
        
        if not use_cam1 and not use_cam2:
            print("[Warning] Both cameras disabled. Cannot start tracking.")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.running_event.clear()
            return
        
        global_track = self.global_live_tracking_var.get()
        mode1 = global_track
        mode2 = global_track
        
        total_time_cam1 = len(selected_cam1_presets) * (p_info['cam1'] + 5) if use_cam1 else 0
        total_time_cam2 = len(selected_cam2_presets) * (p_info['cam2'] + 5) if use_cam2 else 0
        duration = max(total_time_cam1, total_time_cam2)
        if duration <= 0:
            duration = 300
        self.t_end = time.perf_counter() + duration

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.threads = []
        
        if use_cam1:
            self.cam1_raw_q = queue.Queue(maxsize=60)
            self.cam1_ui_q = queue.Queue(maxsize=30)
            
            if mode1:
                self.cam1_inf_q = mp.Queue(maxsize=15)
                self.cam1_ann_q = mp.Queue(maxsize=150)
                self.cam1_cmd_q = mp.Queue()
                self.cam1_ann_frame_q = queue.Queue(maxsize=10)
                self.cam1_writer_q = queue.Queue(maxsize=150)
            else:
                self.cam1_inf_q = None
                self.cam1_ann_q = None
                self.cam1_cmd_q = None
                self.cam1_ann_frame_q = None
                self.cam1_writer_q = None
                
            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_1, self.running_event, self.cam1_raw_q, self.cam1_inf_q, self.cam1_ann_frame_q, mode1), daemon=True))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_1, selected_cam1_presets, p_info['cam1']), daemon=True))
            
            if mode1:
                self.ml_p1 = mp.Process(target=inference_worker, args=(self.cam1_inf_q, self.cam1_ann_q, self.cam1_cmd_q, timestamp_str, "Cam1"))
                self.ml_p1.daemon = True
                self.ml_p1.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event, False), daemon=True))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam1_ann_q, self.cam1_ann_frame_q, self.cam1_ui_q, self.cam1_writer_q, self.running_event), daemon=True))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam1_writer_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam1_annotated.mp4"), self.running_event), daemon=True))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, self.cam1_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event, True), daemon=True))

        if use_cam2:
            self.cam2_raw_q = queue.Queue(maxsize=60)
            self.cam2_ui_q = queue.Queue(maxsize=30)
            
            if mode2:
                self.cam2_inf_q = mp.Queue(maxsize=15)
                self.cam2_ann_q = mp.Queue(maxsize=150)
                self.cam2_cmd_q = mp.Queue()
                self.cam2_ann_frame_q = queue.Queue(maxsize=10)
                self.cam2_writer_q = queue.Queue(maxsize=150)
            else:
                self.cam2_inf_q = None
                self.cam2_ann_q = None
                self.cam2_cmd_q = None
                self.cam2_ann_frame_q = None
                self.cam2_writer_q = None
                
            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_2, self.running_event, self.cam2_raw_q, self.cam2_inf_q, self.cam2_ann_frame_q, mode2), daemon=True))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_2, selected_cam2_presets, p_info['cam2']), daemon=True))
            
            if mode2:
                self.ml_p2 = mp.Process(target=inference_worker, args=(self.cam2_inf_q, self.cam2_ann_q, self.cam2_cmd_q, timestamp_str, "Cam2"))
                self.ml_p2.daemon = True
                self.ml_p2.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event, False), daemon=True))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam2_ann_q, self.cam2_ann_frame_q, self.cam2_ui_q, self.cam2_writer_q, self.running_event), daemon=True))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam2_writer_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam2_annotated.mp4"), self.running_event), daemon=True))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, self.cam2_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event, True), daemon=True))

        for t in self.threads:
            t.start()

        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        
        if not self.gui_loop_active:
            self.gui_loop_active = True
            self.update_gui_frame()

    def stop_tracking(self):
        if not self.running_event.is_set():
            return
        self.running_event.clear()
        
        if hasattr(self, 'cam1_cmd_q') and self.cam1_cmd_q:
            self.cam1_cmd_q.put('STOP')
        if hasattr(self, 'cam2_cmd_q') and self.cam2_cmd_q:
            self.cam2_cmd_q.put('STOP')
        if hasattr(self, 'cam1_ann_q') and self.cam1_ann_q:
            self.cam1_ann_q.put(None)
        if hasattr(self, 'cam2_ann_q') and self.cam2_ann_q:
            self.cam2_ann_q.put(None)
            
        if hasattr(self, 'cam1_writer_q') and self.cam1_writer_q:
            self.cam1_writer_q.put(None)
        if hasattr(self, 'cam2_writer_q') and self.cam2_writer_q:
            self.cam2_writer_q.put(None)
            
        if hasattr(self, 'cam1_raw_q') and self.cam1_raw_q:
            self.cam1_raw_q.put(None)
        if hasattr(self, 'cam2_raw_q') and self.cam2_raw_q:
            self.cam2_raw_q.put(None)
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def ptz_runner(self, ip, presets, duration):
        for preset in presets:
            if not self.running_event.is_set():
                break
            url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset}"
            try:
                requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
            except Exception:
                pass
            
            t_mechanical = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_mechanical) < 5.0:
                time.sleep(0.1)
                
            t_start = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_start) < duration:
                time.sleep(0.1)

    def manual_ptz(self, command):
        target = self.view_target.get()
        ip = CAMERA_IP_1 if target == "Camera 1" else CAMERA_IP_2
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}"
        def send_cmd():
            if command == 'ptzstop':
                time.sleep(0.1)
            try:
                requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=3)
            except Exception:
                pass
        threading.Thread(target=send_cmd, daemon=True).start()

    def update_gui_frame(self):
        if not self.running_event.is_set():
            self.video_frame.configure(image=None, text="Camera ML Feed Offline")
            self.gui_loop_active = False
            return
            
        try:
            now = time.perf_counter()
            remaining = max(0, int(self.t_end - now))
            
            if remaining <= 0:
                self.stop_tracking()
                return
                
            h, rem = divmod(remaining, 3600)
            m, s = divmod(rem, 60)
            self.time_lbl.configure(text=f"Remaining Timer: {h:02d}:{m:02d}:{s:02d}")
            
            frame1 = None
            if hasattr(self, 'cam1_ui_q') and self.cam1_ui_q and not self.cam1_ui_q.empty():
                try:
                    res1 = self.cam1_ui_q.get_nowait()
                    frame1, self.cam1_active_faces = res1
                except queue.Empty:
                    pass
                    
            frame2 = None
            if hasattr(self, 'cam2_ui_q') and self.cam2_ui_q and not self.cam2_ui_q.empty():
                try:
                    res2 = self.cam2_ui_q.get_nowait()
                    frame2, self.cam2_active_faces = res2
                except queue.Empty:
                    pass
            
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
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(1024, 576))
                self.video_frame.configure(image=imgtk, text="")
                self.video_frame.image = imgtk
                
        except Exception:
            pass
        finally:
            if self.running_event.is_set():
                self.after(15, self.update_gui_frame)
            else:
                self.gui_loop_active = False

if __name__ == "__main__":
    mp.freeze_support()
    app = AttendanceApp()
    app.mainloop()