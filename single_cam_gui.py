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
import torchvision.ops as ops
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from requests.auth import HTTPBasicAuth
import multiprocessing as mp
import multiprocessing.shared_memory as shared_memory

# ==========================================
# CONFIGURATION
# ==========================================
USERNAME = "admin"
PASSWORD = "spcvlab@2023"
CAMERA_IP = "10.23.12.40"

CAM_PRESETS = [1, 2, 3, 4, 5, 6, 7, 8]

PROTOCOLS = {
    1: {"name": "5-Minute", "cam": 32},
    2: {"name": "10-Minute", "cam": 70},
    3: {"name": "15-Minute", "cam": 107},
    4: {"name": "20-Minute", "cam": 145},
    5: {"name": "25-Minute", "cam": 182},
    6: {"name": "30-Minute", "cam": 220},
    7: {"name": "35-Minute", "cam": 257},
    8: {"name": "40-Minute", "cam": 295}
}

CONFIDENCE_THRESHOLD = 0.75
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
# MULTIPROCESSING INFERENCE PROCESS (THROTTLED CPU)
# ==========================================
def inference_worker(inf_queue, ann_queue, cmd_queue, timestamp_str, cam_name):
    """
    Evaluates ML on throttled inbound frames inherently preventing deadlocks.
    Writes native annotations directly against duplicate outputs maintaining 1-1 strict bounds flawlessly.
    """
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
        ref_embeddings_tensor = None
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            index.nprobe = 20
            try:
                try:
                    ref_embeddings = index.reconstruct_n(0, index.ntotal)
                except AttributeError:
                    index.make_direct_map()
                    ref_embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])
                
                ref_embeddings_tensor = torch.from_numpy(ref_embeddings).to(device).float()
                ref_embeddings_tensor = torch.nn.functional.normalize(ref_embeddings_tensor, p=2, dim=1)
                if use_half:
                    ref_embeddings_tensor = ref_embeddings_tensor.half()
            except Exception as e:
                print(f"[ML PROCESS | {cam_name}] Failed to extract FAISS vectors to GPU: {e}")
                index = None
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
            
        shm = None
        try:
            shm = shared_memory.SharedMemory(name=item['shm_name'])
            frame_array = np.ndarray(item['shape'], dtype=item['dtype'], buffer=shm.buf)
            
            # Pure GPU Optimization: Immediately move frame tensor native to GPU bounds.
            frame_tensor = torch.from_numpy(frame_array).to(device, non_blocking=True).float()
            
            input_fps = item['fps']
            skipped_frames = item['skipped_frames']
            
            frame_count += (skipped_frames + 1)
            fps = input_fps if input_fps > 0 else 30
            
            # Deadlock Elimination: Collect tracking metadata strictly decoupled from UI loop drawing frames globally
            metadata = {'boxes': [], 'ids': [], 'names': []}
            current_active_faces = 0

            if ref_embeddings_tensor is not None and len(target_names) > 0:
                results = yolo_model.track(frame_array, persist=True, tracker="custom_bytetrack.yaml", verbose=False, quantize=16 if use_half else None, imgsz=640)
                has_detections = results[0].boxes.id is not None
                
                if has_detections:
                    # Explicitly move the tensors to the GPU to match batch_idx
                    boxes = results[0].boxes.xyxy.to(device) 
                    ids = results[0].boxes.id.to(device).int() 
                    current_active_faces = len(ids)
                    
                    ids_list = ids.cpu().tolist()
                    for t_id in ids_list:
                        if t_id not in active_track_memory:
                            active_track_memory[t_id] = {
                                'start_time': format_timestamp(frame_count, fps),
                                'frames_alive': 0, 'buffer': [], 'all_preds': [], 'missing_frames': 0,
                                'crop_buffer': []
                            }
                        active_track_memory[t_id]['frames_alive'] += (skipped_frames + 1)

                    if frame_count % FRAME_SKIP == 0 or (frame_count - skipped_frames) % FRAME_SKIP == 0:
                        batch_tensors, batch_track_ids = [], []
                        
                        box_w = boxes[:, 2] - boxes[:, 0]
                        box_h = boxes[:, 3] - boxes[:, 1]
                        aspect_ratios = box_w / (box_h + 1e-6)
                        
                        valid_mask = (box_w >= 65) & (box_h >= 65) & (aspect_ratios >= 0.55) & (aspect_ratios <= 1.55)
                        
                        if valid_mask.any():
                            valid_boxes = boxes[valid_mask].clone()
                            valid_ids = ids[valid_mask]
                            
                            margin_x = (valid_boxes[:, 2] - valid_boxes[:, 0]) * 0.15
                            margin_y = (valid_boxes[:, 3] - valid_boxes[:, 1]) * 0.15
                            
                            valid_boxes[:, 0] = torch.clamp(valid_boxes[:, 0] - margin_x, min=0)
                            valid_boxes[:, 1] = torch.clamp(valid_boxes[:, 1] - margin_y, min=0)
                            valid_boxes[:, 2] = torch.clamp(valid_boxes[:, 2] + margin_x, max=float(frame_array.shape[1]))
                            valid_boxes[:, 3] = torch.clamp(valid_boxes[:, 3] + margin_y, max=float(frame_array.shape[0]))
                            
                            batch_idx = torch.zeros((valid_boxes.size(0), 1), device=device, dtype=valid_boxes.dtype)
                            roi_boxes = torch.cat((batch_idx, valid_boxes), dim=1)
                            
                            # Standardize to NCHW format
                            frame_tensor_chw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
                            
                            # Native GPU Vectorized Extraction and Resize
                            crops = ops.roi_align(frame_tensor_chw, roi_boxes, output_size=(160, 160)).detach()
                            
                            valid_ids_list = valid_ids.cpu().tolist()
                            for i, t_id in enumerate(valid_ids_list):
                                active_track_memory[t_id]['crop_buffer'].append(crops[i:i+1])
                                
                                if len(active_track_memory[t_id]['crop_buffer']) >= FRAMES_PER_VOTE:
                                    batch_tensors.extend(active_track_memory[t_id]['crop_buffer'])
                                    batch_track_ids.extend([t_id] * len(active_track_memory[t_id]['crop_buffer']))
                                    active_track_memory[t_id]['crop_buffer'] = []
                        
                        if batch_tensors:
                            with torch.inference_mode():
                                batch_tensor = torch.cat(batch_tensors, dim=0) # (N, 3, 160, 160)
                                batch_tensor = batch_tensor[:, [2, 1, 0], :, :] # Swap BGR to RGB natively
                                
                                # GPU Vectorized Blur Filtering
                                gray = transforms.functional.rgb_to_grayscale(batch_tensor)
                                laplacian_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=device, dtype=batch_tensor.dtype)
                                laplacian_out = torch.nn.functional.conv2d(gray, laplacian_kernel, padding=1)
                                laplacian_var = torch.var(laplacian_out, dim=(1, 2, 3))
                                
                                mask = laplacian_var > 7.0
                                mask_list = mask.cpu().tolist()
                                valid_batch_tensor = batch_tensor[mask]
                                valid_batch_track_ids = [batch_track_ids[k] for k in range(len(batch_track_ids)) if mask_list[k]]
                                
                                if valid_batch_tensor.size(0) > 0:
                                    valid_batch_tensor = (valid_batch_tensor / 127.5) - 1.0
                                    if use_half:
                                        valid_batch_tensor = valid_batch_tensor.half()
                                    
                                    embeddings = resnet(valid_batch_tensor)
                                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                                    
                                    sim_matrix = torch.mm(embeddings, ref_embeddings_tensor.t())
                                    max_sims, max_indices = torch.max(sim_matrix, dim=1)
                                    
                                    sims_list = max_sims.cpu().tolist()
                                    indices_list = max_indices.cpu().tolist()
                                    
                                    for i, t_id in enumerate(valid_batch_track_ids):
                                        name = target_names[y_real[indices_list[i]]] if sims_list[i] > CONFIDENCE_THRESHOLD else "Unknown"
                                        active_track_memory[t_id]['buffer'].append(name)
                                        active_track_memory[t_id]['all_preds'].append(name)
                                        
                                        if len(active_track_memory[t_id]['buffer']) >= FRAMES_PER_VOTE:
                                            valid_history = [v for v in active_track_memory[t_id]['all_preds'] if v != "Unknown"]
                                            winner = Counter(valid_history).most_common(1)[0][0] if valid_history else "Unknown"
                                            track_identities[t_id] = winner
                                            active_track_memory[t_id]['buffer'] = []

                    # Collect tracking metadata instead of drawing on frames directly
                    boxes_cpu = boxes.cpu().numpy()
                    for i in range(len(ids_list)):
                        t_id = ids_list[i]
                        box = boxes_cpu[i]
                        name = track_identities.get(t_id, "Analyzing...")
                        metadata['boxes'].append(box)
                        metadata['ids'].append(t_id)
                        metadata['names'].append(name)

                    alive_ids = set(ids_list)
                else:
                    alive_ids = set()

                for t_id in list(active_track_memory.keys()):
                    if t_id not in alive_ids:
                        active_track_memory[t_id]['missing_frames'] += (skipped_frames + 1)
                        if active_track_memory[t_id]['missing_frames'] > 50:
                            archived_tracks[t_id] = active_track_memory.pop(t_id)
                    else:
                        active_track_memory[t_id]['missing_frames'] = 0

        finally:
            if shm is not None:
                shm.close()
                shm.unlink()

        # Enforce highly reliable logic outputs directly to single bound queue strictly avoiding cross queue ID locks!
        try:
            ann_queue.put((metadata['boxes'], metadata['ids'], metadata['names'], skipped_frames, current_active_faces, fps), timeout=0.2)
        except queue.Full:
            pass

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
            status = "Passed" if (total_frames >= 45 and total_samples >= 15 and sample_ratio >= 0.33 and win_ratio >= 0.52) else "Failed"
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
def rstp_reader(ip, running_event, raw_queue, inf_queue, ann_frame_queue=None, enable_inference=True):
    """ 
    Absolute Watchdog. 
    Implements CPU optimization via Input Throttling caching explicit dropped frames natively. 
    """
    url = f"rtsp://admin:spcvlab%402023@{ip}:554/cam/realmonitor?channel=1&subtype=0"
    cap = None
    last_frame_time = time.perf_counter()
    
    while running_event.is_set():
        if cap is None or not cap.isOpened():
            print(f"[*] Native Reader Booting IP {ip}...")
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                time.sleep(5)
                continue
            last_frame_time = time.perf_counter()
            
        ret, frame = cap.read()
        now = time.perf_counter()
        
        if not ret:
            cap.release(); cap = None; continue
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0: fps = 30
        
        target_interval = 1.0 / fps
        delta = now - last_frame_time
        skipped_frames = max(0, round(delta / target_interval) - 1)
        last_frame_time = now
        
        # Raw Writer explicitly gets all original non-throttled data payload dictionaries (Zero-Copy)
        if not raw_queue.full():
            try:
                shm_raw = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm_raw.buf)[:] = frame[:]
                raw_queue.put_nowait({'shm_name': shm_raw.name, 'shape': frame.shape, 'dtype': frame.dtype, 'fps': fps, 'skipped_frames': skipped_frames + 1})
                shm_raw.close()
            except Exception: pass
            
        # Throttled Dispatch payload dicts evaluating constraints preventing CPU Serialization overload arrays entirely
        if enable_inference:
            if not inf_queue.full() and (ann_frame_queue is None or not ann_frame_queue.full()):
                try:
                    shm_inf = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                    np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm_inf.buf)[:] = frame[:]
                    inf_queue.put_nowait({'shm_name': shm_inf.name, 'shape': frame.shape, 'dtype': frame.dtype, 'fps': fps, 'skipped_frames': skipped_frames})
                    shm_inf.close()
                    
                    if ann_frame_queue is not None:
                        shm_ann = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                        np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm_ann.buf)[:] = frame[:]
                        ann_frame_queue.put_nowait({'shm_name': shm_ann.name, 'shape': frame.shape, 'dtype': frame.dtype, 'fps': fps, 'skipped_frames': skipped_frames})
                        shm_ann.close()
                except Exception: pass
                
    if cap: cap.release()

def raw_writer_worker(queue_obj, ui_queue, output_path, running_event, populate_ui=False):
    writer = None
    while True:
        try:
            item = queue_obj.get(timeout=0.2)
            if item is None: break
            
            shm = None
            try:
                shm = shared_memory.SharedMemory(name=item['shm_name'])
                frame = np.ndarray(item['shape'], dtype=item['dtype'], buffer=shm.buf)
                
                input_fps = item['fps']
                duplicates = item['skipped_frames']
                
                if writer is None:
                    h, w = frame.shape[:2]
                    fps = input_fps if input_fps > 0 else 30
                    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                for _ in range(duplicates):
                    writer.write(frame)
                    
                if populate_ui and ui_queue is not None:
                    if not ui_queue.full():
                        ui_frame = cv2.resize(frame, (1024, 576))
                        ui_queue.put_nowait((ui_frame, 0))
            finally:
                if shm is not None:
                    shm.close()
                    shm.unlink()
                    
        except queue.Empty:
            if not running_event.is_set() and queue_obj.empty(): break
    if writer: writer.release()

def async_video_writer_worker(writer_queue, output_path, running_event):
    writer = None
    while True:
        try:
            item = writer_queue.get(timeout=0.2)
            if item is None: break
            frame, input_fps, duplicates = item
            
            if writer is None:
                h, w = frame.shape[:2]
                fps = input_fps if input_fps > 0 else 30
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                
            for _ in range(duplicates):
                writer.write(frame)
                
        except queue.Empty:
            if not running_event.is_set() and writer_queue.empty(): break
            
    if writer: writer.release()

def ann_writer_worker(ann_queue, ann_frame_queue, ui_queue, writer_queue, running_event):
    """
    Highly robust generic bounds mapping directly towards duplicating skipped loops.
    Strictly deadlock-free! 
    """
    while True:
        try:
            item = ann_queue.get(timeout=0.2)
            if item is None: break
            boxes, ids, names, skipped_frames, active_faces, input_fps = item
                
            try:
                frame_item = ann_frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            shm = None
            try:
                shm = shared_memory.SharedMemory(name=frame_item['shm_name'])
                frame_arr = np.ndarray(frame_item['shape'], dtype=frame_item['dtype'], buffer=shm.buf)
                frame = frame_arr.copy()
            finally:
                if shm is not None:
                    shm.close()
                    shm.unlink()

            for i in range(len(ids)):
                box = boxes[i]
                t_id = ids[i]
                name = names[i]
                color = (0, 255, 0) if name not in ["Unknown", "Analyzing..."] else (0, 0, 255)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            if not writer_queue.full():
                writer_queue.put_nowait((frame, input_fps, skipped_frames + 1))
                
            if not ui_queue.full():
                ui_frame = cv2.resize(frame, (1024, 576))
                ui_queue.put_nowait((ui_frame, active_faces))
                
        except queue.Empty:
            if not running_event.is_set() and ann_queue.empty(): break

# ==========================================
# MAIN DESKTOP GUI
# ==========================================
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Single-Process Optimized Architecture")
        self.geometry("1400x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.running_event = mp.Event()
        self.selected_protocol = ctk.IntVar(value=1)
        
        self.t_end = 0
        self.sys_fps = 0
        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.active_faces = 0

        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.ptz_cmd_queue = queue.Queue()
        self.ptz_worker_thread = threading.Thread(target=self.ptz_command_worker, daemon=True)
        self.ptz_worker_thread.start()

    def ptz_command_worker(self):
        while True:
            url = self.ptz_cmd_queue.get()
            if url == "QUIT": break
            try:
                requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=3)
            except Exception:
                pass
        
    def on_closing(self):
        print("Application shutting down... cleaning up memory and processes.")
        self.stop_tracking()
        if hasattr(self, 'ptz_cmd_queue'):
            self.ptz_cmd_queue.put("QUIT")
        self.destroy()
        
    def setup_ui(self):
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Attendance Core ML", font=ctk.CTkFont(size=22, weight="bold"))
        self.logo_label.pack(pady=20, padx=20)

        self.cam_enable_var = ctk.BooleanVar(value=True)
        self.cam_enable_switch = ctk.CTkSwitch(self.sidebar_frame, text="Enable Camera", variable=self.cam_enable_var)
        self.cam_enable_switch.pack(pady=(10, 0), padx=20, anchor="w")
        
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
            
        self.ptz_label = ctk.CTkLabel(self.sidebar_frame, text="Camera Presets:")
        self.ptz_label.pack(pady=(10,0), padx=20, anchor="w")
        self.preset_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.preset_frame.pack(pady=5, padx=20, fill="x")
        self.cam_preset_vars = {}
        for p in CAM_PRESETS:
            var = ctk.BooleanVar(value=True)
            self.cam_preset_vars[p] = var
            cb = ctk.CTkCheckBox(self.preset_frame, text=str(p), variable=var, width=40)
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

        for btn, cmd, stop_cmd in [
            (btn_up, "up", "ptzstop"), (btn_down, "down", "ptzstop"), (btn_left, "left", "ptzstop"), 
            (btn_right, "right", "ptzstop"), (btn_zi, "zoomin", "zoomstop"), (btn_zo, "zoomout", "zoomstop"), 
            (btn_fi, "focusin", "focusstop"), (btn_fo, "focusout", "focusstop")
        ]:
            for widget in [btn, btn._canvas, btn._text_label]:
                widget.bind("<ButtonPress-1>", lambda e, c=cmd: self.manual_ptz(c))
                widget.bind("<ButtonRelease-1>", lambda e, sc=stop_cmd: self.manual_ptz(sc))

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
        if self.running_event.is_set(): return
            
        self.running_event.set()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        sel = self.selected_protocol.get()
        p_info = PROTOCOLS[sel]

        selected_cam_presets = [p for p, var in self.cam_preset_vars.items() if var.get()]
        
        use_cam = self.cam_enable_var.get()
        
        if not use_cam:
            print("[Warning] Camera disabled. Cannot start tracking.")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.running_event.clear()
            return
        
        global_track = self.global_live_tracking_var.get()
        mode1 = global_track
        
        total_time_cam1 = len(selected_cam_presets) * (p_info['cam'] + 5) if use_cam else 0
        self.t_end = time.perf_counter() + total_time_cam1

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.threads = []
        
        if use_cam:
            self.cam1_raw_q = queue.Queue(maxsize=60)
            self.cam1_ui_q = queue.Queue(maxsize=30)
            
            if mode1:
                self.cam1_inf_q = mp.Queue(maxsize=30)
                self.cam1_ann_q = mp.Queue(maxsize=150)
                self.cam1_cmd_q = mp.Queue()
                self.cam1_ann_frame_q = queue.Queue(maxsize=30)
                self.cam1_writer_q = queue.Queue(maxsize=150)
            else:
                self.cam1_inf_q = None
                self.cam1_ann_q = None
                self.cam1_cmd_q = None
                self.cam1_ann_frame_q = None
                self.cam1_writer_q = None
                
            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP, self.running_event, self.cam1_raw_q, self.cam1_inf_q, self.cam1_ann_frame_q, mode1)))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP, selected_cam_presets, p_info['cam'])))
            
            if mode1:
                self.ml_p1 = mp.Process(target=inference_worker, args=(self.cam1_inf_q, self.cam1_ann_q, self.cam1_cmd_q, timestamp_str, "Cam1"))
                self.ml_p1.daemon = True; self.ml_p1.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam_raw.mp4"), self.running_event, False)))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam1_ann_q, self.cam1_ann_frame_q, self.cam1_ui_q, self.cam1_writer_q, self.running_event)))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam1_writer_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam_annotated.mp4"), self.running_event)))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, self.cam1_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam_raw.mp4"), self.running_event, True)))

        for t in self.threads: t.start()

        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.update_gui_frame()

    def stop_tracking(self):
        if not self.running_event.is_set(): return
        self.running_event.clear()
        
        if hasattr(self, 'cam1_cmd_q') and self.cam1_cmd_q:
            self.cam1_cmd_q.put('STOP')

        # 1. CRITICAL FIX: Wait for background reader threads to stop generating frames
        if hasattr(self, 'threads'):
            for t in self.threads:
                if t.is_alive():
                    t.join(timeout=1.0)
                    
        # 2. Wait for ML processes to safely exit
        if hasattr(self, 'ml_p1') and self.ml_p1.is_alive():
            self.ml_p1.join(timeout=2.0)

        # 3. Now safely detach and purge pending SharedMemory scopes
        for q_name in ['cam1_inf_q', 'cam1_raw_q', 'cam1_ann_frame_q']:
            if hasattr(self, q_name) and getattr(self, q_name):
                q = getattr(self, q_name)
                while not q.empty():
                    try: 
                        item = q.get_nowait()
                        if isinstance(item, dict) and 'shm_name' in item:
                            try:
                                shm = shared_memory.SharedMemory(name=item['shm_name'])
                                shm.close()
                                shm.unlink()
                            except Exception: pass
                    except queue.Empty: break

        if hasattr(self, 'cam1_ann_q') and self.cam1_ann_q: self.cam1_ann_q.put(None)
        if hasattr(self, 'cam1_writer_q') and self.cam1_writer_q: self.cam1_writer_q.put(None)
        if hasattr(self, 'cam1_raw_q') and self.cam1_raw_q: self.cam1_raw_q.put(None)
        
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

    def manual_ptz(self, command):
        ip = CAMERA_IP
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}"
        self.ptz_cmd_queue.put(url)

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
            if hasattr(self, 'cam1_ui_q') and self.cam1_ui_q and not self.cam1_ui_q.empty():
                try: 
                    res1 = self.cam1_ui_q.get_nowait()
                    frame1, self.active_faces = res1
                except queue.Empty: pass
            
            if (now - self.last_fps_time) >= 1.0:
                self.sys_fps = self.frames_rendered / (now - self.last_fps_time)
                self.fps_lbl.configure(text=f"System Render FPS: {self.sys_fps:.1f}")
                self.faces_lbl.configure(text=f"Target Faces Detected: {self.active_faces}")
                self.frames_rendered = 0
                self.last_fps_time = now

            target_frame = frame1
            if target_frame is not None:
                self.frames_rendered += 1
                frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(1024, 576))
                self.video_frame.configure(image=imgtk, text="")
                self.video_frame.image = imgtk
                
        except Exception as e: pass
        finally: self.after(10, self.update_gui_frame)

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    app = AttendanceApp()
    app.mainloop()
