import os
import sys

# Auto-configure display for X11 / Tkinter if running in headless/remote terminal
if not os.environ.get("DISPLAY"):
    import glob
    for _sock in sorted(glob.glob("/tmp/.X11-unix/X*"), reverse=True):
        _disp = ":" + _sock.split("X")[-1]
        try:
            import subprocess
            if subprocess.run(["xset", "q"], env=dict(os.environ, DISPLAY=_disp), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                os.environ["DISPLAY"] = _disp
                break
        except Exception:
            pass
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":1"

if not os.environ.get("XAUTHORITY") and os.path.exists(f"/run/user/{os.getuid()}/gdm/Xauthority"):
    os.environ["XAUTHORITY"] = f"/run/user/{os.getuid()}/gdm/Xauthority"

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
from requests.auth import HTTPDigestAuth
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
META_PATH = './face_attendance_meta.pkl'

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
                                
                                mask = laplacian_var > 5.0
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
                        ui_queue.put_nowait((ui_frame, 0, []))
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

            recognized_names = []
            for i in range(len(ids)):
                box = boxes[i]
                t_id = ids[i]
                name = names[i]
                is_known = name not in ["Unknown", "Analyzing..."]
                if is_known:
                    recognized_names.append(name)
                color = (0, 255, 0) if is_known else (0, 0, 255)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, f"ID:{t_id} {name}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            if not writer_queue.full():
                writer_queue.put_nowait((frame, input_fps, skipped_frames + 1))
                
            if not ui_queue.full():
                ui_frame = cv2.resize(frame, (1024, 576))
                ui_queue.put_nowait((ui_frame, active_faces, recognized_names))
                
        except queue.Empty:
            if not running_event.is_set() and ann_queue.empty(): break

# ==========================================
# CONSOLE STREAM REDIRECTOR (THREAD-SAFE)
# ==========================================
class ConsoleRedirector:
    def __init__(self, log_func, original_stream):
        self.log_func = log_func
        self.original_stream = original_stream

    def write(self, message):
        if self.original_stream:
            try:
                self.original_stream.write(message)
                self.original_stream.flush()
            except Exception:
                pass
        if message:
            self.log_func(message)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass

# ==========================================
# FONT HELPERS (Linux/X11 anti-pixelation)
# ==========================================
def _F(size, weight="normal"):
    """Return a CTkFont with explicit Ubuntu/DejaVu Sans family to avoid bitmap fallback."""
    for family in ["Ubuntu", "DejaVu Sans", "Liberation Sans", "Noto Sans", "sans-serif"]:
        try:
            import tkinter.font as tkfont
            if family in tkfont.families():
                return ctk.CTkFont(family=family, size=size, weight=weight)
        except Exception:
            pass
    return ctk.CTkFont(size=size, weight=weight)

def _FM(size, weight="normal"):
    """Monospace font for timers, FPS, counters."""
    for family in ["JetBrains Mono", "Ubuntu Mono", "DejaVu Sans Mono", "Courier New"]:
        try:
            import tkinter.font as tkfont
            if family in tkfont.families():
                return ctk.CTkFont(family=family, size=size, weight=weight)
        except Exception:
            pass
    return ctk.CTkFont(size=size, weight=weight)

# ==========================================
# DESIGN TOKENS
# ==========================================
COLOR = {
    "bg":           "#0b0f17",
    "sidebar":      "#111827",
    "card":         "#1e293b",
    "border":       "#334155",
    "accent_blue":  "#3b82f6",
    "accent_green": "#10b981",
    "accent_red":   "#ef4444",
    "accent_amber": "#f59e0b",
    "accent_purple":"#a78bfa",
    "muted":        "#94a3b8",
    "text":         "#e2e8f0",
}


# ==========================================
# MAIN DESKTOP GUI
# ==========================================
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vision Sentinel | Single-Camera Real-Time Attendance")
        self.geometry("1680x940")
        self.minsize(1360, 760)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=COLOR["bg"])
        
        self.running_event = mp.Event()
        self.selected_protocol = ctk.IntVar(value=1)
        self.search_var = ctk.StringVar(value="")
        self.search_var.trace_add("write", lambda *args: self.refresh_roster_display(force=True))
        self._roster_visible = True
        
        self.t_end = 0
        self.sys_fps = 0
        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.active_faces = 0
        self.session_timestamp = ""

        # Load Student Roster
        self.all_students = self.load_student_roster()
        self.live_present_set = set()
        self.final_presence_map = {}
        self.final_counts_map = {}
        self.is_finalized = False
        self._last_roster_signature = None

        # Build full-name lookup: short_name -> full display name
        self.full_name_map = self._load_full_name_map()

        # Console logging queue & redirection setup
        self.log_queue = queue.Queue()
        self.autoscroll_var = ctk.BooleanVar(value=True)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self._enqueue_log, self._orig_stdout)
        sys.stderr = ConsoleRedirector(self._enqueue_log, self._orig_stderr)

        self.setup_ui()
        self.refresh_roster_display(force=True)
        self._sync_roster_to_tracking()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.ptz_cmd_queue = queue.Queue()
        self.ptz_worker_thread = threading.Thread(target=self.ptz_command_worker, daemon=True)
        self.ptz_worker_thread.start()

        self.after(50, self.process_log_queue)
        self.log("[SYSTEM] Single-Cam Attendance AI Engine initialized successfully.")
        self.log(f"[SYSTEM] Loaded {len(self.all_students)} enrolled students from roster.")
        self.log(f"[SYSTEM] Full-name map loaded: {len(self.full_name_map)} entries.")

    def _load_full_name_map(self):
        """Build {short_name -> 'Full Display Name'} from name_mapping.csv + filtered_true_attendance.csv.
        Only modifies GUI display; never touches backend short names or CSV exports."""
        full_name_map = {}
        try:
            mapping_path = './name_mapping.csv'
            attendance_path = './filtered_true_attendance.csv'
            if os.path.exists(mapping_path) and os.path.exists(attendance_path):
                nm = pd.read_csv(mapping_path, dtype=str).dropna(subset=['Roll_Name', 'Short_Name'])
                nm['Roll_Name'] = nm['Roll_Name'].str.strip()
                nm['Short_Name'] = nm['Short_Name'].str.strip()
                fa = pd.read_csv(attendance_path, dtype=str, usecols=['Roll No', 'Student Name'])
                fa['Roll No'] = fa['Roll No'].str.strip()
                fa['Student Name'] = fa['Student Name'].str.strip()
                merged = nm.merge(fa, left_on='Roll_Name', right_on='Roll No', how='left')
                for _, row in merged.iterrows():
                    short = row['Short_Name'].strip()
                    full = str(row.get('Student Name', '')).strip()
                    if short and full and full.lower() != 'nan':
                        full_name_map[short.lower()] = full.title()
                        full_name_map[short] = full.title()
        except Exception as e:
            print(f"[WARNING] Could not load full-name map: {e}")
        return full_name_map

    def _get_display_name(self, short_name):
        """Resolve a short internal name to its full display name for GUI roster only."""
        return self.full_name_map.get(short_name, self.full_name_map.get(short_name.lower(), short_name))

    def load_student_roster(self):
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, 'rb') as f:
                    data = pickle.load(f)
                names = sorted(list(set(data.get('target_names', []))))
                return names
            except Exception as e:
                print(f"[WARNING] Failed to load student roster: {e}")
        return []

    def _enqueue_log(self, text):
        if text:
            self.log_queue.put(text)

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message.strip()}\n"
        self._enqueue_log(formatted)

    def clear_logs(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def process_log_queue(self):
        lines = []
        while not self.log_queue.empty():
            try:
                lines.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if lines:
            try:
                self.log_textbox.configure(state="normal")
                for line in lines:
                    self.log_textbox.insert("end", line)
                line_count = int(float(self.log_textbox.index("end-1c").split(".")[0]))
                if line_count > 1000:
                    self.log_textbox.delete("1.0", f"{line_count - 800}.0")
                if self.autoscroll_var.get():
                    self.log_textbox.see("end")
                self.log_textbox.configure(state="disabled")
            except Exception:
                pass
        self.after(50, self.process_log_queue)

    def create_card(self, parent, title="", subtitle=None):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=COLOR["card"],
                            border_width=1, border_color=COLOR["border"])
        card.pack(fill="x", padx=10, pady=(0, 8))
        if title:
            ctk.CTkLabel(card, text=title.upper(), font=_F(10, "bold"),
                         text_color=COLOR["muted"]).pack(anchor="w", padx=10, pady=(8, 1))
        if subtitle:
            ctk.CTkLabel(card, text=subtitle, font=_F(9),
                         text_color=COLOR["muted"]).pack(anchor="w", padx=10, pady=(0, 3))
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="x", padx=10, pady=(2, 8))
        return content_frame

    def ptz_command_worker(self):
        while True:
            url = self.ptz_cmd_queue.get()
            if url == "QUIT": break
            try:
                requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=3)
            except Exception:
                pass

    def manual_ptz(self, action, code):
        url = f"http://{CAMERA_IP}/cgi-bin/ptz.cgi?action={action}&channel=1&code={code}&arg1=5&arg2=0&arg3=0"
        self.ptz_cmd_queue.put(url)

    def on_closing(self):
        print("Application shutting down... cleaning up memory and processes.")
        self.stop_tracking()
        if hasattr(self, 'ptz_cmd_queue'):
            self.ptz_cmd_queue.put("QUIT")
        if hasattr(self, '_orig_stdout') and self._orig_stdout:
            sys.stdout = self._orig_stdout
        if hasattr(self, '_orig_stderr') and self._orig_stderr:
            sys.stderr = self._orig_stderr
        self.destroy()

    def _toggle_roster_sidebar(self):
        """Manually collapse/expand the right roster sidebar."""
        if self._roster_visible:
            self.roster_sidebar.pack_forget()
            self._roster_visible = False
            if hasattr(self, 'toggle_roster_btn'):
                self.toggle_roster_btn.configure(text="▶ Show Roster")
        else:
            self.roster_sidebar.pack(side="right", fill="y", padx=0, pady=0, before=self.main_content_frame)
            self._roster_visible = True
            if hasattr(self, 'toggle_roster_btn'):
                self.toggle_roster_btn.configure(text="▼ Hide Roster")

    def _sync_roster_to_tracking(self):
        """Auto-show/hide roster sidebar based on Live Tracking toggle state."""
        tracking_on = self.global_live_tracking_var.get()
        if tracking_on and not self._roster_visible:
            self.roster_sidebar.pack(side="right", fill="y", padx=0, pady=0, before=self.main_content_frame)
            self._roster_visible = True
            if hasattr(self, 'toggle_roster_btn'):
                self.toggle_roster_btn.configure(text="▼ Hide Roster")
        elif not tracking_on and self._roster_visible:
            self.roster_sidebar.pack_forget()
            self._roster_visible = False
            if hasattr(self, 'toggle_roster_btn'):
                self.toggle_roster_btn.configure(text="▶ Show Roster")
        self.update_status_indicators()

    def update_status_indicators(self):
        is_running = self.running_event.is_set()
        cam_on = self.cam_enable_var.get()
        if is_running and cam_on:
            self.cam_status_dot.configure(text="●", text_color=COLOR["accent_green"])
            self.cam_status_state.configure(text="Streaming", text_color=COLOR["accent_green"])
        elif not cam_on:
            self.cam_status_dot.configure(text="●", text_color="#8b949e")
            self.cam_status_state.configure(text="Disabled", text_color="#8b949e")
        else:
            self.cam_status_dot.configure(text="●", text_color=COLOR["accent_red"])
            self.cam_status_state.configure(text="Offline", text_color=COLOR["accent_red"])

        tracking_on = self.global_live_tracking_var.get()
        if is_running:
            if tracking_on:
                self.ml_status_dot.configure(text="●", text_color=COLOR["accent_green"])
                self.ml_status_state.configure(text="Inferencing", text_color=COLOR["accent_green"])
            else:
                self.ml_status_dot.configure(text="●", text_color=COLOR["accent_amber"])
                self.ml_status_state.configure(text="Raw Ingest", text_color=COLOR["accent_amber"])
        else:
            self.ml_status_dot.configure(text="●", text_color="#8b949e")
            self.ml_status_state.configure(text="Standby", text_color="#8b949e")

    def setup_ui(self):
        # 3-Column Master Architecture:
        # Left Sidebar (Controls) | Center Content (HUD + Video + Logs) | Right Sidebar (Attendance Roster)
        self.sidebar_frame = ctk.CTkFrame(self, width=310, corner_radius=0, fg_color=COLOR["sidebar"])
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar_frame.pack_propagate(False)

        self.roster_sidebar = ctk.CTkFrame(self, width=310, corner_radius=0, fg_color=COLOR["sidebar"])
        self.roster_sidebar.pack(side="right", fill="y", padx=0, pady=0)
        self.roster_sidebar.pack_propagate(False)
        self._roster_visible = True

        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # =============================================================
        # LEFT SIDEBAR: Controls, PTZ & System
        # =============================================================
        self.brand_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.brand_frame.pack(fill="x", padx=14, pady=(14, 8))

        ctk.CTkLabel(self.brand_frame, text="VISION SENTINEL", font=_F(16, "bold"),
                     text_color=COLOR["text"]).pack(anchor="w")
        ctk.CTkLabel(self.brand_frame, text="SINGLE-CAM EDGE ML CORE", font=_F(9),
                     text_color=COLOR["muted"]).pack(anchor="w")

        # Telemetry Status Card
        self.status_card = ctk.CTkFrame(self.sidebar_frame, corner_radius=10,
                                        fg_color=COLOR["card"], border_width=1, border_color=COLOR["border"])
        self.status_card.pack(fill="x", padx=10, pady=(0, 8))

        status_grid = ctk.CTkFrame(self.status_card, fg_color="transparent")
        status_grid.pack(fill="x", padx=10, pady=8)
        status_grid.grid_columnconfigure((0, 1), weight=1)

        def _status_pill(row, col, title):
            box = ctk.CTkFrame(status_grid, corner_radius=6, fg_color="#0b0f17",
                               border_width=1, border_color=COLOR["border"])
            box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
            ctk.CTkLabel(box, text=title, font=_F(9, "bold"), text_color=COLOR["muted"]).pack(anchor="w", padx=6, pady=(3, 0))
            row_inner = ctk.CTkFrame(box, fg_color="transparent")
            row_inner.pack(anchor="w", padx=6, pady=(0, 3))
            dot = ctk.CTkLabel(row_inner, text="●", font=_F(9), text_color="#8b949e")
            dot.pack(side="left", padx=(0, 3))
            lbl = ctk.CTkLabel(row_inner, text="Standby", font=_F(10))
            lbl.pack(side="left")
            return dot, lbl

        self.cam_status_dot, self.cam_status_state = _status_pill(0, 0, "CAMERA 1")
        self.ml_status_dot, self.ml_status_state   = _status_pill(0, 1, "AI ENGINE")

        # Tab View
        self.tabview = ctk.CTkTabview(
            self.sidebar_frame, corner_radius=10, fg_color=COLOR["card"],
            segmented_button_selected_color=COLOR["accent_blue"],
            segmented_button_selected_hover_color=COLOR["accent_blue"]
        )
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.tab_controls = self.tabview.add("Controls")
        self.tab_ptz      = self.tabview.add("PTZ")
        self.tab_system   = self.tabview.add("System")

        self.controls_scroll = ctk.CTkScrollableFrame(self.tab_controls, fg_color="transparent")
        self.controls_scroll.pack(fill="both", expand=True, padx=0, pady=0)

        self.ptz_scroll = ctk.CTkScrollableFrame(self.tab_ptz, fg_color="transparent")
        self.ptz_scroll.pack(fill="both", expand=True, padx=0, pady=0)

        self.system_scroll = ctk.CTkScrollableFrame(self.tab_system, fg_color="transparent")
        self.system_scroll.pack(fill="both", expand=True, padx=0, pady=0)

        # Tab 1: Controls
        card_cam_setup = self.create_card(self.controls_scroll, "CAMERA INTERFACE", "Hardware RTSP stream switch")
        self.cam_enable_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(card_cam_setup, text=f"Cam 1  {CAMERA_IP}", variable=self.cam_enable_var,
                      font=_F(11), command=self.update_status_indicators).pack(pady=(4, 6), anchor="w")

        card_ml = self.create_card(self.controls_scroll, "AI RECOGNITION ENGINE", "Real-time tracking & attendance matching")
        self.global_live_tracking_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(card_ml, text="Live Tracking & Face Matching", variable=self.global_live_tracking_var,
                      font=_F(11), command=self._sync_roster_to_tracking).pack(pady=4, anchor="w")

        # Tab 2: PTZ & Presets
        card_proto = self.create_card(self.ptz_scroll, "EXECUTION PROTOCOL", "Preset dwell durations")
        proto_grid = ctk.CTkFrame(card_proto, fg_color="transparent")
        proto_grid.pack(fill="x")
        proto_grid.grid_columnconfigure((0, 1), weight=1)

        for idx, (p, info) in enumerate(PROTOCOLS.items()):
            r, c = divmod(idx, 2)
            rb_text = f"{info['name']} ({info['cam']}s)"
            ctk.CTkRadioButton(proto_grid, text=rb_text, variable=self.selected_protocol, value=p,
                               font=_F(10)).grid(row=r, column=c, pady=3, padx=2, sticky="w")

        card_p1 = self.create_card(self.ptz_scroll, "CAMERA PRESETS", f"Target waypoints ({CAMERA_IP})")
        self.preset_frame = ctk.CTkFrame(card_p1, fg_color="transparent")
        self.preset_frame.pack(fill="x")
        self.cam_preset_vars = {}
        for idx, p in enumerate(CAM_PRESETS):
            var = ctk.BooleanVar(value=True)
            self.cam_preset_vars[p] = var
            ctk.CTkCheckBox(self.preset_frame, text=f"P{p}", variable=var, font=_F(10), width=45
                            ).grid(row=idx // 4, column=idx % 4, padx=2, pady=3, sticky="w")

        card_ptz = self.create_card(self.ptz_scroll, "MANUAL PTZ", "Hold to move, release to stop")
        dpad_container = ctk.CTkFrame(card_ptz, corner_radius=8, fg_color="transparent")
        dpad_container.pack(fill="x", pady=(0, 6))
        dpad_container.grid_columnconfigure((0, 1, 2), weight=1)

        btn_up = ctk.CTkButton(dpad_container, text="▲", height=28, font=_F(13, "bold"))
        btn_left = ctk.CTkButton(dpad_container, text="◀", height=28, font=_F(13, "bold"))
        btn_center = ctk.CTkButton(dpad_container, text="PTZ", height=28, font=_F(10, "bold"), state="disabled")
        btn_right = ctk.CTkButton(dpad_container, text="▶", height=28, font=_F(13, "bold"))
        btn_down = ctk.CTkButton(dpad_container, text="▼", height=28, font=_F(13, "bold"))

        btn_up.grid(row=0, column=1, padx=3, pady=2, sticky="nsew")
        btn_left.grid(row=1, column=0, padx=3, pady=2, sticky="nsew")
        btn_center.grid(row=1, column=1, padx=3, pady=2, sticky="nsew")
        btn_right.grid(row=1, column=2, padx=3, pady=2, sticky="nsew")
        btn_down.grid(row=2, column=1, padx=3, pady=2, sticky="nsew")

        lens_container = ctk.CTkFrame(card_ptz, corner_radius=8, fg_color="transparent")
        lens_container.pack(fill="x")
        lens_container.grid_columnconfigure((0, 1), weight=1)

        btn_zi = ctk.CTkButton(lens_container, text="+ Zoom In", height=26, font=_F(10))
        btn_zo = ctk.CTkButton(lens_container, text="- Zoom Out", height=26, font=_F(10))
        btn_zi.grid(row=0, column=0, padx=3, pady=2, sticky="nsew")
        btn_zo.grid(row=0, column=1, padx=3, pady=2, sticky="nsew")

        for btn, code in [
            (btn_up, "Up"), (btn_down, "Down"), (btn_left, "Left"), 
            (btn_right, "Right"), (btn_zi, "ZoomTele"), (btn_zo, "ZoomWide")
        ]:
            for w in [btn, getattr(btn, "_canvas", None), getattr(btn, "_text_label", None)]:
                if w:
                    w.bind("<ButtonPress-1>", lambda e, c=code: self.manual_ptz("start", c))
                    w.bind("<ButtonRelease-1>", lambda e, c=code: self.manual_ptz("stop", c))

        # Tab 3: System
        card_sys_ui = self.create_card(self.system_scroll, "INTERFACE PREFERENCES")
        theme_selector = ctk.CTkSegmentedButton(card_sys_ui, values=["Dark", "Light", "System"],
                                                font=_F(10), command=lambda mode: ctk.set_appearance_mode(mode.lower()))
        theme_selector.set("Dark")
        theme_selector.pack(fill="x", pady=4)

        # Primary Call-To-Action buttons
        self.cta_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.cta_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.start_btn = ctk.CTkButton(
            self.cta_frame, text="Execute Bound Loop",
            command=self.start_tracking, height=36, corner_radius=8,
            font=_F(12, "bold"), fg_color=COLOR["accent_blue"], hover_color="#2563eb"
        )
        self.start_btn.pack(fill="x", pady=(0, 6))

        self.stop_btn = ctk.CTkButton(
            self.cta_frame, text="Force Stop System",
            command=self.stop_tracking, height=32, corner_radius=8,
            font=_F(11, "bold"), fg_color=COLOR["accent_red"], hover_color="#dc2626", state="disabled"
        )
        self.stop_btn.pack(fill="x")

        # =============================================================
        # CENTER CONTENT FRAME: Telemetry HUD + Video + Audit Trail
        # =============================================================
        hud_outer = ctk.CTkFrame(self.main_content_frame, fg_color="transparent")
        hud_outer.pack(fill="x", pady=(0, 6))
        hud_outer.grid_columnconfigure((0, 1, 2, 3), weight=1)
        hud_outer.grid_columnconfigure(4, weight=0)

        def _hud_tile(col, label, init_val, accent_color, padx=(0, 3)):
            card = ctk.CTkFrame(hud_outer, corner_radius=10, fg_color=COLOR["card"],
                                border_width=1, border_color=COLOR["border"])
            card.grid(row=0, column=col, padx=padx, pady=0, sticky="nsew")
            ctk.CTkLabel(card, text=label, font=_F(9, "bold"), text_color=COLOR["muted"]).pack(anchor="w", padx=10, pady=(6, 0))
            val_lbl = ctk.CTkLabel(card, text=init_val, font=_FM(15, "bold"), text_color=accent_color)
            val_lbl.pack(anchor="w", padx=10, pady=(0, 6))
            return val_lbl

        self.time_lbl   = _hud_tile(0, "REMAINING TIME",      "00:00:00",   COLOR["accent_blue"],  padx=(0, 3))
        self.fps_lbl    = _hud_tile(1, "RENDER SPEED",        "0.0 FPS",    COLOR["accent_green"], padx=(3, 3))
        self.faces_lbl  = _hud_tile(2, "ACTIVE FACES IN VIEW","0 Detected", COLOR["accent_amber"], padx=(3, 3))
        self.active_stream_lbl = _hud_tile(3, "ACTIVE CAMERA","Camera 1",   COLOR["accent_purple"],padx=(3, 3))

        # Toggle-roster button tile
        toggle_tile = ctk.CTkFrame(hud_outer, corner_radius=10, fg_color=COLOR["card"],
                                   border_width=1, border_color=COLOR["border"])
        toggle_tile.grid(row=0, column=4, padx=(3, 0), pady=0, sticky="nsew")
        self.toggle_roster_btn = ctk.CTkButton(
            toggle_tile, text="▼ Hide Roster", font=_F(9, "bold"),
            width=90, height=28, corner_radius=6,
            fg_color=COLOR["border"], hover_color=COLOR["card"],
            text_color=COLOR["muted"], command=self._toggle_roster_sidebar
        )
        self.toggle_roster_btn.pack(expand=True)

        # Video Viewport
        self.video_container = ctk.CTkFrame(self.main_content_frame, corner_radius=10,
                                            fg_color="#000000", border_width=1, border_color=COLOR["border"])
        self.video_container.pack(fill="both", expand=True, padx=0, pady=(0, 6))

        self.video_frame = ctk.CTkLabel(
            self.video_container,
            text="Camera ML Feed Offline\nPress [Execute Bound Loop] to begin real-time session",
            text_color="#6e7681", font=_F(14)
        )
        self.video_frame.pack(fill="both", expand=True)

        # Console & Diagnostic Audit Trail
        log_panel = ctk.CTkFrame(self.main_content_frame, corner_radius=10,
                                 fg_color=COLOR["card"], border_width=1, border_color=COLOR["border"])
        log_panel.pack(fill="x", padx=0, pady=0)

        log_head = ctk.CTkFrame(log_panel, fg_color="transparent")
        log_head.pack(fill="x", padx=10, pady=(6, 4))
        ctk.CTkLabel(log_head, text="SYSTEM CONSOLE & AUDIT TRAIL", font=_F(10, "bold"),
                     text_color=COLOR["muted"]).pack(side="left")

        ctk.CTkSwitch(log_head, text="Autoscroll", variable=self.autoscroll_var,
                      font=_F(10), height=18).pack(side="right", padx=(8, 0))
        ctk.CTkButton(log_head, text="Clear", width=50, height=20, font=_F(10),
                      fg_color=COLOR["border"], hover_color="#475569",
                      command=self.clear_logs).pack(side="right")

        self.log_textbox = ctk.CTkTextbox(
            log_panel, height=110, corner_radius=6,
            font=_FM(10), fg_color="#070a0f", text_color="#cbd5e1"
        )
        self.log_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_textbox.configure(state="disabled")

        # =============================================================
        # RIGHT SIDEBAR: Live & Final Attendance Roster
        # =============================================================
        roster_header = ctk.CTkFrame(self.roster_sidebar, fg_color="transparent")
        roster_header.pack(fill="x", padx=12, pady=(12, 6))

        ctk.CTkLabel(roster_header, text="CLASS ATTENDANCE", font=_F(13, "bold"),
                     text_color=COLOR["text"]).pack(side="left")

        self.roster_mode_badge = ctk.CTkLabel(
            roster_header, text="STANDBY", font=_F(9, "bold"),
            fg_color="#334155", text_color="#cbd5e1", corner_radius=6, padx=6, pady=2
        )
        self.roster_mode_badge.pack(side="right")

        # Roster Summary Card
        self.summary_card = ctk.CTkFrame(self.roster_sidebar, corner_radius=10,
                                         fg_color=COLOR["card"], border_width=1, border_color=COLOR["border"])
        self.summary_card.pack(fill="x", padx=10, pady=(0, 8))

        summary_head = ctk.CTkFrame(self.summary_card, fg_color="transparent")
        summary_head.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(summary_head, text="PRESENT IN CLASS", font=_F(9, "bold"),
                     text_color=COLOR["muted"]).pack(side="left")
        self.attendance_pct_lbl = ctk.CTkLabel(summary_head, text="0.0%", font=_FM(11, "bold"),
                                               text_color=COLOR["accent_green"])
        self.attendance_pct_lbl.pack(side="right")

        self.present_Big_lbl = ctk.CTkLabel(
            self.summary_card, text=f"0 / {len(self.all_students)} Students",
            font=_FM(18, "bold"), text_color=COLOR["accent_green"]
        )
        self.present_Big_lbl.pack(anchor="w", padx=10, pady=(0, 4))

        self.attendance_progress = ctk.CTkProgressBar(self.summary_card, height=6, corner_radius=3,
                                                      progress_color=COLOR["accent_green"])
        self.attendance_progress.pack(fill="x", padx=10, pady=(0, 6))
        self.attendance_progress.set(0)

        pills_row = ctk.CTkFrame(self.summary_card, fg_color="transparent")
        pills_row.pack(fill="x", padx=10, pady=(0, 6))
        pills_row.grid_columnconfigure((0, 1), weight=1)

        abs_pill = ctk.CTkFrame(pills_row, corner_radius=6, fg_color="#1e1218",
                                border_width=1, border_color=COLOR["accent_red"])
        abs_pill.grid(row=0, column=0, padx=(0, 3), sticky="nsew")
        self.absent_count_lbl = ctk.CTkLabel(abs_pill, text=f"Absent: {len(self.all_students)}",
                                             font=_F(10, "bold"), text_color=COLOR["accent_red"])
        self.absent_count_lbl.pack(pady=3)

        pres_pill = ctk.CTkFrame(pills_row, corner_radius=6, fg_color="#0d1f18",
                                 border_width=1, border_color=COLOR["accent_green"])
        pres_pill.grid(row=0, column=1, padx=(3, 0), sticky="nsew")
        self.present_count_lbl = ctk.CTkLabel(pres_pill, text="Present: 0",
                                              font=_F(10, "bold"), text_color=COLOR["accent_green"])
        self.present_count_lbl.pack(pady=3)

        # Search Filter
        search_frame = ctk.CTkFrame(self.roster_sidebar, fg_color="transparent")
        search_frame.pack(fill="x", padx=10, pady=(0, 4))
        self.search_entry = ctk.CTkEntry(
            search_frame, placeholder_text="Search by name...",
            textvariable=self.search_var, height=28, font=_F(11),
            fg_color=COLOR["card"], border_color=COLOR["border"], border_width=1
        )
        self.search_entry.pack(fill="x")

        # Unified Scrollable List: ABSENT FIRST, then PRESENT
        self.roster_scroll = ctk.CTkScrollableFrame(
            self.roster_sidebar, corner_radius=8,
            fg_color=COLOR["card"], border_width=1, border_color=COLOR["border"]
        )
        self.roster_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.update_status_indicators()

    def refresh_roster_display(self, force=False):
        """
        Renders the Attendance Sidebar with ALL ABSENT students first,
        followed by ALL PRESENT students, and updates total class counts.
        Uses self.full_name_map for GUI display only; internal names unchanged.
        """
        if not self.all_students:
            self.all_students = self.load_student_roster()

        if self.is_finalized and self.final_presence_map:
            present_set = {name for name, is_pres in self.final_presence_map.items() if is_pres}
        else:
            present_set = set(self.live_present_set)

        query = self.search_var.get().strip().lower()
        signature = (frozenset(present_set), self.is_finalized, query, len(self.all_students))
        if not force and signature == self._last_roster_signature:
            return
        self._last_roster_signature = signature

        total_count = len(self.all_students)
        present_all = sorted([s for s in self.all_students if s in present_set])
        absent_all  = sorted([s for s in self.all_students if s not in present_set])

        present_count = len(present_all)
        absent_count  = len(absent_all)
        pct = (present_count / total_count * 100.0) if total_count > 0 else 0.0

        # Update Top Summary Card
        self.present_Big_lbl.configure(text=f"{present_count} / {total_count} Students")
        self.attendance_pct_lbl.configure(text=f"{pct:.1f}%")
        self.attendance_progress.set(present_count / total_count if total_count > 0 else 0.0)
        self.absent_count_lbl.configure(text=f"Absent: {absent_count}")
        self.present_count_lbl.configure(text=f"Present: {present_count}")

        # Filter by search query — match against BOTH short name AND full display name
        def _matches(s):
            if not query:
                return True
            dn = self._get_display_name(s).lower()
            return query in s.lower() or query in dn

        absent_filtered  = [s for s in absent_all  if _matches(s)]
        present_filtered = [s for s in present_all if _matches(s)]

        # Clear existing items
        for child in self.roster_scroll.winfo_children():
            child.destroy()

        # 1. ABSENT SECTION HEADER
        abs_header = ctk.CTkFrame(self.roster_scroll, corner_radius=6, fg_color="#1e1218")
        abs_header.pack(fill="x", padx=2, pady=(2, 3))
        ctk.CTkLabel(abs_header, text=f"ABSENT  ({len(absent_filtered)})",
                     font=_F(10, "bold"), text_color=COLOR["accent_red"]).pack(side="left", padx=8, pady=4)

        if absent_filtered:
            for idx, s in enumerate(absent_filtered, 1):
                display_name = self._get_display_name(s)
                row = ctk.CTkFrame(self.roster_scroll, corner_radius=6, fg_color="transparent")
                row.pack(fill="x", padx=2, pady=1)
                ctk.CTkLabel(row, text="●", font=_F(9), text_color=COLOR["accent_red"]).pack(side="left", padx=(6, 3))
                ctk.CTkLabel(row, text=f"{idx}. {display_name}", font=_F(11), anchor="w",
                             text_color=COLOR["text"]).pack(side="left", fill="x", expand=True)
                ctk.CTkLabel(row, text="ABSENT", font=_F(8, "bold"),
                             text_color=COLOR["accent_red"]).pack(side="right", padx=5)
        else:
            ctk.CTkLabel(self.roster_scroll, text="All students present!", font=_F(10),
                         text_color=COLOR["accent_green"]).pack(pady=6)

        # 2. PRESENT SECTION HEADER
        pres_header = ctk.CTkFrame(self.roster_scroll, corner_radius=6, fg_color="#0d1f18")
        pres_header.pack(fill="x", padx=2, pady=(8, 3))
        ctk.CTkLabel(pres_header, text=f"PRESENT  ({len(present_filtered)})",
                     font=_F(10, "bold"), text_color=COLOR["accent_green"]).pack(side="left", padx=8, pady=4)

        if present_filtered:
            for idx, s in enumerate(present_filtered, 1):
                display_name = self._get_display_name(s)
                row = ctk.CTkFrame(self.roster_scroll, corner_radius=6, fg_color="transparent")
                row.pack(fill="x", padx=2, pady=1)
                ctk.CTkLabel(row, text="●", font=_F(9), text_color=COLOR["accent_green"]).pack(side="left", padx=(6, 3))
                ctk.CTkLabel(row, text=f"{idx}. {display_name}", font=_F(11, "bold"), anchor="w",
                             text_color=COLOR["text"]).pack(side="left", fill="x", expand=True)
                badge_text = "VERIFIED" if self.is_finalized else "LIVE"
                if self.is_finalized and s in self.final_counts_map and self.final_counts_map[s] > 0:
                    badge_text = f"✓ {self.final_counts_map[s]}"
                ctk.CTkLabel(row, text=badge_text, font=_F(8, "bold"),
                             text_color=COLOR["accent_green"]).pack(side="right", padx=5)
        else:
            ctk.CTkLabel(self.roster_scroll, text="No students marked present yet", font=_F(10),
                         text_color=COLOR["muted"]).pack(pady=6)

    def start_tracking(self):
        if self.running_event.is_set(): return
            
        self.running_event.set()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        self.all_students = self.load_student_roster()
        self.live_present_set.clear()
        self.final_presence_map.clear()
        self.final_counts_map.clear()
        self.is_finalized = False
        self.roster_mode_badge.configure(text="LIVE TRACKING", fg_color=COLOR["accent_blue"], text_color="#ffffff")
        self.refresh_roster_display(force=True)

        sel = self.selected_protocol.get()
        p_info = PROTOCOLS[sel]

        selected_cam_presets = [p for p, var in self.cam_preset_vars.items() if var.get()]
        
        use_cam = self.cam_enable_var.get()
        
        if not use_cam:
            print("[Warning] Camera disabled. Cannot start tracking.")
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.running_event.clear()
            self.update_status_indicators()
            return
        
        global_track = self.global_live_tracking_var.get()
        mode1 = global_track
        
        total_time_cam1 = len(selected_cam_presets) * (p_info['cam'] + 5) if use_cam else 0
        self.t_end = time.perf_counter() + total_time_cam1

        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_str = self.session_timestamp
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
        self.update_status_indicators()
        self.log(f"[SYSTEM] Session {self.session_timestamp} started ({p_info['name']}).")

    def stop_tracking(self):
        if not self.running_event.is_set(): return
        self.running_event.clear()
        self.log("[SYSTEM] Stopping tracking and computing verified attendance...")
        
        if hasattr(self, 'cam1_cmd_q') and self.cam1_cmd_q:
            self.cam1_cmd_q.put('STOP')

        # 1. Wait for background reader threads to stop generating frames
        if hasattr(self, 'threads'):
            for t in self.threads:
                if t.is_alive():
                    t.join(timeout=1.0)
                    
        # 2. Wait for ML processes to safely exit
        if hasattr(self, 'ml_p1') and self.ml_p1.is_alive():
            self.ml_p1.join(timeout=3.0)

        # 3. Read verified attendance CSV generated by inference_worker
        out_csv = os.path.join(RESULTS_DIR, f"{self.session_timestamp}_Cam1_output.csv")
        if os.path.exists(out_csv):
            try:
                df = pd.read_csv(out_csv)
                self.final_presence_map = dict(zip(df['Name'], df['Status'] == 'Present'))
                self.final_counts_map = dict(zip(df['Name'], df['Detection Count']))
                self.is_finalized = True
                self.roster_mode_badge.configure(text="FINAL VERIFIED", fg_color=COLOR["accent_green"], text_color="#ffffff")
                pres_count = sum(1 for v in self.final_presence_map.values() if v)
                self.log(f"[ROSTER] Final attendance verified: {pres_count}/{len(self.all_students)} Present. Saved to {out_csv}")
            except Exception as e:
                self.log(f"[WARNING] Failed reading final attendance CSV: {e}")
                self.roster_mode_badge.configure(text="SESSION ENDED", fg_color=COLOR["accent_red"], text_color="#ffffff")
        else:
            self.roster_mode_badge.configure(text="SESSION ENDED", fg_color=COLOR["accent_red"], text_color="#ffffff")

        self.refresh_roster_display(force=True)

        # 4. Safely detach and purge pending SharedMemory scopes
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
        self.update_status_indicators()

    def ptz_runner(self, ip, presets, duration):
        for preset in presets:
            if not self.running_event.is_set(): break
            url = f"http://{ip}/cgi-bin/ptz.cgi?action=start&channel=1&code=GotoPreset&arg1=0&arg2={preset}&arg3=0"
            try: requests.get(url, auth=HTTPDigestAuth(USERNAME, PASSWORD), timeout=5)
            except: pass
            
            t_mechanical = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_mechanical) < 5.0:
                time.sleep(0.1)
                
            t_start = time.perf_counter()
            while self.running_event.is_set() and (time.perf_counter() - t_start) < duration:
                time.sleep(0.1)

    def update_gui_frame(self):
        if not self.running_event.is_set():
            self.video_frame.configure(image=None, text="Camera ML Feed Offline\nPress [Execute Bound Loop] to begin real-time session")
            return
            
        try:
            now = time.perf_counter()
            remaining = max(0, int(self.t_end - now))
            
            if remaining <= 0:
                self.stop_tracking()
                return
                
            h, rem = divmod(remaining, 3600)
            m, s = divmod(rem, 60)
            self.time_lbl.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
            
            newly_detected = False
            frame1 = None
            if hasattr(self, 'cam1_ui_q') and self.cam1_ui_q and not self.cam1_ui_q.empty():
                try: 
                    res1 = self.cam1_ui_q.get_nowait()
                    if len(res1) == 3:
                        frame1, self.active_faces, names1 = res1
                        for n in names1:
                            if n not in self.live_present_set:
                                self.live_present_set.add(n)
                                newly_detected = True
                    else:
                        frame1, self.active_faces = res1
                except queue.Empty: pass

            if newly_detected:
                self.refresh_roster_display()
            
            if (now - self.last_fps_time) >= 1.0:
                self.sys_fps = self.frames_rendered / (now - self.last_fps_time)
                self.fps_lbl.configure(text=f"{self.sys_fps:.1f} FPS")
                self.faces_lbl.configure(text=f"{self.active_faces} Detected")
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
