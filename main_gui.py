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
from PIL import Image
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
# MULTIPROCESSING INFERENCE PROCESS
# ==========================================
def inference_worker(inf_queue, ann_queue, cmd_queue, result_queue, timestamp_str, cam_name):
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
            
        if os.path.exists(META_PATH):
            with open(META_PATH, 'rb') as f:
                saved_data = pickle.load(f)
            target_names = saved_data['target_names']
            y_real = saved_data['y_real']
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
            frame_tensor = torch.from_numpy(frame_array).to(device, non_blocking=True).float()
            
            input_fps = item['fps']
            skipped_frames = item['skipped_frames']
            
            frame_count += (skipped_frames + 1)
            fps = input_fps if input_fps > 0 else 30
            
            metadata = {'boxes': [], 'ids': [], 'names': []}
            current_active_faces = 0

            if ref_embeddings_tensor is not None and len(target_names) > 0:
                results = yolo_model.track(frame_array, persist=True, tracker="custom_bytetrack.yaml", verbose=False, quantize=16 if use_half else None, imgsz=640)
                has_detections = results[0].boxes.id is not None
                
                if has_detections:
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
                            
                            frame_tensor_chw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
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
                                batch_tensor = torch.cat(batch_tensors, dim=0)
                                batch_tensor = batch_tensor[:, [2, 1, 0], :, :]
                                
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

        try:
            ann_queue.put((metadata['boxes'], metadata['ids'], metadata['names'], skipped_frames, current_active_faces, fps), timeout=0.2)
        except queue.Full:
            pass

    # ==========================
    # FINAL ATTENDANCE SECURE DUMP
    # ==========================
    print(f"[ML PROCESS | {cam_name}] Shutting down natively... Generating Attendance Logs.")
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

    if result_queue is not None:
        try:
            result_queue.put({
                'cam': cam_name,
                'presence': student_presence,
                'counts': student_detection_count
            }, timeout=1.0)
        except Exception:
            pass


# ==========================================
# BACKGROUND OUTPUT AND INGESTION THREADS
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
        
        if not raw_queue.full():
            try:
                shm_raw = shared_memory.SharedMemory(create=True, size=frame.nbytes)
                np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm_raw.buf)[:] = frame[:]
                raw_queue.put_nowait({'shm_name': shm_raw.name, 'shape': frame.shape, 'dtype': frame.dtype, 'fps': fps, 'skipped_frames': skipped_frames + 1})
                shm_raw.close()
            except Exception: pass
            
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
                        ui_queue.put_nowait((frame.copy(), 0, []))
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
                ui_queue.put_nowait((frame, active_faces, recognized_names))
                
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
# MAIN DESKTOP GUI
# ==========================================
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
    "bg":          "#0b0f17",
    "sidebar":     "#111827",
    "card":        "#1e293b",
    "border":      "#334155",
    "accent_blue": "#3b82f6",
    "accent_green":"#10b981",
    "accent_red":  "#ef4444",
    "accent_amber":"#f59e0b",
    "accent_purple":"#a78bfa",
    "muted":       "#94a3b8",
    "text":        "#e2e8f0",
}


class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vision Sentinel Core | Real-Time Dual-Camera Attendance AI")
        self.geometry("1680x940")
        self.minsize(1360, 760)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=COLOR["bg"])
        
        self.running_event = mp.Event()
        self.selected_protocol = ctk.IntVar(value=1)
        self.view_target = ctk.StringVar(value="Camera 1") 
        self.search_var = ctk.StringVar(value="")
        self.search_var.trace_add("write", lambda *args: self.refresh_roster_display(force=True))
        self._roster_visible = True
        
        self.t_end = 0
        self.sys_fps = 0
        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.cam1_active_faces = 0
        self.cam2_active_faces = 0
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
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.ptz_cmd_queue = queue.Queue()
        self.ptz_worker_thread = threading.Thread(target=self.ptz_command_worker, daemon=True)
        self.ptz_worker_thread.start()

        self.after(50, self.process_log_queue)
        self.log("[SYSTEM] Attendance AI Core initialized successfully.")
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

    def ptz_command_worker(self):
        while True:
            url = self.ptz_cmd_queue.get()
            if url == "QUIT": break
            try:
                requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=3)
            except Exception:
                pass
        
    def on_closing(self):
        self.stop_tracking()
        if hasattr(self, 'ptz_cmd_queue'):
            self.ptz_cmd_queue.put("QUIT")
        if hasattr(self, '_orig_stdout') and self._orig_stdout:
            sys.stdout = self._orig_stdout
        if hasattr(self, '_orig_stderr') and self._orig_stderr:
            sys.stderr = self._orig_stderr
        self.destroy()

    def create_card(self, parent, title="", subtitle=None):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color=COLOR["card"],
                            border_width=1, border_color=COLOR["border"])
        card.pack(fill="x", padx=4, pady=4)
        
        if title:
            header_frame = ctk.CTkFrame(card, fg_color="transparent")
            header_frame.pack(fill="x", padx=10, pady=(8, 2))
            lbl = ctk.CTkLabel(header_frame, text=title, font=_F(10, "bold"),
                               text_color=COLOR["muted"])
            lbl.pack(side="left", anchor="w")
            if subtitle:
                sub = ctk.CTkLabel(card, text=subtitle, font=_F(10),
                                   text_color=COLOR["muted"])
                sub.pack(anchor="w", padx=10, pady=(0, 3))

        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="x", padx=10, pady=(2, 8))
        return content_frame

    def toggle_all_presets(self, cam_num, select=True):
        vars_dict = self.cam1_preset_vars if cam_num == 1 else self.cam2_preset_vars
        for var in vars_dict.values():
            var.set(select)

    def on_view_target_change(self, value=None):
        target = self.view_target.get()
        if hasattr(self, 'active_stream_lbl'):
            self.active_stream_lbl.configure(text=target)
        if hasattr(self, 'ptz_target_lbl'):
            self.ptz_target_lbl.configure(text=f"Directing: {target}")
        disp_faces = self.cam1_active_faces if target == "Camera 1" else self.cam2_active_faces
        if hasattr(self, 'faces_lbl'):
            self.faces_lbl.configure(text=f"{disp_faces} Detected")

    def _toggle_roster_sidebar(self):
        """Manually collapse/expand the right roster sidebar."""
        if self._roster_visible:
            self.roster_sidebar.pack_forget()
            self._roster_visible = False
            if hasattr(self, 'toggle_roster_btn'):
                self.toggle_roster_btn.configure(text="▶ Show Roster")
        else:
            # Re-pack sidebar before main content so center expands/shrinks properly
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
        cam1_on = self.cam1_enable_var.get()
        if is_running and cam1_on:
            self.cam1_status_dot.configure(text="●", text_color=COLOR["accent_green"])
            self.cam1_status_state.configure(text="Streaming", text_color=COLOR["accent_green"])
        elif not cam1_on:
            self.cam1_status_dot.configure(text="●", text_color="#8b949e")
            self.cam1_status_state.configure(text="Disabled", text_color="#8b949e")
        else:
            self.cam1_status_dot.configure(text="●", text_color=COLOR["accent_red"])
            self.cam1_status_state.configure(text="Offline", text_color=COLOR["accent_red"])

        cam2_on = self.cam2_enable_var.get()
        if is_running and cam2_on:
            self.cam2_status_dot.configure(text="●", text_color=COLOR["accent_green"])
            self.cam2_status_state.configure(text="Streaming", text_color=COLOR["accent_green"])
        elif not cam2_on:
            self.cam2_status_dot.configure(text="●", text_color="#8b949e")
            self.cam2_status_state.configure(text="Disabled", text_color="#8b949e")
        else:
            self.cam2_status_dot.configure(text="●", text_color=COLOR["accent_red"])
            self.cam2_status_state.configure(text="Offline", text_color=COLOR["accent_red"])

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
        # Left Sidebar (Controls) | Center Content (HUD + Video + Logs) | Right Sidebar (Live & Final Attendance Roster)
        self.sidebar_frame = ctk.CTkFrame(self, width=310, corner_radius=0, fg_color=COLOR["sidebar"])
        self.sidebar_frame.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar_frame.pack_propagate(False)

        # Roster sidebar starts visible (live tracking is ON by default)
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
        ctk.CTkLabel(self.brand_frame, text="Dual-Camera AI Pipeline", font=_F(10),
                     text_color=COLOR["muted"]).pack(anchor="w")

        # System Status Pill Grid
        self.status_card = ctk.CTkFrame(self.sidebar_frame, corner_radius=10,
                                        fg_color=COLOR["card"], border_width=1, border_color=COLOR["border"])
        self.status_card.pack(fill="x", padx=10, pady=(0, 6))

        status_grid = ctk.CTkFrame(self.status_card, fg_color="transparent")
        status_grid.pack(fill="x", padx=6, pady=6)
        status_grid.grid_columnconfigure((0, 1, 2), weight=1)

        cam1_status_box = ctk.CTkFrame(status_grid, corner_radius=6, fg_color="transparent")
        cam1_status_box.grid(row=0, column=0, padx=2, sticky="nsew")
        cam1_dot_box = ctk.CTkFrame(cam1_status_box, fg_color="transparent")
        cam1_dot_box.pack()
        self.cam1_status_dot = ctk.CTkLabel(cam1_dot_box, text="●", font=_F(11), text_color=COLOR["accent_red"])
        self.cam1_status_dot.pack(side="left", padx=2)
        ctk.CTkLabel(cam1_dot_box, text="Cam 1", font=_F(10, "bold"), text_color=COLOR["text"]).pack(side="left")
        self.cam1_status_state = ctk.CTkLabel(cam1_status_box, text="Offline", font=_F(10), text_color=COLOR["accent_red"])
        self.cam1_status_state.pack()

        cam2_status_box = ctk.CTkFrame(status_grid, corner_radius=6, fg_color="transparent")
        cam2_status_box.grid(row=0, column=1, padx=2, sticky="nsew")
        cam2_dot_box = ctk.CTkFrame(cam2_status_box, fg_color="transparent")
        cam2_dot_box.pack()
        self.cam2_status_dot = ctk.CTkLabel(cam2_dot_box, text="●", font=_F(11), text_color=COLOR["accent_red"])
        self.cam2_status_dot.pack(side="left", padx=2)
        ctk.CTkLabel(cam2_dot_box, text="Cam 2", font=_F(10, "bold"), text_color=COLOR["text"]).pack(side="left")
        self.cam2_status_state = ctk.CTkLabel(cam2_status_box, text="Offline", font=_F(10), text_color=COLOR["accent_red"])
        self.cam2_status_state.pack()

        ml_status_box = ctk.CTkFrame(status_grid, corner_radius=6, fg_color="transparent")
        ml_status_box.grid(row=0, column=2, padx=2, sticky="nsew")
        ml_dot_box = ctk.CTkFrame(ml_status_box, fg_color="transparent")
        ml_dot_box.pack()
        self.ml_status_dot = ctk.CTkLabel(ml_dot_box, text="●", font=_F(11), text_color="#8b949e")
        self.ml_status_dot.pack(side="left", padx=2)
        ctk.CTkLabel(ml_dot_box, text="ML", font=_F(10, "bold"), text_color=COLOR["text"]).pack(side="left")
        self.ml_status_state = ctk.CTkLabel(ml_status_box, text="Standby", font=_F(10), text_color="#8b949e")
        self.ml_status_state.pack()

        # Sidebar Tabs
        self.tabview = ctk.CTkTabview(self.sidebar_frame, corner_radius=8,
                                      fg_color=COLOR["card"], border_color=COLOR["border"], border_width=1)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(0, 6))

        tab_controls = self.tabview.add("Controls")
        tab_ptz = self.tabview.add("PTZ & Presets")
        tab_system = self.tabview.add("System")

        self.controls_scroll = ctk.CTkScrollableFrame(tab_controls, fg_color="transparent")
        self.controls_scroll.pack(fill="both", expand=True)

        self.ptz_scroll = ctk.CTkScrollableFrame(tab_ptz, fg_color="transparent")
        self.ptz_scroll.pack(fill="both", expand=True)

        self.system_scroll = ctk.CTkScrollableFrame(tab_system, fg_color="transparent")
        self.system_scroll.pack(fill="both", expand=True)

        # Tab 1: Controls
        card_viewport = self.create_card(self.controls_scroll, "ACTIVE VIEWPORT", "Switch live camera preview")
        self.view_toggle = ctk.CTkSegmentedButton(
            card_viewport,
            values=["Camera 1", "Camera 2"],
            variable=self.view_target,
            font=_F(12, "bold"),
            command=self.on_view_target_change
        )
        self.view_toggle.pack(fill="x", pady=4)

        card_cam_setup = self.create_card(self.controls_scroll, "CAMERA STREAMS", "Enable or disable RTSP feeds")
        self.cam1_enable_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(card_cam_setup, text=f"Cam 1  {CAMERA_IP_1}", variable=self.cam1_enable_var,
                      font=_F(11), command=self.update_status_indicators).pack(pady=(4, 6), anchor="w")
        self.cam2_enable_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(card_cam_setup, text=f"Cam 2  {CAMERA_IP_2}", variable=self.cam2_enable_var,
                      font=_F(11), command=self.update_status_indicators).pack(pady=(0, 4), anchor="w")

        card_ml = self.create_card(self.controls_scroll, "AI RECOGNITION ENGINE", "Real-time tracking & attendance matching")
        self.global_live_tracking_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(card_ml, text="Live Tracking & Face Matching", variable=self.global_live_tracking_var,
                      font=_F(11), command=self._sync_roster_to_tracking).pack(pady=4, anchor="w")

        # Tab 2: PTZ & Presets
        card_proto = self.create_card(self.ptz_scroll, "EXECUTION PROTOCOL", "Preset dwell durations (Cam 1 / Cam 2)")
        proto_grid = ctk.CTkFrame(card_proto, fg_color="transparent")
        proto_grid.pack(fill="x")
        proto_grid.grid_columnconfigure((0, 1), weight=1)

        for idx, (p, info) in enumerate(PROTOCOLS.items()):
            r, c = divmod(idx, 2)
            rb_text = f"{info['name']} ({info['cam1']}s/{info['cam2']}s)"
            ctk.CTkRadioButton(proto_grid, text=rb_text, variable=self.selected_protocol, value=p,
                               font=_F(10)).grid(row=r, column=c, pady=3, padx=2, sticky="w")

        card_presets = self.create_card(self.ptz_scroll, "CAMERA PRESETS", "Active tour waypoints")
        cam1_header = ctk.CTkFrame(card_presets, fg_color="transparent")
        cam1_header.pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(cam1_header, text="Cam 1 (1-8):", font=_F(10, "bold"), text_color=COLOR["text"]).pack(side="left")
        ctk.CTkButton(cam1_header, text="All", width=34, height=18, font=_F(10),
                      command=lambda: self.toggle_all_presets(1, True)).pack(side="right", padx=(2, 0))
        ctk.CTkButton(cam1_header, text="None", width=38, height=18, font=_F(10),
                      command=lambda: self.toggle_all_presets(1, False)).pack(side="right")

        self.preset_frame1 = ctk.CTkFrame(card_presets, fg_color="transparent")
        self.preset_frame1.pack(fill="x", pady=(2, 4))
        self.preset_frame1.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.cam1_preset_vars = {}
        for idx, p in enumerate(CAM1_PRESETS):
            var = ctk.BooleanVar(value=True)
            self.cam1_preset_vars[p] = var
            ctk.CTkCheckBox(self.preset_frame1, text=f"P{p}", variable=var, font=_F(10), width=45
                            ).grid(row=idx // 4, column=idx % 4, padx=2, pady=3, sticky="w")

        cam2_header = ctk.CTkFrame(card_presets, fg_color="transparent")
        cam2_header.pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(cam2_header, text="Cam 2 (1-6):", font=_F(10, "bold"), text_color=COLOR["text"]).pack(side="left")
        ctk.CTkButton(cam2_header, text="All", width=34, height=18, font=_F(10),
                      command=lambda: self.toggle_all_presets(2, True)).pack(side="right", padx=(2, 0))
        ctk.CTkButton(cam2_header, text="None", width=38, height=18, font=_F(10),
                      command=lambda: self.toggle_all_presets(2, False)).pack(side="right")

        self.preset_frame2 = ctk.CTkFrame(card_presets, fg_color="transparent")
        self.preset_frame2.pack(fill="x", pady=(2, 4))
        self.preset_frame2.grid_columnconfigure((0, 1, 2), weight=1)
        self.cam2_preset_vars = {}
        for idx, p in enumerate(CAM2_PRESETS):
            var = ctk.BooleanVar(value=True)
            self.cam2_preset_vars[p] = var
            ctk.CTkCheckBox(self.preset_frame2, text=f"P{p}", variable=var, font=_F(10), width=45
                            ).grid(row=idx // 3, column=idx % 3, padx=2, pady=3, sticky="w")

        card_ptz = self.create_card(self.ptz_scroll, "MANUAL PTZ", "Hold to move, release to stop")
        self.ptz_target_lbl = ctk.CTkLabel(card_ptz, text=f"Directing: {self.view_target.get()}",
                                           font=_F(10, "bold"), text_color=COLOR["accent_blue"])
        self.ptz_target_lbl.pack(anchor="w", pady=(0, 4))

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

        btn_fi = ctk.CTkButton(lens_container, text="+ Focus Near", height=26, font=_F(10))
        btn_fo = ctk.CTkButton(lens_container, text="- Focus Far", height=26, font=_F(10))
        btn_fi.grid(row=1, column=0, padx=3, pady=2, sticky="nsew")
        btn_fo.grid(row=1, column=1, padx=3, pady=2, sticky="nsew")

        for btn, cmd, stop_cmd in [
            (btn_up, "up", "ptzstop"), (btn_down, "down", "ptzstop"),
            (btn_left, "left", "ptzstop"), (btn_right, "right", "ptzstop"),
            (btn_zi, "zoomin", "zoomstop"), (btn_zo, "zoomout", "zoomstop"),
            (btn_fi, "focusin", "focusstop"), (btn_fo, "focusout", "focusstop")
        ]:
            for w in [btn, getattr(btn, "_canvas", None), getattr(btn, "_text_label", None)]:
                if w:
                    w.bind("<ButtonPress-1>", lambda e, c=cmd: self.manual_ptz(c))
                    w.bind("<ButtonRelease-1>", lambda e, sc=stop_cmd: self.manual_ptz(sc))

        # Tab 3: System
        card_sys_ui = self.create_card(self.system_scroll, "INTERFACE PREFERENCES")
        theme_selector = ctk.CTkSegmentedButton(card_sys_ui, values=["Dark", "Light", "System"],
                                                font=_F(10), command=lambda mode: ctk.set_appearance_mode(mode.lower()))
        theme_selector.set("Dark")
        theme_selector.pack(fill="x", pady=4)

        # Bottom Call-to-Action Buttons
        self.cta_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.cta_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        self.start_btn = ctk.CTkButton(
            self.cta_frame,
            text="▶  START LIVE TRACKING",
            font=_F(13, "bold"),
            height=42,
            fg_color=COLOR["accent_green"],
            hover_color="#059669",
            text_color="#ffffff",
            corner_radius=10,
            command=self.start_tracking
        )
        self.start_btn.pack(fill="x", pady=(0, 6))

        self.stop_btn = ctk.CTkButton(
            self.cta_frame,
            text="■  STOP & FINALIZE ROSTER",
            font=_F(11, "bold"),
            height=36,
            fg_color="transparent",
            border_width=2,
            border_color=COLOR["accent_red"],
            hover_color="#b91c1c",
            text_color=COLOR["accent_red"],
            corner_radius=10,
            command=self.stop_tracking
        )
        self.stop_btn.pack(fill="x")
        self.stop_btn.configure(state="disabled")

        # =============================================================
        # CENTER PANEL: Telemetry HUD + Video Feed + Console Logs
        # =============================================================
        # HUD bar: 4 tiles + toggle roster button on far right
        hud_outer = ctk.CTkFrame(self.main_content_frame, fg_color="transparent")
        hud_outer.pack(fill="x", padx=0, pady=(0, 6))
        hud_outer.grid_columnconfigure((0, 1, 2, 3), weight=1)
        hud_outer.grid_columnconfigure(4, weight=0)

        self.hud_frame = hud_outer

        def _hud_tile(col, label_text, value_text, value_color, padx=(0, 4)):
            t = ctk.CTkFrame(hud_outer, corner_radius=10, fg_color=COLOR["card"],
                             border_width=1, border_color=COLOR["border"])
            t.grid(row=0, column=col, padx=padx, pady=0, sticky="nsew")
            ctk.CTkLabel(t, text=label_text, font=_F(9, "bold"),
                         text_color=COLOR["muted"]).pack(pady=(7, 0), padx=10, anchor="w")
            lbl = ctk.CTkLabel(t, text=value_text, font=_FM(18, "bold"), text_color=value_color)
            lbl.pack(pady=(0, 7), padx=10, anchor="w")
            return lbl

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
            text="Camera Feed Standby\n\nClick  ▶ START LIVE TRACKING  to begin classroom attendance monitoring",
            font=_F(13, "bold"),
            text_color=COLOR["muted"],
            fg_color="black"
        )
        self.video_frame.pack(fill="both", expand=True, padx=2, pady=2)

        # Bottom Console Logs
        self.log_container = ctk.CTkFrame(self.main_content_frame, corner_radius=10, height=120,
                                          fg_color=COLOR["card"], border_width=1, border_color=COLOR["border"])
        self.log_container.pack(fill="x", padx=0, pady=0)
        self.log_container.pack_propagate(False)

        log_header = ctk.CTkFrame(self.log_container, fg_color="transparent")
        log_header.pack(fill="x", padx=10, pady=(5, 2))
        ctk.CTkLabel(log_header, text="SYSTEM PIPELINE CONSOLE", font=_F(9, "bold"),
                     text_color=COLOR["muted"]).pack(side="left")
        ctk.CTkButton(log_header, text="Clear", width=50, height=18, font=_F(9),
                      command=self.clear_logs).pack(side="right", padx=(6, 0))
        ctk.CTkCheckBox(log_header, text="Auto-scroll", variable=self.autoscroll_var,
                        font=_F(9), checkbox_width=13, checkbox_height=13).pack(side="right", padx=6)

        self.log_textbox = ctk.CTkTextbox(self.log_container, font=_FM(10), wrap="word", corner_radius=6)
        self.log_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self.log_textbox.configure(state="disabled")

        # =============================================================
        # RIGHT SIDEBAR: CLASS ATTENDANCE ROSTER (ABSENT FIRST, THEN PRESENT)
        # =============================================================
        roster_header = ctk.CTkFrame(self.roster_sidebar, fg_color="transparent")
        roster_header.pack(fill="x", padx=12, pady=(12, 4))

        title_row = ctk.CTkFrame(roster_header, fg_color="transparent")
        title_row.pack(fill="x")
        ctk.CTkLabel(title_row, text="CLASS ATTENDANCE", font=_F(14, "bold"),
                     text_color=COLOR["text"]).pack(side="left")

        self.roster_mode_badge = ctk.CTkLabel(
            title_row, text="STANDBY", font=_F(9, "bold"),
            fg_color=COLOR["border"], text_color=COLOR["muted"],
            corner_radius=6, padx=7, pady=2
        )
        self.roster_mode_badge.pack(side="right")

        # Summary Counter Banner
        self.summary_card = ctk.CTkFrame(self.roster_sidebar, corner_radius=10,
                                         fg_color=COLOR["card"], border_width=1, border_color=COLOR["border"])
        self.summary_card.pack(fill="x", padx=10, pady=(2, 6))

        count_top = ctk.CTkFrame(self.summary_card, fg_color="transparent")
        count_top.pack(fill="x", padx=10, pady=(8, 2))
        ctk.CTkLabel(count_top, text="PRESENT IN CLASS", font=_F(9, "bold"),
                     text_color=COLOR["muted"]).pack(side="left")
        self.attendance_pct_lbl = ctk.CTkLabel(count_top, text="0.0%", font=_F(10, "bold"),
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

        # 1. ABSENT SECTION HEADER (Shown First)
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

        # 2. PRESENT SECTION HEADER (Shown Second)
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
        
        # Reset live & final roster state for new session
        self.all_students = self.load_student_roster()
        self.live_present_set.clear()
        self.final_presence_map.clear()
        self.final_counts_map.clear()
        self.is_finalized = False
        self.roster_mode_badge.configure(text="LIVE TRACKING", fg_color=COLOR["accent_blue"], text_color="#ffffff")
        self.refresh_roster_display(force=True)

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
            self.update_status_indicators()
            return
        
        global_track = self.global_live_tracking_var.get()
        mode1 = global_track
        mode2 = global_track
        
        total_time_cam1 = len(selected_cam1_presets) * (p_info['cam1'] + 5) if use_cam1 else 0
        total_time_cam2 = len(selected_cam2_presets) * (p_info['cam2'] + 5) if use_cam2 else 0
        self.t_end = time.perf_counter() + max(total_time_cam1, total_time_cam2)

        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.threads = []
        self.result_q = mp.Queue()
        
        if use_cam1:
            self.cam1_raw_q = queue.Queue(maxsize=60)
            self.cam1_ui_q = queue.Queue(maxsize=30)
            
            if mode1:
                self.cam1_inf_q = mp.Queue(maxsize=30)
                self.cam1_ann_q = mp.Queue(maxsize=150)
                self.cam1_cmd_q = mp.Queue()
                self.cam1_ann_frame_q = queue.Queue(maxsize=30)
                self.cam1_writer_q = queue.Queue(maxsize=150)
            else:
                self.cam1_inf_q = self.cam1_ann_q = self.cam1_cmd_q = self.cam1_ann_frame_q = self.cam1_writer_q = None
                
            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_1, self.running_event, self.cam1_raw_q, self.cam1_inf_q, self.cam1_ann_frame_q, mode1)))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_1, selected_cam1_presets, p_info['cam1'])))
            
            if mode1:
                self.ml_p1 = mp.Process(target=inference_worker, args=(self.cam1_inf_q, self.cam1_ann_q, self.cam1_cmd_q, self.result_q, self.session_timestamp, "Cam1"))
                self.ml_p1.daemon = True; self.ml_p1.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{self.session_timestamp}_cam1_raw.mp4"), self.running_event, False)))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam1_ann_q, self.cam1_ann_frame_q, self.cam1_ui_q, self.cam1_writer_q, self.running_event)))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam1_writer_q, os.path.join(RESULTS_DIR, f"{self.session_timestamp}_cam1_annotated.mp4"), self.running_event)))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, self.cam1_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{self.session_timestamp}_cam1_raw.mp4"), self.running_event, True)))

        if use_cam2:
            self.cam2_raw_q = queue.Queue(maxsize=60)
            self.cam2_ui_q = queue.Queue(maxsize=30)
            
            if mode2:
                self.cam2_inf_q = mp.Queue(maxsize=30)
                self.cam2_ann_q = mp.Queue(maxsize=150)
                self.cam2_cmd_q = mp.Queue()
                self.cam2_ann_frame_q = queue.Queue(maxsize=30)
                self.cam2_writer_q = queue.Queue(maxsize=150)
            else:
                self.cam2_inf_q = self.cam2_ann_q = self.cam2_cmd_q = self.cam2_ann_frame_q = self.cam2_writer_q = None
                
            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_2, self.running_event, self.cam2_raw_q, self.cam2_inf_q, self.cam2_ann_frame_q, mode2)))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_2, selected_cam2_presets, p_info['cam2'])))
            
            if mode2:
                self.ml_p2 = mp.Process(target=inference_worker, args=(self.cam2_inf_q, self.cam2_ann_q, self.cam2_cmd_q, self.result_q, self.session_timestamp, "Cam2"))
                self.ml_p2.daemon = True; self.ml_p2.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{self.session_timestamp}_cam2_raw.mp4"), self.running_event, False)))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam2_ann_q, self.cam2_ann_frame_q, self.cam2_ui_q, self.cam2_writer_q, self.running_event)))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam2_writer_q, os.path.join(RESULTS_DIR, f"{self.session_timestamp}_cam2_annotated.mp4"), self.running_event)))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, self.cam2_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{self.session_timestamp}_cam2_raw.mp4"), self.running_event, True)))

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
        if hasattr(self, 'cam2_cmd_q') and self.cam2_cmd_q:
            self.cam2_cmd_q.put('STOP')

        if hasattr(self, 'threads'):
            for t in self.threads:
                if t.is_alive():
                    t.join(timeout=1.0)

        if hasattr(self, 'ml_p1') and self.ml_p1.is_alive():
            self.ml_p1.join(timeout=3.0)
        if hasattr(self, 'ml_p2') and self.ml_p2.is_alive():
            self.ml_p2.join(timeout=3.0)

        # Collect final verified attendance from both camera worker processes
        combined_presence = {s: False for s in self.all_students}
        combined_counts = {s: 0 for s in self.all_students}
        received_final = False

        if hasattr(self, 'result_q') and self.result_q:
            while not self.result_q.empty():
                try:
                    res = self.result_q.get_nowait()
                    received_final = True
                    for s, is_pres in res.get('presence', {}).items():
                        if is_pres:
                            combined_presence[s] = True
                    for s, cnt in res.get('counts', {}).items():
                        combined_counts[s] = combined_counts.get(s, 0) + cnt
                except queue.Empty:
                    break

        if received_final:
            self.final_presence_map = combined_presence
            self.final_counts_map = combined_counts
            self.is_finalized = True
            self.roster_mode_badge.configure(text="FINAL VERIFIED", fg_color="#238636", text_color="#ffffff")
            
            # Save merged dual-camera attendance sheet
            merged_rows = [
                {
                    'Name': s,
                    'Status': 'Present' if combined_presence.get(s, False) else 'Absent',
                    'Total Detections': combined_counts.get(s, 0)
                }
                for s in self.all_students
            ]
            merged_csv_path = os.path.join(RESULTS_DIR, f"{self.session_timestamp}_COMBINED_output.csv")
            pd.DataFrame(merged_rows).to_csv(merged_csv_path, index=False)
            pres_total = sum(1 for v in combined_presence.values() if v)
            self.log(f"[ROSTER] Final attendance verified: {pres_total}/{len(self.all_students)} Present. Saved to {merged_csv_path}")
        else:
            self.roster_mode_badge.configure(text="SESSION ENDED", fg_color="#da3633", text_color="#ffffff")

        self.refresh_roster_display(force=True)

        # Clean up shared memory blocks
        for q_name in ['cam1_inf_q', 'cam2_inf_q', 'cam1_raw_q', 'cam2_raw_q', 'cam1_ann_frame_q', 'cam2_ann_frame_q']:
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

        for q_attr in ['cam1_ann_q', 'cam2_ann_q', 'cam1_writer_q', 'cam2_writer_q', 'cam1_raw_q', 'cam2_raw_q']:
            if hasattr(self, q_attr) and getattr(self, q_attr):
                getattr(self, q_attr).put(None)
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.update_status_indicators()

    def ptz_runner(self, ip, presets, duration):
        for preset in presets:
            if not self.running_event.is_set(): break
            url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&poscall&{preset}"
            try: requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), timeout=5)
            except Exception: pass
            
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
        self.ptz_cmd_queue.put(url)

    def update_gui_frame(self):
        if not self.running_event.is_set():
            self.video_frame.configure(image=None, text="Camera Feed Offline — Session Completed")
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
                    frame1, self.cam1_active_faces, names1 = self.cam1_ui_q.get_nowait()
                    for n in names1:
                        if n not in self.live_present_set:
                            self.live_present_set.add(n)
                            newly_detected = True
                except queue.Empty: pass
                    
            frame2 = None
            if hasattr(self, 'cam2_ui_q') and self.cam2_ui_q and not self.cam2_ui_q.empty():
                try: 
                    frame2, self.cam2_active_faces, names2 = self.cam2_ui_q.get_nowait()
                    for n in names2:
                        if n not in self.live_present_set:
                            self.live_present_set.add(n)
                            newly_detected = True
                except queue.Empty: pass

            if newly_detected:
                self.refresh_roster_display()
            
            if (now - self.last_fps_time) >= 1.0:
                self.sys_fps = self.frames_rendered / (now - self.last_fps_time)
                self.fps_lbl.configure(text=f"{self.sys_fps:.1f} FPS")
                disp_faces = self.cam1_active_faces if self.view_target.get() == "Camera 1" else self.cam2_active_faces
                self.faces_lbl.configure(text=f"{disp_faces} Detected")
                self.frames_rendered = 0
                self.last_fps_time = now

            target_frame = frame1 if self.view_target.get() == "Camera 1" else frame2
            if target_frame is not None:
                self.frames_rendered += 1
                
                curr_w = self.video_container.winfo_width()
                curr_h = self.video_container.winfo_height()
                avail_w = max(160, curr_w - 8)
                avail_h = max(90, curr_h - 8)
                
                if (avail_w / avail_h) > (16.0 / 9.0):
                    render_h = avail_h
                    render_w = int(render_h * (16.0 / 9.0))
                else:
                    render_w = avail_w
                    render_h = int(render_w * (9.0 / 16.0))
                    
                render_w = max(16, render_w)
                render_h = max(9, render_h)
                
                resized_frame = cv2.resize(target_frame, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(render_w, render_h))
                self.video_frame.configure(image=imgtk, text="")
                self.video_frame.image = imgtk
                
        except Exception:
            pass
        finally:
            self.after(10, self.update_gui_frame)

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    try:
        app = AttendanceApp()
        app.mainloop()
    except Exception as e:
        if "no display name" in str(e) or "$DISPLAY" in str(e) or "couldn't connect to display" in str(e):
            print(f"\n[ERROR] Display connection failed: {e}")
            print("[INFO] No active X display found. If connecting remotely via SSH, enable X11 forwarding with 'ssh -X' or 'ssh -Y', or set 'export DISPLAY=:1'.")
        raise
