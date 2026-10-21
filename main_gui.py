import os
import sys
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
# CONSOLE STREAM REDIRECTOR (THREAD-SAFE)
# ==========================================
class ConsoleRedirector:
    """Redirects stdout/stderr writes to a queue-backed GUI logger while preserving original terminal streams."""
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
# ==========================================
# TYPOGRAPHY & DESIGN CONSTANTS
# ==========================================
FONT_FAMILY = "DejaVu Sans"
FONT_MONO = "DejaVu Sans Mono"


# ==========================================
# MAIN DESKTOP GUI
# ==========================================
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Real-Time Dual-Camera Attendance AI System | Multiprocess Vision Engine")
        self.geometry("1450x900")
        self.minsize(1350, 800)
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

        # Console logging queue & redirection setup
        self.log_queue = queue.Queue()
        self.autoscroll_var = ctk.BooleanVar(value=True)
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self._enqueue_log, self._orig_stdout)
        sys.stderr = ConsoleRedirector(self._enqueue_log, self._orig_stderr)

        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.ptz_cmd_queue = queue.Queue()
        self.ptz_worker_thread = threading.Thread(target=self.ptz_command_worker, daemon=True)
        self.ptz_worker_thread.start()

        # Start log queue consumer
        self.after(50, self.process_log_queue)
        self.log("[SYSTEM] Attendance AI Core initialized successfully.")
        self.log(f"[SYSTEM] Hardware targets: Camera 1 ({CAMERA_IP_1}), Camera 2 ({CAMERA_IP_2}).")

    def _enqueue_log(self, text):
        if text:
            self.log_queue.put(text)

    def log(self, message):
        """Helper method to route messages with timestamp to the integrated console log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message.strip()}\n"
        self._enqueue_log(formatted)

    def clear_logs(self):
        """Clears the console log window."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def process_log_queue(self):
        """Polls log_queue and flushes messages to the read-only CTkTextbox."""
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
        print("Application shutting down... cleaning up memory and processes.")
        self.stop_tracking()
        if hasattr(self, 'ptz_cmd_queue'):
            self.ptz_cmd_queue.put("QUIT")
        if hasattr(self, '_orig_stdout') and self._orig_stdout:
            sys.stdout = self._orig_stdout
        if hasattr(self, '_orig_stderr') and self._orig_stderr:
            sys.stderr = self._orig_stderr
        self.destroy()

    def create_card(self, parent, title="", subtitle=None):
        """Helper to create visually grouped card containers with consistent styling."""
        card = ctk.CTkFrame(
            parent,
            corner_radius=10,
            fg_color=("#f0f2f5", "#1c2027"),
            border_width=1,
            border_color=("#d0d7de", "#2d3340")
        )
        card.pack(fill="x", padx=6, pady=4)
        
        if title:
            header_frame = ctk.CTkFrame(card, fg_color="transparent")
            header_frame.pack(fill="x", padx=10, pady=(6, 2))
            
            lbl = ctk.CTkLabel(
                header_frame,
                text=title,
                font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                text_color=("#1f2328", "#58a6ff")
            )
            lbl.pack(side="left", anchor="w")
            
            if subtitle:
                sub = ctk.CTkLabel(
                    card,
                    text=subtitle,
                    font=ctk.CTkFont(family=FONT_FAMILY, size=10),
                    text_color=("#656d76", "#8b949e")
                )
                sub.pack(anchor="w", padx=10, pady=(0, 4))
                
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="x", padx=10, pady=(2, 6))
        return content_frame

    def toggle_all_presets(self, cam_num, select=True):
        """Quickly select or deselect all presets for a given camera."""
        vars_dict = self.cam1_preset_vars if cam_num == 1 else self.cam2_preset_vars
        for var in vars_dict.values():
            var.set(select)

    def on_view_target_change(self, value=None):
        """Updates active target indicators when segmented button changes."""
        target = self.view_target.get()
        if hasattr(self, 'active_stream_lbl'):
            self.active_stream_lbl.configure(text=target)
        if hasattr(self, 'ptz_target_lbl'):
            self.ptz_target_lbl.configure(text=f"Directing: {target}")
        disp_faces = self.cam1_active_faces if target == "Camera 1" else self.cam2_active_faces
        if hasattr(self, 'faces_lbl'):
            self.faces_lbl.configure(text=f"{disp_faces} Detected")
        self.log(f"[UI] Active telemetry viewport changed to {target}.")

    def update_status_indicators(self):
        """Updates colored status indicator dots and badges for Cam1, Cam2, and ML Engine."""
        is_running = self.running_event.is_set()
        
        # Camera 1 Status
        cam1_on = self.cam1_enable_var.get()
        if is_running and cam1_on:
            self.cam1_status_dot.configure(text="●", text_color="#3fb950")
            self.cam1_status_state.configure(text="Streaming", text_color="#3fb950")
        elif not cam1_on:
            self.cam1_status_dot.configure(text="●", text_color="#8b949e")
            self.cam1_status_state.configure(text="Disabled", text_color="#8b949e")
        else:
            self.cam1_status_dot.configure(text="●", text_color="#f85149")
            self.cam1_status_state.configure(text="Offline", text_color="#f85149")

        # Camera 2 Status
        cam2_on = self.cam2_enable_var.get()
        if is_running and cam2_on:
            self.cam2_status_dot.configure(text="●", text_color="#3fb950")
            self.cam2_status_state.configure(text="Streaming", text_color="#3fb950")
        elif not cam2_on:
            self.cam2_status_dot.configure(text="●", text_color="#8b949e")
            self.cam2_status_state.configure(text="Disabled", text_color="#8b949e")
        else:
            self.cam2_status_dot.configure(text="●", text_color="#f85149")
            self.cam2_status_state.configure(text="Offline", text_color="#f85149")

        # ML Engine Status
        tracking_on = self.global_live_tracking_var.get()
        if is_running:
            if tracking_on:
                self.ml_status_dot.configure(text="●", text_color="#3fb950")
                self.ml_status_state.configure(text="Inferencing", text_color="#3fb950")
            else:
                self.ml_status_dot.configure(text="●", text_color="#d29922")
                self.ml_status_state.configure(text="Raw Ingest", text_color="#d29922")
        else:
            self.ml_status_dot.configure(text="●", text_color="#8b949e")
            self.ml_status_state.configure(text="Standby", text_color="#8b949e")

    def setup_ui(self):
        # Master Window Layout: Left Sidebar + Right Main Content
        self.sidebar_frame = ctk.CTkFrame(self, width=390, corner_radius=12, fg_color=("#ebebeb", "#16191f"), border_width=1, border_color=("#d0d7de", "#242933"))
        self.sidebar_frame.pack(side="left", fill="y", padx=(12, 6), pady=12)
        self.sidebar_frame.pack_propagate(False)

        self.main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content_frame.pack(side="right", fill="both", expand=True, padx=(6, 12), pady=12)

        # -------------------------------------------------------------
        # 1. SIDEBAR: Header & Branding
        # -------------------------------------------------------------
        self.brand_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.brand_frame.pack(fill="x", padx=12, pady=(14, 8))

        self.logo_label = ctk.CTkLabel(
            self.brand_frame,
            text="VISION SENTINEL CORE",
            font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
            text_color=("#1f2328", "#ffffff")
        )
        self.logo_label.pack(anchor="w")

        self.subtitle_label = ctk.CTkLabel(
            self.brand_frame,
            text="Dual-Camera Multiprocess AI Pipeline",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=("#656d76", "#8b949e")
        )
        self.subtitle_label.pack(anchor="w")

        # -------------------------------------------------------------
        # 2. SIDEBAR: Live Status Indicators Card (Requirement 5)
        # -------------------------------------------------------------
        self.status_card = ctk.CTkFrame(
            self.sidebar_frame,
            corner_radius=8,
            fg_color=("#dedede", "#1c2027"),
            border_width=1,
            border_color=("#d0d7de", "#2d3340")
        )
        self.status_card.pack(fill="x", padx=10, pady=(0, 6))

        status_header = ctk.CTkFrame(self.status_card, fg_color="transparent")
        status_header.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(status_header, text="LIVE SYSTEM STATUS", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=("#57606a", "#8b949e")).pack(side="left")

        status_grid = ctk.CTkFrame(self.status_card, fg_color="transparent")
        status_grid.pack(fill="x", padx=6, pady=(0, 6))
        status_grid.grid_columnconfigure((0, 1, 2), weight=1)

        # Cam 1 Status Indicator
        cam1_status_box = ctk.CTkFrame(status_grid, fg_color=("#ffffff", "#21262d"), corner_radius=6)
        cam1_status_box.grid(row=0, column=0, padx=2, pady=2, sticky="nsew")
        cam1_dot_box = ctk.CTkFrame(cam1_status_box, fg_color="transparent")
        cam1_dot_box.pack(pady=(4, 0))
        self.cam1_status_dot = ctk.CTkLabel(cam1_dot_box, text="●", font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color="#f85149")
        self.cam1_status_dot.pack(side="left", padx=3)
        ctk.CTkLabel(cam1_dot_box, text="Cam 1", font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold")).pack(side="left")
        self.cam1_status_state = ctk.CTkLabel(cam1_status_box, text="Offline", font=ctk.CTkFont(family=FONT_FAMILY, size=10), text_color="#f85149")
        self.cam1_status_state.pack(pady=(0, 4))

        # Cam 2 Status Indicator
        cam2_status_box = ctk.CTkFrame(status_grid, fg_color=("#ffffff", "#21262d"), corner_radius=6)
        cam2_status_box.grid(row=0, column=1, padx=2, pady=2, sticky="nsew")
        cam2_dot_box = ctk.CTkFrame(cam2_status_box, fg_color="transparent")
        cam2_dot_box.pack(pady=(4, 0))
        self.cam2_status_dot = ctk.CTkLabel(cam2_dot_box, text="●", font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color="#f85149")
        self.cam2_status_dot.pack(side="left", padx=3)
        ctk.CTkLabel(cam2_dot_box, text="Cam 2", font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold")).pack(side="left")
        self.cam2_status_state = ctk.CTkLabel(cam2_status_box, text="Offline", font=ctk.CTkFont(family=FONT_FAMILY, size=10), text_color="#f85149")
        self.cam2_status_state.pack(pady=(0, 4))

        # ML Engine Status Indicator
        ml_status_box = ctk.CTkFrame(status_grid, fg_color=("#ffffff", "#21262d"), corner_radius=6)
        ml_status_box.grid(row=0, column=2, padx=2, pady=2, sticky="nsew")
        ml_dot_box = ctk.CTkFrame(ml_status_box, fg_color="transparent")
        ml_dot_box.pack(pady=(4, 0))
        self.ml_status_dot = ctk.CTkLabel(ml_dot_box, text="●", font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color="#8b949e")
        self.ml_status_dot.pack(side="left", padx=3)
        ctk.CTkLabel(ml_dot_box, text="ML Core", font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold")).pack(side="left")
        self.ml_status_state = ctk.CTkLabel(ml_status_box, text="Standby", font=ctk.CTkFont(family=FONT_FAMILY, size=10), text_color="#8b949e")
        self.ml_status_state.pack(pady=(0, 4))

        # -------------------------------------------------------------
        # 3. SIDEBAR: Tabview (Requirement 1)
        # -------------------------------------------------------------
        self.tabview = ctk.CTkTabview(
            self.sidebar_frame,
            corner_radius=10,
            fg_color=("#dedede", "#1b1e25")
        )
        self.tabview.pack(fill="both", expand=True, padx=8, pady=(2, 6))

        tab_controls = self.tabview.add("Controls")
        tab_ptz = self.tabview.add("PTZ & Presets")
        tab_system = self.tabview.add("System")

        # Scrollable wrappers inside tabs to ensure responsiveness on any screen height
        self.controls_scroll = ctk.CTkScrollableFrame(tab_controls, fg_color="transparent")
        self.controls_scroll.pack(fill="both", expand=True)

        self.ptz_scroll = ctk.CTkScrollableFrame(tab_ptz, fg_color="transparent")
        self.ptz_scroll.pack(fill="both", expand=True)

        self.system_scroll = ctk.CTkScrollableFrame(tab_system, fg_color="transparent")
        self.system_scroll.pack(fill="both", expand=True)

        # --- Tab 1: Controls ---
        # Card: Active Telemetry Viewport
        card_viewport = self.create_card(self.controls_scroll, "ACTIVE TELEMETRY VIEW", "Select active camera feed for viewport & telemetry display")
        self.view_toggle = ctk.CTkSegmentedButton(
            card_viewport,
            values=["Camera 1", "Camera 2"],
            variable=self.view_target,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            command=self.on_view_target_change
        )
        self.view_toggle.pack(fill="x", pady=4)

        # Card: Camera Feed Setup (Requirement 2: Card-based grouping)
        card_cam_setup = self.create_card(self.controls_scroll, "CAMERA FEED SETUP", "Toggle RTSP ingestion streams for cameras")
        self.cam1_enable_var = ctk.BooleanVar(value=True)
        self.cam1_enable_switch = ctk.CTkSwitch(
            card_cam_setup,
            text=f"Enable Camera 1 ({CAMERA_IP_1})",
            variable=self.cam1_enable_var,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            command=self.update_status_indicators
        )
        self.cam1_enable_switch.pack(pady=(4, 6), anchor="w")

        self.cam2_enable_var = ctk.BooleanVar(value=True)
        self.cam2_enable_switch = ctk.CTkSwitch(
            card_cam_setup,
            text=f"Enable Camera 2 ({CAMERA_IP_2})",
            variable=self.cam2_enable_var,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            command=self.update_status_indicators
        )
        self.cam2_enable_switch.pack(pady=(0, 4), anchor="w")

        # Card: Recognition & Tracking
        card_ml = self.create_card(self.controls_scroll, "REAL-TIME ML ENGINE", "Facial detection & identity recognition pipeline")
        self.global_live_tracking_var = ctk.BooleanVar(value=True)
        self.global_live_tracking_switch = ctk.CTkSwitch(
            card_ml,
            text="Live Tracking & Face Matching",
            variable=self.global_live_tracking_var,
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            command=self.update_status_indicators
        )
        self.global_live_tracking_switch.pack(pady=4, anchor="w")

        ml_note = ctk.CTkLabel(
            card_ml,
            text="When enabled, runs YOLOv8n-Face + InceptionResNetV1 in parallel spawn workers. When disabled, operates in raw recording mode.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=10),
            text_color=("#656d76", "#8b949e"),
            wraplength=320,
            justify="left"
        )
        ml_note.pack(anchor="w", pady=(2, 0))

        # --- Tab 2: PTZ & Presets ---
        # Card: Execution Protocol (Compact 2-column format)
        card_proto = self.create_card(self.ptz_scroll, "EXECUTION PROTOCOL", "Preset dwell durations (Cam 1 / Cam 2)")
        proto_grid = ctk.CTkFrame(card_proto, fg_color="transparent")
        proto_grid.pack(fill="x")
        proto_grid.grid_columnconfigure((0, 1), weight=1)

        for idx, (p, info) in enumerate(PROTOCOLS.items()):
            r = idx // 2
            c = idx % 2
            rb_text = f"{info['name']} ({info['cam1']}s/{info['cam2']}s)"
            rb = ctk.CTkRadioButton(
                proto_grid,
                text=rb_text,
                variable=self.selected_protocol,
                value=p,
                font=ctk.CTkFont(family=FONT_FAMILY, size=11)
            )
            rb.grid(row=r, column=c, pady=3, padx=2, sticky="w")

        # Card: Presets Selection
        card_presets = self.create_card(self.ptz_scroll, "CAMERA PRESETS", "Active waypoint presets in tour cycle")
        
        # Cam 1 Presets
        cam1_header = ctk.CTkFrame(card_presets, fg_color="transparent")
        cam1_header.pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(cam1_header, text="Camera 1 Presets (1-8):", font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold")).pack(side="left")
        ctk.CTkButton(cam1_header, text="All", width=36, height=20, font=ctk.CTkFont(family=FONT_FAMILY, size=10), command=lambda: self.toggle_all_presets(1, True)).pack(side="right", padx=(2, 0))
        ctk.CTkButton(cam1_header, text="None", width=40, height=20, font=ctk.CTkFont(family=FONT_FAMILY, size=10), command=lambda: self.toggle_all_presets(1, False)).pack(side="right")

        self.preset_frame1 = ctk.CTkFrame(card_presets, fg_color="transparent")
        self.preset_frame1.pack(fill="x", pady=(2, 4))
        self.preset_frame1.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self.cam1_preset_vars = {}
        for idx, p in enumerate(CAM1_PRESETS):
            var = ctk.BooleanVar(value=True)
            self.cam1_preset_vars[p] = var
            cb = ctk.CTkCheckBox(self.preset_frame1, text=f"P{p}", variable=var, width=44, font=ctk.CTkFont(family=FONT_FAMILY, size=11))
            cb.grid(row=idx // 4, column=idx % 4, padx=3, pady=2, sticky="w")

        # Cam 2 Presets
        cam2_header = ctk.CTkFrame(card_presets, fg_color="transparent")
        cam2_header.pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(cam2_header, text="Camera 2 Presets (1-6):", font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold")).pack(side="left")
        ctk.CTkButton(cam2_header, text="All", width=36, height=20, font=ctk.CTkFont(family=FONT_FAMILY, size=10), command=lambda: self.toggle_all_presets(2, True)).pack(side="right", padx=(2, 0))
        ctk.CTkButton(cam2_header, text="None", width=40, height=20, font=ctk.CTkFont(family=FONT_FAMILY, size=10), command=lambda: self.toggle_all_presets(2, False)).pack(side="right")

        self.preset_frame2 = ctk.CTkFrame(card_presets, fg_color="transparent")
        self.preset_frame2.pack(fill="x", pady=(2, 4))
        self.preset_frame2.grid_columnconfigure((0, 1, 2), weight=1)
        self.cam2_preset_vars = {}
        for idx, p in enumerate(CAM2_PRESETS):
            var = ctk.BooleanVar(value=True)
            self.cam2_preset_vars[p] = var
            cb = ctk.CTkCheckBox(self.preset_frame2, text=f"P{p}", variable=var, width=44, font=ctk.CTkFont(family=FONT_FAMILY, size=11))
            cb.grid(row=idx // 3, column=idx % 3, padx=3, pady=2, sticky="w")

        # Card: Modernized Manual PTZ Controls (Requirement 3: Standard Unicode arrows & compact D-pad)
        card_ptz = self.create_card(self.ptz_scroll, "MANUAL PTZ CONTROLS", "Hold button to move/zoom, release to stop")
        
        self.ptz_target_lbl = ctk.CTkLabel(
            card_ptz,
            text=f"Directing: {self.view_target.get()}",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"),
            text_color=("#1f6aa5", "#58a6ff")
        )
        self.ptz_target_lbl.pack(anchor="w", pady=(0, 4))

        # D-pad & Lens Controls Container
        ptz_control_box = ctk.CTkFrame(card_ptz, fg_color="transparent")
        ptz_control_box.pack(fill="x")
        ptz_control_box.grid_columnconfigure(0, weight=3) # D-pad
        ptz_control_box.grid_columnconfigure(1, weight=3) # Lens

        # D-Pad Grid
        dpad_container = ctk.CTkFrame(ptz_control_box, fg_color=("#ffffff", "#21262d"), corner_radius=8, border_width=1, border_color=("#d0d7de", "#2d3340"))
        dpad_container.grid(row=0, column=0, padx=(0, 4), pady=2, sticky="nsew")

        btn_up = ctk.CTkButton(dpad_container, text="▲", width=42, height=34, font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"), corner_radius=6)
        btn_left = ctk.CTkButton(dpad_container, text="◀", width=42, height=34, font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"), corner_radius=6)
        btn_center = ctk.CTkButton(dpad_container, text="PTZ", width=42, height=34, font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), fg_color=("#e1e4e8", "#161b22"), text_color=("#57606a", "#8b949e"), state="disabled", corner_radius=6)
        btn_right = ctk.CTkButton(dpad_container, text="▶", width=42, height=34, font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"), corner_radius=6)
        btn_down = ctk.CTkButton(dpad_container, text="▼", width=42, height=34, font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"), corner_radius=6)

        btn_up.grid(row=0, column=1, padx=3, pady=3)
        btn_left.grid(row=1, column=0, padx=3, pady=2)
        btn_center.grid(row=1, column=1, padx=3, pady=2)
        btn_right.grid(row=1, column=2, padx=3, pady=2)
        btn_down.grid(row=2, column=1, padx=3, pady=3)

        # Lens Controls Container (Zoom & Focus)
        lens_container = ctk.CTkFrame(ptz_control_box, fg_color=("#ffffff", "#21262d"), corner_radius=8, border_width=1, border_color=("#d0d7de", "#2d3340"))
        lens_container.grid(row=0, column=1, padx=(4, 0), pady=2, sticky="nsew")
        lens_container.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(lens_container, text="ZOOM", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=("#57606a", "#8b949e")).grid(row=0, column=0, columnspan=2, pady=(3, 1))
        btn_zi = ctk.CTkButton(lens_container, text="+ In", height=28, font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"), corner_radius=6)
        btn_zo = ctk.CTkButton(lens_container, text="- Out", height=28, font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"), corner_radius=6)
        btn_zi.grid(row=1, column=0, padx=2, pady=1, sticky="ew")
        btn_zo.grid(row=1, column=1, padx=2, pady=1, sticky="ew")

        ctk.CTkLabel(lens_container, text="FOCUS", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=("#57606a", "#8b949e")).grid(row=2, column=0, columnspan=2, pady=(3, 1))
        btn_fi = ctk.CTkButton(lens_container, text="+ Near", height=28, font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"), corner_radius=6)
        btn_fo = ctk.CTkButton(lens_container, text="- Far", height=28, font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"), corner_radius=6)
        btn_fi.grid(row=3, column=0, padx=2, pady=(1, 3), sticky="ew")
        btn_fo.grid(row=3, column=1, padx=2, pady=(1, 3), sticky="ew")

        # Bind button press & release events for smooth continuous PTZ operation
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

        # --- Tab 3: System ---
        card_sys_net = self.create_card(self.system_scroll, "STREAM & NETWORK SETUP", "Configured IP streams & endpoints")
        ctk.CTkLabel(card_sys_net, text=f"• Cam 1 IP: {CAMERA_IP_1} (RTSP 554)", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)
        ctk.CTkLabel(card_sys_net, text=f"• Cam 2 IP: {CAMERA_IP_2} (RTSP 554)", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)
        ctk.CTkLabel(card_sys_net, text="• PTZ Protocol: CGI Command API", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)
        ctk.CTkLabel(card_sys_net, text="• Ingestion Auth: HTTP Basic Auth", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)

        card_sys_storage = self.create_card(self.system_scroll, "STORAGE & MODEL ASSETS", "Output directories and checkpoint locations")
        ctk.CTkLabel(card_sys_storage, text=f"• Videos Dir: {BASE_OUTPUT_DIR}/", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)
        ctk.CTkLabel(card_sys_storage, text=f"• Results Dir: {RESULTS_DIR}/", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)
        ctk.CTkLabel(card_sys_storage, text="• FAISS Index: ./face_attendance_faiss.bin", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)
        ctk.CTkLabel(card_sys_storage, text="• Face Detection: yolov8n-face.pt", font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", pady=1)

        card_sys_ui = self.create_card(self.system_scroll, "APPEARANCE & CONTROLS", "User interface preferences")
        ctk.CTkLabel(card_sys_ui, text="Theme Mode:", font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold")).pack(anchor="w", pady=(2, 2))
        theme_selector = ctk.CTkSegmentedButton(
            card_sys_ui,
            values=["Dark", "Light", "System"],
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            command=lambda mode: ctk.set_appearance_mode(mode.lower())
        )
        theme_selector.set("Dark")
        theme_selector.pack(fill="x", pady=(0, 6))

        clear_btn_sys = ctk.CTkButton(
            card_sys_ui,
            text="Clear Console History",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            fg_color=("#30363d", "#21262d"),
            hover_color=("#484f58", "#30363d"),
            command=self.clear_logs
        )
        clear_btn_sys.pack(fill="x", pady=2)

        # -------------------------------------------------------------
        # 4. SIDEBAR: Prominent Call-to-Action Buttons (Requirement 7)
        # -------------------------------------------------------------
        self.cta_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.cta_frame.pack(side="bottom", fill="x", padx=10, pady=(0, 10))

        self.start_btn = ctk.CTkButton(
            self.cta_frame,
            text="▶  EXECUTE BOUND LOOP",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            height=46,
            fg_color="#2ea043",
            hover_color="#238636",
            text_color="#ffffff",
            corner_radius=8,
            command=self.start_tracking
        )
        self.start_btn.pack(fill="x", pady=(0, 6))

        self.stop_btn = ctk.CTkButton(
            self.cta_frame,
            text="■  FORCE STOP SYSTEM",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            height=40,
            fg_color="#da3633",
            hover_color="#b62324",
            text_color="#ffffff",
            corner_radius=8,
            command=self.stop_tracking
        )
        self.stop_btn.pack(fill="x", pady=0)
        self.stop_btn.configure(state="disabled")

        # -------------------------------------------------------------
        # 5. RIGHT MAIN: Horizontal Telemetry HUD (Requirement 4)
        # -------------------------------------------------------------
        self.hud_frame = ctk.CTkFrame(
            self.main_content_frame,
            corner_radius=12,
            fg_color=("#f0f2f5", "#16191f"),
            border_width=1,
            border_color=("#d0d7de", "#242933")
        )
        self.hud_frame.pack(fill="x", padx=0, pady=(0, 10))
        self.hud_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # HUD Tile 0: Remaining Timer
        tile0 = ctk.CTkFrame(self.hud_frame, fg_color=("#ffffff", "#1f242d"), corner_radius=8, border_width=1, border_color=("#e1e4e8", "#2d3340"))
        tile0.grid(row=0, column=0, padx=6, pady=8, sticky="nsew")
        ctk.CTkLabel(tile0, text="REMAINING TIMER", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=("#57606a", "#8b949e")).pack(pady=(6, 0), padx=10, anchor="w")
        self.time_lbl = ctk.CTkLabel(tile0, text="00:00:00", font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"), text_color="#58a6ff")
        self.time_lbl.pack(pady=(0, 6), padx=10, anchor="w")

        # HUD Tile 1: Render Speed
        tile1 = ctk.CTkFrame(self.hud_frame, fg_color=("#ffffff", "#1f242d"), corner_radius=8, border_width=1, border_color=("#e1e4e8", "#2d3340"))
        tile1.grid(row=0, column=1, padx=6, pady=8, sticky="nsew")
        ctk.CTkLabel(tile1, text="SYSTEM RENDER FPS", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=("#57606a", "#8b949e")).pack(pady=(6, 0), padx=10, anchor="w")
        self.fps_lbl = ctk.CTkLabel(tile1, text="0.0 FPS", font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"), text_color="#3fb950")
        self.fps_lbl.pack(pady=(0, 6), padx=10, anchor="w")

        # HUD Tile 2: Target Faces
        tile2 = ctk.CTkFrame(self.hud_frame, fg_color=("#ffffff", "#1f242d"), corner_radius=8, border_width=1, border_color=("#e1e4e8", "#2d3340"))
        tile2.grid(row=0, column=2, padx=6, pady=8, sticky="nsew")
        ctk.CTkLabel(tile2, text="TARGET FACES DETECTED", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=("#57606a", "#8b949e")).pack(pady=(6, 0), padx=10, anchor="w")
        self.faces_lbl = ctk.CTkLabel(tile2, text="0 Detected", font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"), text_color="#d29922")
        self.faces_lbl.pack(pady=(0, 6), padx=10, anchor="w")

        # HUD Tile 3: Active Viewport
        tile3 = ctk.CTkFrame(self.hud_frame, fg_color=("#ffffff", "#1f242d"), corner_radius=8, border_width=1, border_color=("#e1e4e8", "#2d3340"))
        tile3.grid(row=0, column=3, padx=6, pady=8, sticky="nsew")
        ctk.CTkLabel(tile3, text="ACTIVE VIEWPORT", font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"), text_color=("#57606a", "#8b949e")).pack(pady=(6, 0), padx=10, anchor="w")
        self.active_stream_lbl = ctk.CTkLabel(tile3, text="Camera 1", font=ctk.CTkFont(family=FONT_FAMILY, size=20, weight="bold"), text_color="#bc8cff")
        self.active_stream_lbl.pack(pady=(0, 6), padx=10, anchor="w")

        # -------------------------------------------------------------
        # 6. RIGHT MAIN: Main Video Viewport
        # -------------------------------------------------------------
        self.video_container = ctk.CTkFrame(
            self.main_content_frame,
            corner_radius=12,
            fg_color=("#0a0c10", "#080a0e"),
            border_width=1,
            border_color=("#d0d7de", "#21262d")
        )
        self.video_container.pack(fill="both", expand=True, padx=0, pady=(0, 10))

        self.video_frame = ctk.CTkLabel(
            self.video_container,
            text="Camera Feed Standby\n\nClick 'Execute Bound Loop' to start ingestion and AI pipeline",
            font=ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"),
            text_color=("#57606a", "#6e7681"),
            bg_color="transparent"
        )
        self.video_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # -------------------------------------------------------------
        # 7. RIGHT MAIN: Integrated Console Logs (Requirement 6)
        # -------------------------------------------------------------
        self.log_container = ctk.CTkFrame(
            self.main_content_frame,
            corner_radius=10,
            fg_color=("#f0f2f5", "#16191f"),
            border_width=1,
            border_color=("#d0d7de", "#242933"),
            height=150
        )
        self.log_container.pack(fill="x", padx=0, pady=0)
        self.log_container.pack_propagate(False)

        log_header = ctk.CTkFrame(self.log_container, fg_color="transparent")
        log_header.pack(fill="x", padx=10, pady=(6, 4))

        log_title = ctk.CTkLabel(
            log_header,
            text="INTEGRATED SYSTEM CONSOLE & PIPELINE LOGS",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"),
            text_color=("#57606a", "#8b949e")
        )
        log_title.pack(side="left")

        clear_btn = ctk.CTkButton(
            log_header,
            text="Clear Logs",
            width=70,
            height=22,
            font=ctk.CTkFont(family=FONT_FAMILY, size=10, weight="bold"),
            fg_color=("#d0d7de", "#21262d"),
            hover_color=("#afb8c1", "#30363d"),
            text_color=("#24292f", "#c9d1d9"),
            corner_radius=4,
            command=self.clear_logs
        )
        clear_btn.pack(side="right", padx=(6, 0))

        autoscroll_cb = ctk.CTkCheckBox(
            log_header,
            text="Auto-scroll",
            variable=self.autoscroll_var,
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            checkbox_width=16,
            checkbox_height=16,
            corner_radius=4
        )
        autoscroll_cb.pack(side="right", padx=6)

        self.log_textbox = ctk.CTkTextbox(
            self.log_container,
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            wrap="word",
            fg_color=("#ffffff", "#0d1117"),
            text_color=("#24292f", "#c9d1d9"),
            border_width=1,
            border_color=("#d0d7de", "#21262d"),
            corner_radius=6
        )
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.log_textbox.configure(state="disabled")

        # Initialize status indicators
        self.update_status_indicators()

    def start_tracking(self):
        if self.running_event.is_set(): return
            
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
            self.update_status_indicators()
            return
        
        global_track = self.global_live_tracking_var.get()
        mode1 = global_track
        mode2 = global_track
        
        total_time_cam1 = len(selected_cam1_presets) * (p_info['cam1'] + 5) if use_cam1 else 0
        total_time_cam2 = len(selected_cam2_presets) * (p_info['cam2'] + 5) if use_cam2 else 0
        self.t_end = time.perf_counter() + max(total_time_cam1, total_time_cam2)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.threads = []
        
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
                self.cam1_inf_q = None
                self.cam1_ann_q = None
                self.cam1_cmd_q = None
                self.cam1_ann_frame_q = None
                self.cam1_writer_q = None
                
            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_1, self.running_event, self.cam1_raw_q, self.cam1_inf_q, self.cam1_ann_frame_q, mode1)))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_1, selected_cam1_presets, p_info['cam1'])))
            
            if mode1:
                self.ml_p1 = mp.Process(target=inference_worker, args=(self.cam1_inf_q, self.cam1_ann_q, self.cam1_cmd_q, timestamp_str, "Cam1"))
                self.ml_p1.daemon = True; self.ml_p1.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event, False)))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam1_ann_q, self.cam1_ann_frame_q, self.cam1_ui_q, self.cam1_writer_q, self.running_event)))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam1_writer_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam1_annotated.mp4"), self.running_event)))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam1_raw_q, self.cam1_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam1_raw.mp4"), self.running_event, True)))

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
                self.cam2_inf_q = None
                self.cam2_ann_q = None
                self.cam2_cmd_q = None
                self.cam2_ann_frame_q = None
                self.cam2_writer_q = None
                
            self.threads.append(threading.Thread(target=rstp_reader, args=(CAMERA_IP_2, self.running_event, self.cam2_raw_q, self.cam2_inf_q, self.cam2_ann_frame_q, mode2)))
            self.threads.append(threading.Thread(target=self.ptz_runner, args=(CAMERA_IP_2, selected_cam2_presets, p_info['cam2'])))
            
            if mode2:
                self.ml_p2 = mp.Process(target=inference_worker, args=(self.cam2_inf_q, self.cam2_ann_q, self.cam2_cmd_q, timestamp_str, "Cam2"))
                self.ml_p2.daemon = True; self.ml_p2.start()
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, None, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event, False)))
                self.threads.append(threading.Thread(target=ann_writer_worker, args=(self.cam2_ann_q, self.cam2_ann_frame_q, self.cam2_ui_q, self.cam2_writer_q, self.running_event)))
                self.threads.append(threading.Thread(target=async_video_writer_worker, args=(self.cam2_writer_q, os.path.join(RESULTS_DIR, f"{timestamp_str}_cam2_annotated.mp4"), self.running_event)))
            else:
                self.threads.append(threading.Thread(target=raw_writer_worker, args=(self.cam2_raw_q, self.cam2_ui_q, os.path.join(BASE_OUTPUT_DIR, f"{timestamp_str}_cam2_raw.mp4"), self.running_event, True)))

        for t in self.threads: t.start()

        self.last_fps_time = time.perf_counter()
        self.frames_rendered = 0
        self.update_status_indicators()
        print(f"[SYSTEM] Bound loop started. Protocol: {p_info['name']}. Session: {timestamp_str}")
        self.update_gui_frame()

    def stop_tracking(self):
        if not self.running_event.is_set(): return
        self.running_event.clear()
        print("[SYSTEM] Stopping tracking... terminating workers and cleaning shared memory.")
        
        if hasattr(self, 'cam1_cmd_q') and self.cam1_cmd_q:
            self.cam1_cmd_q.put('STOP')
        if hasattr(self, 'cam2_cmd_q') and self.cam2_cmd_q:
            self.cam2_cmd_q.put('STOP')

        # 1. CRITICAL FIX: Wait for background reader threads to stop generating frames
        if hasattr(self, 'threads'):
            for t in self.threads:
                if t.is_alive():
                    t.join(timeout=1.0)
                    
        # 2. Wait for ML processes to safely exit
        if hasattr(self, 'ml_p1') and self.ml_p1.is_alive():
            self.ml_p1.join(timeout=2.0)
        if hasattr(self, 'ml_p2') and self.ml_p2.is_alive():
            self.ml_p2.join(timeout=2.0)

        # 3. Now safely detach and purge pending SharedMemory scopes
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

        if hasattr(self, 'cam1_ann_q') and self.cam1_ann_q: self.cam1_ann_q.put(None)
        if hasattr(self, 'cam2_ann_q') and self.cam2_ann_q: self.cam2_ann_q.put(None)
        if hasattr(self, 'cam1_writer_q') and self.cam1_writer_q: self.cam1_writer_q.put(None)
        if hasattr(self, 'cam2_writer_q') and self.cam2_writer_q: self.cam2_writer_q.put(None)
        if hasattr(self, 'cam1_raw_q') and self.cam1_raw_q: self.cam1_raw_q.put(None)
        if hasattr(self, 'cam2_raw_q') and self.cam2_raw_q: self.cam2_raw_q.put(None)
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.update_status_indicators()
        print("[SYSTEM] System stopped successfully.")

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
        target = self.view_target.get()
        ip = CAMERA_IP_1 if target == "Camera 1" else CAMERA_IP_2
        url = f"http://{ip}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}"
        self.ptz_cmd_queue.put(url)

    def update_gui_frame(self):
        if not self.running_event.is_set():
            self.video_frame.configure(image=None, text="📷  Camera ML Feed Offline — System Idle")
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
            
            frame1 = None
            if hasattr(self, 'cam1_ui_q') and self.cam1_ui_q and not self.cam1_ui_q.empty():
                try: 
                    res1 = self.cam1_ui_q.get_nowait()
                    frame1, self.cam1_active_faces = res1
                except queue.Empty: pass
                    
            frame2 = None
            if hasattr(self, 'cam2_ui_q') and self.cam2_ui_q and not self.cam2_ui_q.empty():
                try: 
                    res2 = self.cam2_ui_q.get_nowait()
                    frame2, self.cam2_active_faces = res2
                except queue.Empty: pass
            
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