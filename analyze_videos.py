import os
import cv2
import pandas as pd

# The default directory where raw videos are saved in main_gui.py
VIDEO_DIR = "VIDEOS"
OUTPUT_REPORT = "video_protocols_report_fixed.csv"

def format_time(seconds):
    """Converts seconds into MM:SS format."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def categorize_protocol(duration_sec):
    """Matches the video duration to the correct protocol, short test, or random length."""
    # Catch manually stopped test videos (under 90 seconds)
    if duration_sec <= 90:  
        return "Test (Short)"

    # Base protocol times in seconds
    protocols = {
        "5-Minute Protocol": 5 * 60,
        "10-Minute Protocol": 10 * 60,
        "15-Minute Protocol": 15 * 60,
        "20-Minute Protocol": 20 * 60,
        "25-Minute Protocol": 25 * 60,
        "30-Minute Protocol": 30 * 60,
        "35-Minute Protocol": 35 * 60,
        "40-Minute Protocol": 40 * 60
    }

    # Increased tolerance to 45 seconds to account for observed system lag
    tolerance = 45 

    # Check if it matches a standard protocol
    for name, expected_sec in protocols.items():
        if abs(duration_sec - expected_sec) <= tolerance:
            return name

    # If it's longer than 90s but doesn't fit a standard protocol window
    return "Random Length / Interrupted"

def analyze_videos():
    if not os.path.exists(VIDEO_DIR):
        print(f"Directory '{VIDEO_DIR}' not found. Please ensure it exists.")
        return

    video_data = []

    for filename in os.listdir(VIDEO_DIR):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            filepath = os.path.join(VIDEO_DIR, filename)
            
            # Read video metadata using OpenCV
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                print(f"Could not open {filename}")
                continue
                
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Prevent division by zero if video is corrupted
            if fps <= 0:
                fps = 30.0 

            duration_sec = frame_count / fps
            protocol_name = categorize_protocol(duration_sec)
            
            video_data.append({
                "File Name": filename,
                "Duration (Seconds)": round(duration_sec, 2),
                "Duration (MM:SS)": format_time(duration_sec),
                "Identified Protocol": protocol_name
            })

    # Export to CSV
    if video_data:
        df = pd.DataFrame(video_data)
        # Sort by the identified protocol name for easier reading
        df = df.sort_values(by=["Identified Protocol", "Duration (Seconds)"])
        df.to_csv(OUTPUT_REPORT, index=False)
        print(f"Successfully analyzed {len(video_data)} videos.")
        print(f"Report saved to: {os.path.abspath(OUTPUT_REPORT)}")
    else:
        print(f"No video files found in '{VIDEO_DIR}'.")

if __name__ == "__main__":
    analyze_videos()