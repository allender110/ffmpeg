import subprocess
import os
def extract_keyframes(video_path, output_dir, frame_name="keyframe_%03d.jpg"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', "select=eq(pict_type\\,PICT_TYPE_I)",  # Select only I-frames
        '-vsync', 'vfr',
        os.path.join(output_dir, frame_name)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Keyframes extracted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while extracting keyframes: {e.stderr.decode()}")
    except FileNotFoundError:
        print("ffmpeg is not installed or not found in PATH. Please install ffmpeg first.")


# 使用示例
video_path = "tests1.mp4"
output_dir = "picture"

extract_keyframes(video_path, output_dir)