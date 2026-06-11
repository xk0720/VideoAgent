import os
import subprocess
import base64
import io
import numpy as np
from PIL import Image
import imageio
from utils.query_llm import query_openai
import logging
import json
import math

from typing import List, Union
from decord import VideoReader, cpu, DECORDError


os.chdir(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config/config.yaml")
import yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

llm_config = config.get('llm', {})
logger = logging.getLogger(__name__)


def merge_videos(video_paths: list[str] | str, output_file="merged.mp4", audio_list: list[str] | None = None):
    """
    Merges multiple video files into a single video file.

    Args:
        video_paths (list[str] | str): A list of paths to the video files to merge, or a folder path containing videos.
        output_file (str): The path to save the merged video.
        audio_list (list[str] | None): A list of paths to the audio files to merge. If None, no audio is merged.

    Returns:
        str: The path to the merged video file if successful, None otherwise.
    """
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        if isinstance(video_paths, str):
            # If a folder path is provided, list files within it
            folder_path = video_paths
            video_files_to_merge = sorted(
                [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mov', '.avi'))],
                key=lambda x: x.lower()
            )
        elif isinstance(video_paths, list):
            # If a list of file paths is provided, use it directly
            video_files_to_merge = sorted(
                [f for f in video_paths if f.lower().endswith(('.mp4', '.mov', '.avi'))],
                key=lambda x: x.lower()
            )
        else:
            raise TypeError("video_paths must be a string (folder path) or a list of strings (file paths).")

        if not video_files_to_merge:
            print("No video files found to merge.")
            return None

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        list_file = "concat_list.txt"
        with open(list_file, "w") as f:
            for file in video_files_to_merge:
                f.write(f"file '{file}'\n")
        
        # If audio_list is provided, merge audio with video
        if audio_list is not None:
            # Create a temporary file for merged video without audio
            temp_video_file = "temp_merged_video.mp4"
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264",  # Re-encode video for better compatibility
                "-preset", "ultrafast",  # Use fast encoding preset
                "-crf", "23",  # Constant rate factor for quality
                "-an",  # Disable audio
                "-y",
                temp_video_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Video concatenation without audio failed: {result.stderr}")
                # Fallback to copy if re-encoding fails
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    "-an",  # Disable audio
                    "-y",
                    temp_video_file
                ]
                subprocess.run(cmd)

            # If there's only one audio file, directly merge it with the video
            if len(audio_list) == 1:
                cmd = [
                    "ffmpeg",
                    "-i", temp_video_file,
                    "-i", audio_list[0],
                    "-c:v", "copy",  # Copy video stream
                    "-c:a", "aac",   # Re-encode audio to AAC for better compatibility
                    "-y",
                    output_file
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Failed to merge video and audio: {result.stderr}")
                    # Fallback to copy if re-encoding fails
                    cmd = [
                        "ffmpeg",
                        "-i", temp_video_file,
                        "-i", audio_list[0],
                        "-c", "copy",
                        "-y",
                        output_file
                    ]
                    subprocess.run(cmd)
            else:
                # If there are multiple audio files, merge them first
                temp_audio_file = "temp_merged_audio.wav"
                # Create a list file for audio files
                audio_list_file = "audio_concat_list.txt"
                with open(audio_list_file, "w") as f:
                    for file in audio_list:
                        f.write(f"file '{file}'\n")
                
                # Use concat demuxer instead of concat protocol for better compatibility
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", audio_list_file,
                    "-c:a", "aac",  # Re-encode audio to AAC for better compatibility
                    "-y",
                    temp_audio_file
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Audio concatenation failed: {result.stderr}")
                    # Fallback: try re-encoding with different parameters
                    cmd = [
                        "ffmpeg",
                        "-f", "concat",
                        "-safe", "0",
                        "-i", audio_list_file,
                        "-c:a", "pcm_s16le",  # Use WAV codec for better compatibility
                        "-y",
                        temp_audio_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"Second audio concatenation attempt failed: {result.stderr}")
                        # Final fallback: try copying if re-encoding fails
                        cmd = [
                            "ffmpeg",
                            "-f", "concat",
                            "-safe", "0",
                            "-i", audio_list_file,
                            "-c", "copy",
                            "-y",
                            temp_audio_file
                        ]
                        subprocess.run(cmd)
                
                # Merge video and merged audio
                # Re-encode audio to AAC for MP4 compatibility
                cmd = [
                    "ffmpeg",
                    "-i", temp_video_file,
                    "-i", temp_audio_file,
                    "-c:v", "copy",  # Copy video stream
                    "-c:a", "aac",   # Re-encode audio to AAC
                    "-b:a", "192k",  # Set audio bitrate
                    "-y",
                    output_file
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Failed to merge video and audio: {result.stderr}")
                    # Fallback: try with different audio encoding parameters
                    cmd = [
                        "ffmpeg",
                        "-i", temp_video_file,
                        "-i", temp_audio_file,
                        "-c:v", "copy",  # Copy video stream
                        "-c:a", "aac",   # Re-encode audio to AAC
                        "-ar", "44100",  # Set audio sample rate
                        "-y",
                        output_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"Second attempt to merge video and audio failed: {result.stderr}")
                        # Final fallback: copy video only if audio merge fails
                        cmd = [
                            "ffmpeg",
                            "-i", temp_video_file,
                            "-c", "copy",
                            "-y",
                            output_file
                        ]
                        subprocess.run(cmd)
                
            #     # Remove temporary audio files
            #     os.remove(temp_audio_file)
            #     os.remove(audio_list_file)

            # # Remove temporary video file
            # os.remove(temp_video_file)
        else:
            # No audio to merge, just merge videos
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264",  # Re-encode video for better compatibility
                "-preset", "ultrafast",  # Use fast encoding preset
                "-crf", "23",  # Constant rate factor for quality
                "-c:a", "aac",  # Re-encode audio to AAC if present
                "-y",
                output_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Video concatenation failed: {result.stderr}")
                # Fallback to copy if re-encoding fails
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    "-y",
                    output_file
                ]
                subprocess.run(cmd)

        # os.remove(list_file)
        # print(f"video merged to {output_file}")
        return output_file

def save_last_frame_decord(video_path, output_path=None):
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"video file is not exist: {video_path}")

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            
            total_frames = len(vr)
            if total_frames == 0:
                raise ValueError("can not detect frame from video")

            last_frame = vr[-1].asnumpy()  # uint8 (H, W, 3)

        except DECORDError as e:
            raise RuntimeError(f"video decoded fail: {str(e)}")
        except IndexError:
            raise RuntimeError("index error")
        finally:
            if 'vr' in locals():
                del vr

        if output_path is None:
            dir_name = os.path.dirname(video_path)
            file_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(dir_name, f"{file_name}_last_frame.png")

        try:
            img = Image.fromarray(last_frame)
            img.save(output_path, format='PNG', compress_level=0)
            print(f"save the last frame to: {output_path}")
            return output_path
        except IOError as e:
            raise RuntimeError(f"failed to save frame: {str(e)}")
    
def extract_frames(video_path: str, output_dir: str, target_fps: int = 1, grey=False) -> None:
    """
    Extract video frames at specified FPS using decord and save losslessly with Pillow
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        target_fps: Target frames per second (default=1)
    """
    try:
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video reader
        vr = VideoReader(video_path, ctx=cpu(0))
        
        # Get video metadata
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        # Calculate frame sampling interval
        interval = max(1, round(original_fps / target_fps))
        
        # Generate frame indices to extract
        frame_indices = np.arange(0, total_frames, interval)
        
        # Process each selected frame
        for second, idx in enumerate(frame_indices):
            # Read frame (BGR format from decord)
            bgr_frame = vr[idx].asnumpy()
            
            # Convert BGR to RGB color space
            # rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(bgr_frame)
            
            filename = os.path.join(output_dir, f"{idx:05d}.png")
            
            if grey:
                image = image.resize(
                    (512, 512),
                    resample=Image.Resampling.LANCZOS  # 或 Image.BICUBIC、Image.BILINEAR
                )
            # Save with lossless compression (PNG format)
            image.save(
                filename,
                format="PNG",
                optimize=True,     # Enable optimization
                compress_level=0   # Maximum compression (0-9)
            )
            
        print(f"Successfully extracted {len(frame_indices)} frames to {output_dir}")

        return original_fps

    except Exception as e:
        print(f"Frame extraction failed: {str(e)}")
        raise



def get_video_duration_seconds(video_path: Union[str, os.PathLike]) -> float:
    p = str(video_path)
    if not (p.startswith("http://") or p.startswith("https://")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"No such file: {p}")

    vr = VideoReader(p, ctx=cpu(0))
    n_frames = len(vr)
    if n_frames == 0:
        raise ValueError("Empty video (0 frames).")

    try:
        ts = vr.get_frame_timestamp(n_frames - 1)
        if isinstance(ts, (list, tuple, np.ndarray)):
            end_ts = ts[-1]
        else:
            end_ts = ts
        dur = float(end_ts)
        if math.isfinite(dur) and dur > 0:
            return dur
    except Exception:
        pass

    try:
        fps = float(vr.get_avg_fps())
        if fps > 0:
            return n_frames / fps
    except Exception:
        pass

    raise RuntimeError("Failed to determine video duration via decord.")


def format_hhmmss_ms(seconds: float) -> str:
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"



def stitch_frames_to_video(image_folder: str, output_video_path: str, fps: int = 30) -> None:
    """
    Stitches a sequence of image frames from a folder into a video file using imageio.

    Args:
        image_folder (str): Path to the directory containing the image frames.
        output_video_path (str): Path to save the output video file (e.g., 'output.mp4').
        fps (int): Frames per second for the output video.
    """
    try:
        import natsort
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        sorted_images = natsort.natsorted(images)
    except ImportError:
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        sorted_images = sorted(images)

    if not sorted_images:
        print(f"not found any images in'{image_folder}' ")
        return

    file_paths = [os.path.join(image_folder, img) for img in sorted_images]

    print(f"start concat {len(file_paths)} ")

    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for file_path in file_paths:
            try:
                image = imageio.imread(file_path)
                writer.append_data(image)
            except Exception as e:
                print(f"warning: jumped frame {file_path}, error: {e}")

    print(f"finished concat, saved: {output_video_path}")



def split_video_by_windows(video_path: str, time_windows: list, output_dir: str = None):
    """
    Split video into segments based on time windows and keep the parts between windows.
    
    Args:
        video_path: Path to input video file
        time_windows: List of time windows in format [[start1,end1],[start2,end2],...] (in seconds)
        output_dir: Directory to save segments (default: same directory as input video)
        
    Returns:
        list: Paths to all created video segments (both in-window and between-window segments)
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    if not time_windows:
        raise ValueError("Time windows list cannot be empty")
        
    # Validate time windows
    for window in time_windows:
        if len(window) != 2:
            raise ValueError(f"Invalid time window format: {window}. Expected [start, end]")
        if window[0] >= window[1]:
            raise ValueError(f"Invalid time window: start ({window[0]}) must be before end ({window[1]})")
            
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video base name without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Get video duration
    vr = VideoReader(video_path, ctx=cpu(0))
    duration = len(vr) / vr.get_avg_fps()
    del vr
    
    # Sort time windows by start time
    time_windows = sorted(time_windows, key=lambda x: x[0])
    
    # Generate all segments (both in-window and between-window)
    all_segments = []
    in_window_segments = []
    prev_end = 0.0
    
    for i, (start, end) in enumerate(time_windows):
        # Add the segment before this window (if any)
        if start > prev_end:
            output_path = os.path.join(output_dir, f"{base_name}_{prev_end:.1f}-{start:.1f}.mp4")
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", str(prev_end),
                "-to", str(start),
                "-c", "copy",
                "-y",
                output_path
            ]
            try:
                subprocess.run(cmd, check=True)
                all_segments.append(output_path)
            except subprocess.CalledProcessError as e:
                print(f"Failed to save between-window segment {i}: {e}")
        
        # Add the current window segment
        output_path = os.path.join(output_dir, f"{base_name}_{start:.1f}-{end:.1f}.mp4")
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            "-y",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            all_segments.append(output_path)
            in_window_segments.append(output_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to split segment {i+1}: {e}")
        
        prev_end = end
    
    # Add the segment after last window (if any)
    if prev_end < duration:
        output_path = os.path.join(output_dir, f"{base_name}_{prev_end:.1f}-{duration:.1f}.mp4")
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(prev_end),
            "-to", str(duration),
            "-c", "copy",
            "-y",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            all_segments.append(output_path)
        except subprocess.CalledProcessError as e:
            print(f"Failed to save final between-window segment: {e}")
            
    return all_segments, in_window_segments


async def storyboard_generate(user_prompt: str, gentype: str=None) -> dict:
    """
    Transforms a brief story outline into a detailed storyboard, complete with character introductions (including physical characteristics) and descriptions of various scene segments.
    This tool helps in pre-visualizing video narratives.
    
    Args:
        user_prompt: Users' input text, used to generate a video story script.
        
    Returns:
        dict: A dictionary representing the generated storyboard, with the following keys:
              - 'characters' (list): A list of character objects, each containing their physical characteristics, and descriptions.
              - 'shots' (list): A list of shot objects, defining the video sequence.
              - 'style' (str): A concise description of the overall visual style.
    """
    if gentype == "entity2video":
        with open("prompts/generation/storyboard_gen/entity2video.txt", "r") as f:
            refine_prompt = f.read()
    else:
        with open("prompts/generation/storyboard_gen/storyboard_gen.txt", "r") as f:
            refine_prompt = f.read()
    
    query_content = f"{refine_prompt}\n\n{user_prompt}"

    messages = [
        {"role": "user", "content": query_content}
    ]
    
    response = query_openai(
        api_key=llm_config.get('openai_api_key', None),
        model=llm_config.get('model', 'gpt-5-2025-08-07'),
        messages=messages,
        max_tokens=8192
    )

    response_text = response["content"]
    
    # Extract JSON from markdown code block if present
    import re
    json_match = re.search(r"```json\n(.*)\n```", response_text, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        json_string = response_text # Assume it's pure JSON if no markdown block

    # Parse the JSON response and populate the server_timeline
    try:
        storyboard_data = json.loads(json_string)

        return storyboard_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}")
        return {"error": f"Failed to parse storyboard: {e}", "raw_response": response_text}
    except Exception as e:
        logger.error(f"Error processing storyboard data: {e}")
        return {"error": f"Error processing storyboard data: {e}", "raw_response": response_text}

def split_video_by_fps(
    video_path: str,
    fps: float,
    clip_max_frames: int
) -> List[np.ndarray]:
    """
    use decord to read video, sample frames at given fps, and split into clips of max clip_max_frames each.
    """
    vr = VideoReader(video_path, ctx=cpu())
    total_frames = len(vr)
    if total_frames == 0:
        return []

    orig_fps = float(vr.get_avg_fps())
    eff_fps = min(fps, orig_fps)

    step = orig_fps / eff_fps
    indices = np.round(np.arange(0, total_frames, step)).astype(np.int64)
    indices = np.clip(indices, 0, total_frames - 1)
    _, unique_pos = np.unique(indices, return_index=True)
    indices = indices[np.sort(unique_pos)]

    if indices.size == 0:
        return []

    clips: List[np.ndarray] = []
    for start in range(0, len(indices), clip_max_frames):
        batch_idx = indices[start:start + clip_max_frames]
        frames_nd = vr.get_batch(batch_idx)
        frames_np = frames_nd.asnumpy()
        clips.append(frames_np)

    return clips



def encode_clips_to_base64(clips: List[np.ndarray], image_format: str = "JPEG") -> List[List[str]]:
    all_clips_base64: List[List[str]] = []

    for clip in clips:
        clip_base64_frames = []
        for frame in clip:
            img = Image.fromarray(frame.astype("uint8"))
            buf = io.BytesIO()
            img.save(buf, format=image_format)
            byte_data = buf.getvalue()
            base64_str = "data:image/{};base64,".format(image_format.lower()) + base64.b64encode(byte_data).decode("utf-8")
            clip_base64_frames.append(base64_str)
        all_clips_base64.append(clip_base64_frames)

    return all_clips_base64

