import cv2
import os

def extract_all_frames(video_path, output_dir="frames", image_ext="jpg"):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频: {video_path}")
    print(f"FPS: {fps}")
    print(f"预计总帧数: {total_frames}")
    print(f"输出目录: {output_dir}")

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        output_path = os.path.join(
            output_dir,
            f"frame_{frame_index:06d}.{image_ext}"
        )

        cv2.imwrite(output_path, frame)

        frame_index += 1

        if frame_index % 100 == 0:
            print(f"已导出 {frame_index}/{total_frames} 帧")

    cap.release()

    print(f"\n完成，共导出 {frame_index} 帧。")


if __name__ == "__main__":
    extract_all_frames(
        video_path="/Users/kevin/Desktop/case2.mp4",
        output_dir="/Users/kevin/Desktop/frames_exp",
        image_ext="png"
    )