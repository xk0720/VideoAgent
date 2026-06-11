import yaml
import json
import os
from typing import Dict, List
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from mcp_tools.base import ToolResponse, setup_logger
from utils.video_process import merge_videos, storyboard_generate, save_last_frame_decord
from utils.query_llm import refine_gen_prompt, audio_prompt_gen
from utils.image_process import download_image
from utils.wavespeed_api import text_to_video_generate, image_to_video_generate, frame_to_frame_video, text_to_image_generate, image_to_image_generate, audio_gen, hailuo_i2v_pro

# Load configuration
os.chdir(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

video_gen_config = config.get('video_gen', {})
image_gen_config = config.get('image_gen', {})
# base_output_path = video_gen_config.get('base_output_path', '/share/project/liangzy/liangzy2/UniVideo/generated_videos')

# Configure logging
logger = setup_logger(__name__, "logs/mcp_tools", "video_gen.log")
logger.info(f"Loaded video_gen_config: {video_gen_config}")

mcp = FastMCP("Video_Generation_Server")

@mcp.tool()
async def text2video_gen(prompt: str) -> str:
    """
    Generates a short video (approx. 5 seconds) from a text description.
    If segment_id is provided, it updates the corresponding segment in the server timeline.
    Otherwise, it generates a video without adding it to the timeline.

    Args:
        prompt (str): The prompt to generate the video.
        save_path (str): The path to save the generated video. Suggest numbering each video when naming them.

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("text_to_video")
    
    if model == "seedance":
        api_key = video_gen_config.get("wavespeed_api")
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = text_to_video_generate(api_key, prompt, save_path=save_path)
        
        return return_dict


@mcp.tool()
async def storyvideo_gen(prompt: str) -> ToolResponse:
    """
    Generates a story-based video from a text prompt by creating a storyboard, generating character images,
    creating keyframes, generating video segments, and merging them into a final video.

    This function follows these steps:
    1. Generates a storyboard based on the input prompt
    2. Extracts character descriptions from the storyboard and generates character images
    3. For each shot in the storyboard:
       - Generates keyframe images based on shot descriptions and characters
       - Converts keyframes to video segments
    4. Merges all video segments into a final video

    Args:
        prompt (str): A text description of the story or video content to generate.

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """

    model = video_gen_config.get("text_to_video")
    if model == "seedance":
        if save_dir is None:
            save_dir = f"infer/v2v/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)

        # 1.generate a storyboard first
        storyboard = await storyboard_generate(prompt)
        time = datetime.now().strftime("%m%d%H%M%S")
        with open(f"{time}_storyboard.json", "w") as f:
            json.dump(storyboard, f, indent=4)
        # 2.generate character images
        characters = storyboard.get("characters")
        characters_image_path = dict()

        # TODO: style
        style = storyboard.get("style")

        for character in characters:
            char_id = character.get("id")
            char_description = character.get("description")
            refined_char_description = refine_gen_prompt(char_description, media_type="character")

            api_key = image_gen_config.get("wavespeed_api")
            image_url = text_to_image_generate(api_key, refined_char_description, aspect_ratio="1:1")
            time = datetime.now().strftime("%m%d%H%M%S")
            image_save_path = f"{save_dir}/{time}_{char_id}.jpg"
            characters_image_path[char_id] = image_save_path
            download_image(image_url, save_path=image_save_path)

        # 3.key frame generation
        shots = storyboard.get("shots")
        shots_image_path = dict()
        for shot in shots:
            shot_id = shot.get("id")
            setting_description = shot.get("setting_description")
            plot_correspondence = shot.get("plot_correspondence")
            static_shot_description = shot.get("static_shot_description")
            onstage_characters = shot.get("onstage_characters")
            onstage_characters_image_path_list = [characters_image_path.get(char_id) for char_id in onstage_characters]

            shot_perspective_design = shot.get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"
            # refined_keyframe_prompt = refine_gen_prompt(keyframe_prompt, media_type="image")

            api_key = image_gen_config.get("wavespeed_api")
            if len(onstage_characters_image_path_list) == 0:
                image_url = text_to_image_generate(api_key, keyframe_prompt, aspect_ratio="4:3")
            else:
                image_url = image_to_image_generate(api_key, keyframe_prompt, onstage_characters_image_path_list, aspect_ratio="4:3")
            time = datetime.now().strftime("%m%d%H%M%S")
            image_save_path = f"{save_dir}/{time}_{shot_id}.jpg"
            shots_image_path[shot_id] = image_save_path
            download_image(image_url, save_path=image_save_path)

        # 4.generate video segments based on keyframes
        video_segment_list = []
        for idx, (keyframe_id, keyframe) in enumerate(shots_image_path.items()):
            api_key = video_gen_config.get("wavespeed_api")

            setting_description = shots[idx].get("setting_description")
            plot_correspondence = shots[idx].get("plot_correspondence")
            static_shot_description = shots[idx].get("static_shot_description")

            shot_perspective_design = shots[idx].get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"
            # refined_keyframe_prompt = refine_gen_prompt(keyframe_prompt, media_type="video")
            
            time = datetime.now().strftime("%m%d%H%M%S")
            save_path = f"{save_dir}/{time}_{keyframe_id}.mp4"
            return_dict = image_to_video_generate(api_key, keyframe_prompt, keyframe, save_path=save_path)

            if return_dict.get("success"):
                video_segment_list.append(return_dict.get("output_path"))
        # TODO: 5.audio integration
        # audio_list = []
        # api = video_gen_config.get("wavespeed_api")
        # for idx, shot in enumerate(shots):
        #     # audio_description = shot.get("audio_description")
            
        #     video_path = video_segment_list[idx]
        #     audio_description = audio_prompt_gen(video_path)
        #     time = datetime.now().strftime("%m%d%H%M%S")
        #     save_path = f"{save_dir}/{time}_{idx+1}.mp4"
        #     return_dict = audio_gen(
        #         api_key=api,
        #         prompt=audio_description,
        #         video_url=video_path,
        #         save_path=save_path,
        #     )

        #     if return_dict.get("success"):
        #         audio_list.append(return_dict.get("output_path"))
            

        # 6.concatenate video segments
        time = datetime.now().strftime("%m%d%H%M%S")
        prompt_name = prompt.replace(" ", "_")[:50]
        movie_save_path = f"{save_dir}/{time}_{prompt_name}.mp4"
        video_path = merge_videos(video_segment_list, output_file=movie_save_path)
        
        return ToolResponse(
            success=True,
            output_path=video_path,
            message="Video generated successfully."
        )



@mcp.tool()
async def entity2video(prompt: str, images: List[str]) -> str:
    """
    Generates a story-based video from a text prompt and a list of character images by creating a storyboard,
    using the provided images for characters, generating keyframes, creating video segments, and merging them
    into a final video.

    This function is similar to storyvideo_gen but allows users to provide their own character images instead
    of generating them from descriptions. It follows these steps:
    1. Generates a storyboard based on the input prompt
    2. Uses the provided images as character images
    3. For each shot in the storyboard:
       - Generates keyframe images based on shot descriptions and characters
       - Converts keyframes to video segments
    4. Merges all video segments into a final video

    Args:
        prompt (str): A text description of the story or video content to generate.
        images (List[str]): A list of file paths to character images that should be used in the video.

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """

    model = video_gen_config.get("text_to_video")
    if model == "seedance":
        save_dir = f"infer/v2v/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)

        prompt = f"{prompt}\nsource images path: {str(images)}"
        storyboard = await storyboard_generate(prompt, gentype="entity2video")
        characters_image_path = dict()

        # TODO: style
        style = storyboard.get("style")

        characters = storyboard.get("characters")
        for idx, character in enumerate(characters):
            char_id = f"char_{idx+1}"
            character_path = character.get("path")
            characters_image_path[char_id] = character_path

        shots = storyboard.get("shots")
        shots_image_path = dict()
        for shot in shots:
            shot_id = shot.get("id")
            setting_description = shot.get("setting_description")
            plot_correspondence = shot.get("plot_correspondence")
            static_shot_description = shot.get("static_shot_description")
            onstage_characters = shot.get("onstage_characters")
            onstage_characters_image_path_list = [characters_image_path.get(char_id) for char_id in onstage_characters]

            shot_perspective_design = shot.get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"
            api_key = image_gen_config.get("wavespeed_api")
            if len(onstage_characters_image_path_list) == 0:
                image_url = text_to_image_generate(api_key, keyframe_prompt)
            else:
                image_url = image_to_image_generate(api_key, keyframe_prompt, onstage_characters_image_path_list)
            # sleep(3)
            time = datetime.now().strftime("%m%d%H%M%S")
            image_save_path = f"{save_dir}/{time}_{shot_id}.jpg"
            shots_image_path[shot_id] = image_save_path
            download_image(image_url, save_path=image_save_path)

        video_segment_list = []
        for idx, (keyframe_id, keyframe) in enumerate(shots_image_path.items()):
            api_key = video_gen_config.get("wavespeed_api")

            setting_description = shots[idx].get("setting_description")
            plot_correspondence = shots[idx].get("plot_correspondence")
            static_shot_description = shots[idx].get("static_shot_description")

            shot_perspective_design = shots[idx].get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"
            
            time = datetime.now().strftime("%m%d%H%M%S")
            save_path = f"{save_dir}/{time}_{keyframe_id}.mp4"
            return_dict = image_to_video_generate(api_key, keyframe_prompt, keyframe, save_path=save_path)

            if return_dict.get("success"):
                video_segment_list.append(return_dict.get("output_path"))

        time = datetime.now().strftime("%m%d%H%M%S")
        prompt_name = prompt.replace(" ", "_")[:50]
        movie_save_path = f"{save_dir}/{time}_{prompt_name}.mp4"
        video_path = merge_videos(video_segment_list, output_file=movie_save_path)
        
        return ToolResponse(
            success=True,
            output_path=video_path,
            message="Video generated successfully."
        )




@mcp.tool()
async def image2video_gen(prompt: str, image_path: str) -> str:
    """
    Generates a short video (approx. 5 seconds) using a text prompt and an input image as a visual reference.
    This tool is useful for creating videos that maintain visual consistency with a provided image while incorporating new elements described in the prompt.

    Args:
        prompt (str): The prompt to generate the video.
        image_path (str): Input image path for use as video content reference, supporting common formats (.jpg/.png/.bmp, etc.).

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("image_to_video")
    
    if model == "seedance":
        api_key = video_gen_config.get("wavespeed_api")
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = image_to_video_generate(api_key, prompt, image_path, save_path=save_path)
        
        return return_dict


@mcp.tool()
async def video_extension(prompt: str, video_path: str) -> Dict:
    """
    Extends an existing video by generating new content based on a text prompt and the last frame of the input video.
    If segment_id is provided, it updates the corresponding segment in the server timeline.
    Otherwise, it generates a video without adding it to the timeline.

    Args:
        prompt (str): The prompt to generate the video.
        video_path (str): Input video path for use as video content reference, supporting common formats (.mp4/.avi, etc.).

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was extended successfully, False otherwise.
              - 'output_path' (str, optional): The path to the extended video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the extension failed.
    """
    time_ = datetime.now().strftime("%m%d%H%M%S")
    last_frame_save_path = save_last_frame_decord(video_path, f"results/{time_}_last_frame.png")
    # Pass segment_id to image2video_gen if it's meant to update a segment
    extend_result = await image2video_gen(prompt, last_frame_save_path) 
    if extend_result.get("success"):
        output_video_path = extend_result.get("output_path")

        return ToolResponse(
            success=True,
            output_path=output_video_path,
            message="Video extended and merged successfully."
        )


    else:
        return ToolResponse(
            success=False,
            message="Video extension failed at image2video generation step.",
        )


@mcp.tool()
async def frame2frame_video_gen(prompt: str, first_frame_path: str, last_frame_path: str) -> str:
    """
    Generates a short video (approx. 5 seconds) that transitions between a specified first frame and a last frame, guided by a text prompt.
    This tool is effective for creating dynamic action sequences or smooth transitions between two distinct visual states.

    Args:
        prompt (str): The prompt to generate the video.
        first_frame_path (str): The path to the first frame.
        last_frame_path (str): The path to the last frame.
        save_path (str): The path to save the generated video. Suggest numbering each video when naming them.

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("frame_to_frame_video")
    
    if model == "wan_api":
        api_key = video_gen_config.get("wavespeed_api")
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = hailuo_i2v_pro(api_key, prompt, first_frame_path, last_frame_path, save_path=save_path)
        
        return return_dict


@mcp.tool()
async def merge2videos(video_paths: list[str]):
    """
    Merges multiple video files into a single video file.

    Args:
        video_paths (list[str]): A list of paths to the video files to merge, or a folder path containing videos.

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str): The path to the merged video.
              - 'message' (str): A success message.
    """
    save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    _time = datetime.now().strftime("%m%d%H%M%S")
    save_path = f"{save_dir}/{_time}.mp4"
    video_path = merge_videos(video_paths, output_file=save_path)

    return ToolResponse(
        success=True,
        output_path=video_path,
        message="Videos merged successfully."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
