import yaml
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from mcp_tools.base import ToolResponse, setup_logger
from utils.wavespeed_api import audio_gen, speech_gen
from utils.query_llm import audio_prompt_gen, speech_prompt_gen

# Load configuration
# config_path = "config/mcp_tools_config/config.yaml"
# os.chdir(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

audio_gen_config = config.get('audio_gen', {})

# Configure logging
logger = setup_logger(__name__, "logs/mcp_tools", "audio_gen.log")
logger.info(f"Loaded audio_gen_config: {audio_gen_config}")

mcp = FastMCP("Audio_Generation_Server")

@mcp.tool()
async def generate_audio_for_video(video_path: str) -> ToolResponse:
    """
    Generates audio for a given video using AI audio generation with automatic prompt generation.
    
    This function takes a video file and automatically generates matching audio content.
    The audio will be synchronized with the video duration and saved with a timestamped filename.
    
    Args:
        video_path (str): Path to the input video file (local file path or URL)
    
    Returns:
        ToolResponse: A response object containing:
                     - success (bool): True if audio was generated successfully
                     - output_path (str): Path to the generated audio file
                     - message (str): Success or error message
                     - error (str, optional): Error details if generation failed
    """
    try:
        # Get API key from config
        api_key = audio_gen_config.get("wavespeed_api")
        if not api_key:
            return ToolResponse(
                success=False,
                message="API key not found in configuration",
                error="Missing wavespeed_api key in audio_gen config"
            )
        
        # Validate video file exists if it's a local path
        if not video_path.startswith('http') and not os.path.exists(video_path):
            return ToolResponse(
                success=False,
                message=f"Video file not found: {video_path}",
                error=f"Input video file does not exist: {video_path}"
            )
        
        # Get default values from config
        default_model = audio_gen_config.get("default_model", "mmaudio-v2")
        default_duration = audio_gen_config.get("duration", 5)
        default_guidance_scale = audio_gen_config.get("guidance_scale", 4.5)
        default_num_inference_steps = audio_gen_config.get("num_inference_steps", 25)
        
        # Auto-generate save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_path = audio_gen_config.get("base_output_path", "results/audio")
        os.makedirs(base_output_path, exist_ok=True)
        save_path = f"{base_output_path}/audio_{timestamp}.wav"
        
        # Use a generic prompt for automatic audio generation# 动态生成音频提示词
        try:
            prompt = audio_prompt_gen(video_path)
            logger.info(f"Generated audio prompt: {prompt[:100]}...")
        except Exception as e:
            logger.warning(f"Failed to generate dynamic prompt, using fallback: {e}")
            prompt = "Generate appropriate background audio and sound effects for this video content"
        
        # Generate audio using wavespeed API
        logger.info(f"Generating audio for video: {video_path}")
        result = audio_gen(
            api_key=api_key,
            prompt=prompt,
            video_url=video_path,
            model=default_model,
            save_path=save_path,
            provider="wavespeed-ai",
            duration=default_duration,
            guidance_scale=default_guidance_scale,
            mask_away_clip=False,
            negative_prompt="",
            num_inference_steps=default_num_inference_steps
        )
        
        if result and result.get('success'):
            logger.info(f"Audio generation successful: {result.get('output_path')}")
            return ToolResponse(
                success=True,
                output_path=result.get('output_path'),
                message=result.get('message', 'Audio generated successfully')
            )
        else:
            error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
            logger.error(f"Audio generation failed: {error_msg}")
            return ToolResponse(
                success=False,
                message="Audio generation failed",
                error=error_msg
            )
            
    except Exception as e:
        logger.error(f"Exception in generate_audio_for_video: {str(e)}")
        return ToolResponse(
            success=False,
            message="An error occurred during audio generation",
            error=str(e)
        )


@mcp.tool()
async def generate_speech(text: str = None, video_path: str = None) -> ToolResponse:
    """
    Generates speech audio from text using AI text-to-speech synthesis with default settings.
    
    This function converts text into natural-sounding speech using predefined voice characteristics
    and audio parameters from the configuration. Can also automatically generate speech based on video content.
    
    Args:
        text (str, optional): The text content to convert to speech
        video_path (str, optional): Path to video file for automatic speech generation
    
    Note:
        Either text or video_path must be provided. If video_path is provided, text will be auto-generated.
    
    Returns:
        ToolResponse: A response object containing:
                     - success (bool): True if speech was generated successfully
                     - output_path (str): Path to the generated speech audio file
                     - message (str): Success or error message
                     - error (str, optional): Error details if generation failed
    """
    try:
        # Get API key from config
        api_key = audio_gen_config.get("wavespeed_api")
        if not api_key:
            return ToolResponse(
                success=False,
                message="API key not found in configuration",
                error="Missing wavespeed_api key in audio_gen config"
            )
        
        # Validate input parameters
        if not text and not video_path:
            return ToolResponse(
                success=False,
                message="Either text or video_path must be provided",
                error="Input text or video_path is required for speech generation"
            )
        
        # If video_path is provided, generate text from video
        if video_path:
            # Validate video file exists if it's a local path
            if not video_path.startswith('http') and not os.path.exists(video_path):
                return ToolResponse(
                    success=False,
                    message=f"Video file not found: {video_path}",
                    error=f"Input video file does not exist: {video_path}"
                )
            
            try:
                # Generate speech text based on video content using specialized speech prompt
                text = speech_prompt_gen(video_path)
                logger.info(f"Generated speech text from video: {text[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to generate speech text from video, using fallback: {e}")
                text = "This is an automatically generated speech for the video content."
        
        # Validate final text
        if not text or not text.strip():
            return ToolResponse(
                success=False,
                message="Generated text content is empty",
                error="Failed to generate valid text content for speech"
            )
        
        # Get default values from config
        default_voice = audio_gen_config.get("default_voice", "Wise_Woman")
        default_emotion = audio_gen_config.get("default_emotion", "neutral")
        speech_model = audio_gen_config.get("speech_model", "speech-2.5-turbo-preview")
        
        # Auto-generate save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_path = audio_gen_config.get("base_output_path", "results/audio")
        os.makedirs(base_output_path, exist_ok=True)
        save_path = f"{base_output_path}/speech_{timestamp}.wav"
        
        # Generate speech using wavespeed API
        logger.info(f"Generating speech for text: {text[:50]}...")
        result = speech_gen(
            api_key=api_key,
            prompt=text,
            voice_id=default_voice,
            emotion=default_emotion,
            english_normalization=False,
            pitch=0,
            speed=1.0,
            volume=1.0,
            save_path=save_path,
            provider="minimax",
            model=speech_model
        )
        
        if result and result.get('success'):
            logger.info(f"Speech generation successful: {result.get('output_path')}")
            return ToolResponse(
                success=True,
                output_path=result.get('output_path'),
                message=result.get('message', 'Speech generated successfully')
            )
        else:
            error_msg = result.get('error', 'Unknown error occurred') if result else 'No result returned'
            logger.error(f"Speech generation failed: {error_msg}")
            return ToolResponse(
                success=False,
                message="Speech generation failed",
                error=error_msg
            )
            
    except Exception as e:
        logger.error(f"Exception in generate_speech: {str(e)}")
        return ToolResponse(
            success=False,
            message="An error occurred during speech generation",
            error=str(e)
        )


if __name__ == "__main__":
    mcp.run()