import os
import yaml
import json
import torch
import decord
from datetime import datetime
from ultralytics import SAM
from PIL import Image
from mcp.server.fastmcp import FastMCP

from mcp_tools.base import ToolResponse, setup_logger
from utils.video_process import split_video_by_fps, encode_clips_to_base64
from mcp_tools.video_tracking import video_referring_segmentation
from utils.query_llm import prepare_multimodal_messages_openai_format, query_openrouter, multimodal_query



# Load configuration
# config_path = "config/mcp_tools_config/config.yaml"
# os.chdir(os.path.dirname(os.path.dirname(__file__)))
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

video_understanding_config = config.get('video_understanding', {})

# Configure logging
logger = setup_logger(__name__, "logs/mcp_tools", "video_understanding.log")
logger.info(f"Loaded video_understanding_config: {video_understanding_config}")

# Create an MCP server
mcp = FastMCP("Video_Understanding_Server")


@mcp.tool()
def vision2text_gen(prompt: str, multimodal_path: str, type: str) -> dict:
    """
    Analyzes and describes the content of a video or image based on a given prompt, converting visual information into text.
    This tool is useful for understanding ambiguous or complex visual inputs, providing detailed textual descriptions of the content.

    Args:
        prompt (str): User's instruction.
        multimodal_path (str): The path of the video or image.
        type (str): The type of the multimodal input, either "video" or "image".

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the vision content was understood successfully, False otherwise.
              - 'message' (str, optional): The details of the vision content if successful.
              - 'error' (str, optional): An error message if the operation failed.
    """
    try:
        with torch.no_grad():
            if type == "video":
                content = multimodal_query(prompt, video_path=multimodal_path)
            elif type == "image":
                content = multimodal_query(prompt, image_path=multimodal_path)
            else:
                return ToolResponse(
                    success=False,
                    message="The type of the multimodal input should be either 'video' or 'image'."
                )

        return ToolResponse(
            success=True,
            message="Vision content understood successfully.",
            content=content
        )
    except Exception as e:
        return ToolResponse(
            success=False,
            message=f"An error occurred: {str(e)}"
        )



if __name__ == "__main__":
    mcp.run(transport="stdio")

