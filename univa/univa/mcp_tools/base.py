import os
import logging
from pydantic import BaseModel
from typing import Optional, List, Any


class ToolResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    content: Optional[Any] = None
    output_path: Optional[str|List[str]] = None
    
    class Config:
        extra = "allow"


def setup_logger(name: str, log_dir: str, log_file: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)