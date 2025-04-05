
import logging
from dotenv import load_dotenv

# try:
#     import torch
#     import intel_extension_for_pytorch as ipex
# except:
#     pass

from .app import RizzApp
from .config import RizzConfig

def main() -> int:
    load_dotenv()

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    for logger_name in ("praw", "prawcore"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    config = RizzConfig()
    app = RizzApp(config)
    app.launch()
    return 0
