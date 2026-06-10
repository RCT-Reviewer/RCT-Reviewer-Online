"""
RCT-Reviewer
"""

# Author:
#   Vihaan Sahu <pteroisvolitans12@gmail.com>


import logging
from pathlib import Path

__version__ = "1.0.0"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


DATA_ROOT = Path(__file__).parent.parent / "data"

def get_data_path(relative_path: str) -> Path:
    """Resolve path to model data files."""
    path = DATA_ROOT / relative_path
    if not path.exists():
        log.warning(f"Data file not found: {path}")
    return path