import sys
from pathlib import Path
import os

def setup_sys_path():
    """
    Set up the sys.path to include the project root directory for imports.
    Works in both regular scripts and interactive environments like Jupyter Notebook.
    """
    try:
        script_path = Path(__file__).resolve()
    except NameError:
        script_path = Path(os.getcwd()).resolve()

    # Adjust level based on folder structure
    project_root = script_path.parent  # Two levels up to the project root # repackaging the repo seems to crash the Figure 5 and Table 3 script, removing the [2] fixes that; to watch out in case other imports crash
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
