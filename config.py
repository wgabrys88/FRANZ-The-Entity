"""Hot-reloadable configuration. Reloaded by main.py every turn.

All tunable parameters for the agent loop, executor, and capture
live here. Panel infrastructure settings remain in panel.py.
"""

TEMPERATURE: float = 0.7
TOP_P: float = 0.9
MAX_TOKENS: int = 900
MODEL: str = "qwen3-vl-2b-instruct-1m"
WIDTH: int = 512
HEIGHT: int = 288
SANDBOX: bool = True
EXECUTE_ACTIONS: bool = True
PHYSICAL_EXECUTION: bool = False
VISUAL_MARKS: bool = True
LOOP_DELAY: float = 0.01
RESTRICTED_EXEC: bool = True
