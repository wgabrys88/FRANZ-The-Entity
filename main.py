"""Agent loop.

Runs an infinite loop where each turn loads prior VLM output (the story),
executes actions via execute.py, captures a screenshot, sends everything
to the VLM, and stores the raw response as the new story.

Reads FRANZ_RUN_DIR from environment (set by panel.py) to locate all
per-execution artifacts. Falls back to a timestamped subdirectory under
panel_log/ if not set. All tunable parameters come from config.py which
is hot-reloaded every turn.
"""

import importlib
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Final

import config as franz_config

API: Final = "http://localhost:1234/v1/chat/completions"
EXECUTE_SCRIPT: Final = Path(__file__).parent / "execute.py"

_run_dir_path = Path(os.environ.get("FRANZ_RUN_DIR", ""))
if not _run_dir_path.is_dir():
    _run_dir_path = Path(__file__).parent / "panel_log" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    _run_dir_path.mkdir(parents=True, exist_ok=True)
RUN_DIR_RESOLVED: Final = _run_dir_path
STATE_FILE: Final = RUN_DIR_RESOLVED / "state.json"

SYSTEM_PROMPT: Final = """\
You are a Python expert, computer control assistant and a teacher.
A user sends you a screenshot of their screen and a message. You reply with your analysis and a script proposal.

REPLY MUST ADHERE TO THE FOLLOWING TEMPLATE:

WHAT THE USER LEARNED SO FAR: (each item teaches the user how to deduct it independently)
1. (a discovery about the tools or the screen)
2. (what works, what doesn't)
3. (coordinates or positions confirmed)
4. (a technique that proved effective)
5. (a mistake the user made and the correction)
Keep minimum 5 items. Fix wrong ones. Drop obvious ones. Maximum 10.

SITUATION:
What is on the screen now and what changed. 3-4 sentences.

SPECIFIC ACTION PROPOSAL:
I decided, I would look at this situation like this -- (reasoning, what needs to happen, why). I wrote a single python script.

REPLY RULES:
- Provide exactly one python script per reply.
- Your visual interpretation may be imprecise. Consider that what you see might not be what you assume. Prefer exploring new approaches and positions over repeating familiar ones.
- If an action did not produce the expected result, try a different tool, different coordinates, or a completely different strategy.
- Do not hesitate. Commit to your proposal.\
"""


def _load_state() -> tuple[str, int]:
    try:
        o = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(o, dict):
            return str(o.get("story", "")), int(o.get("turn", 0))
    except Exception:
        pass
    return "", 0


def _save_state(turn: int, story: str, prev_story: str, raw: str, er: dict[str, object]) -> None:
    try:
        STATE_FILE.write_text(json.dumps({
            "turn": turn,
            "story": story,
            "prev_story": prev_story,
            "vlm_raw": raw,
            "executed": er.get("executed", []),
            "malformed": er.get("malformed", []),
            "ignored": er.get("ignored", []),
            "timestamp": datetime.now().isoformat(),
        }, indent=2), encoding="utf-8")
    except Exception:
        pass


def _sampling_dict() -> dict[str, float | int]:
    return {
        "temperature": float(franz_config.TEMPERATURE),
        "top_p": float(franz_config.TOP_P),
        "max_tokens": int(franz_config.MAX_TOKENS),
    }


def _infer(screenshot_b64: str, prev_story: str, feedback: str) -> str:
    payload: dict[str, object] = {
        "model": str(franz_config.MODEL),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prev_story}]},
            {"role": "user", "content": [
                {"type": "text", "text": feedback},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
            ]},
        ],
        **_sampling_dict(),
    }
    body_bytes = json.dumps(payload).encode()
    req = urllib.request.Request(API, body_bytes, {"Content-Type": "application/json"})
    delay = 0.5
    last_err: Exception | None = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=None) as resp:
                return json.load(resp)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2.0, 8.0)
    raise RuntimeError(f"VLM request failed after retries: {last_err}")


def _run_executor(raw: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(EXECUTE_SCRIPT)],
        input=json.dumps({"raw": raw, "run_dir": str(RUN_DIR_RESOLVED)}),
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}


def main() -> None:
    story, turn = _load_state()
    while True:
        turn += 1
        try:
            importlib.reload(franz_config)
        except Exception:
            pass
        prev_story = story
        er = _run_executor(prev_story)
        screenshot_b64 = str(er.get("screenshot_b64", ""))
        feedback = (
            str(er["feedback"])
            if "feedback" in er
            else "Runtime error: executor subprocess failed."
        )
        raw = _infer(screenshot_b64, prev_story, feedback)
        story = raw
        _save_state(turn, story, prev_story, raw, er)
        time.sleep(float(franz_config.LOOP_DELAY))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
