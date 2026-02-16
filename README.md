# Changelog (FRANZ-The-Entity) deep structural modifications

CONFIG.PY
=========
- New centralized configuration file replacing scattered constants across modules.
- Added: MODEL, WIDTH, HEIGHT, SANDBOX, EXECUTE_ACTIONS, PHYSICAL_EXECUTION,
  VISUAL_MARKS, LOOP_DELAY.
- Retained: TEMPERATURE, TOP_P, MAX_TOKENS, RESTRICTED_EXEC.
- Removed: MARKS_CLASSIC (classic mark mode eliminated), MARKS_CURSOR (replaced
  by single VISUAL_MARKS toggle).

MAIN.PY (agent loop)
====================
- System prompt completely rewritten:
  - Identity: "Python expert, computer control assistant and a teacher"
  - Template: WHAT THE USER LEARNED SO FAR (5-10 items, self-teaching) +
    SITUATION (what changed, 3-4 sentences) + SPECIFIC ACTION PROPOSAL
    with single python script.
  - Removed: cat-drawing task, code examples, tool documentation, coordinate
    reference, future-self memory explanation, all formatting examples.
  - Added curiosity/humility rule: "Your visual interpretation may be imprecise.
    Consider that what you see might not be what you assume."
  - Added anti-repetition framed as exploration preference.
  - All non-ASCII characters replaced with ASCII equivalents.
- All tunable parameters now read from config.py after hot-reload each turn.
  MODULE, WIDTH, HEIGHT, SANDBOX, etc. no longer hardcoded as module Finals.
- Simplified JSON passed to execute.py: only raw VLM text and run_dir.
  All settings read from config by executor subprocess.
- Removed: TOOLS_ENABLED dict, wants_screenshot from state, tools from state.
- Removed: MODEL, WIDTH, HEIGHT, VISUAL_MARKS, LOOP_DELAY, EXECUTE_ACTIONS,
  SANDBOX, PHYSICAL_EXECUTION as module-level Finals (now in config).
- Cleaned up RUN_DIR_RESOLVED initialization to avoid dual Final assignment.

EXECUTE.PY
==========
- Statement-by-statement execution via _exec_statements(). Each AST statement
  is compiled and executed independently. Expression statements are eval'd to
  capture return values (enables help() output in feedback). Non-expression
  statements are exec'd. Every statement gets a status regardless of earlier
  failures.
- Coordinate validation via _validate_coord(). All tool functions raise
  ValueError when coordinates are outside 0-1000 instead of silent clamping.
  Produces authentic Python errors the model learns from.
- help() function added to sandboxed namespace. Returns docstrings of all
  available functions when called with no arguments, or a single function's
  docstring when called with a function argument. Output captured via eval
  return value and shown in feedback. Uses " | " separator for compact
  single-line listing.
- Feedback format: ordered function-name -> status lines without numbering
  and without coordinate echo. Examples:
    left_click -> OK
    drag -> OK
    type -> TypeError: type() requires a string, got int
    help -> click(x, y) -- alias for left_click | drag(...) -- ... | ...
- "ADHERE TO YOUR INSTRUCTIONS" appended when multiple python blocks detected.
- "No actions performed.\nUse help() to list available functions." for missing
  or empty python blocks.
- click() now has its own implementation (not delegating to left_click) to
  ensure consistent fname matching in sandbox validation.
- All settings read from config.py. JSON stdin reduced to raw + run_dir only.
- Removed: _clean_exec_error (replaced by per-statement error capture),
  _namespace_help (tools are self-discovered via help()), screenshot() from
  namespace (screenshot happens every turn), wants_screenshot tracking,
  per-tool gating via tools dict (replaced by single EXECUTE_ACTIONS switch),
  traceback import.

CAPTURE.PY
==========
- Removed classic mark mode entirely:
  - Deleted _apply_marks_classic function (~120 lines).
  - Deleted _render_digit, _render_number functions.
  - Deleted _DIGITS precomputed list.
  - Deleted Canvas.arrow method.
  - Deleted Canvas.circle method (alpha-blended version with filled/thickness).
  - Deleted constants: MARK_FILL, MARK_OUTLINE, TRAIL_COLOR, BLACK.
  - Removed math import (only used by arrow method).
- Cursor mark mode is now the sole visual mark mode, controlled by single
  VISUAL_MARKS config toggle.
- All settings read from config.py. capture() function signature simplified
  to capture(actions, run_dir) -- width, height, sandbox, marks read from config.
- Removed _text_pixel_width (was dead code in original).
- Removed "screenshot" from action skip list in _sandbox_apply (tool no longer
  exists in namespace).
- Updated docstring to reflect cursor-only mark architecture.

PANEL.PY
========
- Updated docstring for clarity.
- No functional changes. Panel remains a transparent byte-for-byte proxy
  that logs, verifies SST, saves screenshots, and serves the dashboard.
  All feedback format changes are transparent to the proxy layer.

ARCHITECTURE
============
- Two-pipe single-source-of-truth design formalized:
  Pipe 1 (SST): VLM output -> untouched -> VLM input. The self-sustaining story.
  Pipe 2 (Reality): Python environment -> authentic feedback -> VLM. Ground truth.
- Feedback is authentic Python execution output: function names with OK or real
  error messages, in execution order, no numbering, no coordinate echo, no
  help dumps.
- Tool discovery is self-service via help() in the restricted Python namespace.
  No automatic tool documentation in error messages.
- Coordinate validation produces real Python ValueError instead of silent clamping.
- Statement-by-statement execution ensures every intended action is attempted
  and reported regardless of earlier failures.
- Configuration centralized in config.py, hot-reloaded every turn.
  Subprocess communication reduced to minimal per-turn data (raw text + run_dir).
- System prompt is generic (no hardcoded task), identity-framed (Python expert
  teacher), curiosity-driven (explore new approaches, acknowledge visual
  imprecision), and designed to produce a self-sustaining story template
  that can eventually replace the system prompt itself.
