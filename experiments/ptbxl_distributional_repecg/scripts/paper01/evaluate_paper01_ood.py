"""Retired unsafe legacy OOD evaluator for Paper 01.

Now exits cleanly so grid scripts don't crash.
Real evaluation uses checkpoint-backed native-task evaluation (see paper14 pattern).
"""

import sys
print("SKIPPED: retired OOD evaluator. Use checkpoint-backed native-task evaluation.", file=sys.stderr)
sys.exit(0)
