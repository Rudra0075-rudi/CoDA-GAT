"""
CoDA-GAT — Block B6: pre-registration timestamp lock.

Hashes preregistration_b6.md and writes preregistration_b6.LOCKED.json
containing that hash plus an ISO-8601 UTC timestamp. If the .md file is
edited afterward, its hash will no longer match the locked record --
that mismatch is what makes this tamper-evident. Run this ONCE, after
the pre-registration text is final, and BEFORE Block B7 runs.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG_PATH = REPO_ROOT / "preregistration_b6.md"
LOCK_PATH = REPO_ROOT / "preregistration_b6.LOCKED.json"


def main():
    if not PREREG_PATH.exists():
        print(f"[B6-lock] ERROR: {PREREG_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)

    if LOCK_PATH.exists():
        with open(LOCK_PATH) as f:
            existing = json.load(f)
        print(f"[B6-lock] REFUSED: {LOCK_PATH} already exists.")
        print(f"[B6-lock]   existing hash:      {existing.get('sha256')}")
        print(f"[B6-lock]   existing timestamp: {existing.get('locked_at_utc')}")
        print("[B6-lock] A pre-registration lock is meant to be written once. If you need to "
              "revise the document, that is a new pre-registration (rename/version the .md and "
              "lock file, e.g. preregistration_b6_v2.md) -- do not overwrite this one.")
        sys.exit(1)

    content_bytes = PREREG_PATH.read_bytes()
    sha256 = hashlib.sha256(content_bytes).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "filename": PREREG_PATH.name,
        "sha256": sha256,
        "locked_at_utc": timestamp,
        "file_size_bytes": len(content_bytes),
    }

    with open(LOCK_PATH, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")

    print("[B6-lock] Pre-registration locked.")
    print(f"[B6-lock]   file:      {PREREG_PATH.name}")
    print(f"[B6-lock]   sha256:    {sha256}")
    print(f"[B6-lock]   timestamp: {timestamp}")
    print(f"[B6-lock]   written to: {LOCK_PATH}")
    print("[B6-lock] Do not edit preregistration_b6.md after this point -- any change will make "
          "its hash no longer match this record. B7 may now be run.")


if __name__ == "__main__":
    main()
