"""Hold a capacity child before exec until its process identity is durable."""

from __future__ import annotations

import os
import sys


READY_TOKEN = b"\x01"
PARENT_DIED_BEFORE_RELEASE = 125


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: gm_capacity_child_gate.py READ_FD COMMAND [ARG ...]")
    read_fd = int(sys.argv[1])
    if read_fd <= 2:
        raise SystemExit("capacity child gate requires a dedicated inherited file descriptor")
    command = sys.argv[2:]
    try:
        token = os.read(read_fd, 1)
    finally:
        os.close(read_fd)
    if token != READY_TOKEN:
        raise SystemExit(PARENT_DIED_BEFORE_RELEASE)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
