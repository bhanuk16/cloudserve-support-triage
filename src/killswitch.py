"""Kill switch: stop all automatic answering immediately, without a deployment.

    python -m src.killswitch on      # every ticket now escalates to a person
    python -m src.killswitch off
    python -m src.killswitch status

Mechanism: a flag file (KILL_SWITCH_FILE). The router checks it on every ticket, so it
takes effect on the very next ticket — no restart. Tickets already past routing finish
their current step but are still validated before sending. Engaging and releasing
are appended to storage/kill_switch_audit.log. Also settable at start-up with
AUTO_RESPOND_ENABLED=false, and over HTTP via POST /admin/kill-switch (ADMIN_TOKEN).
"""
from __future__ import annotations

import getpass
import sys
from datetime import datetime, timezone

from .config import settings


def set_kill_switch(engaged: bool, actor: str | None = None) -> None:
    f = settings.kill_switch_file
    f.parent.mkdir(parents=True, exist_ok=True)
    if engaged:
        f.write_text(datetime.now(timezone.utc).isoformat())
    elif f.exists():
        f.unlink()
    who = actor or getpass.getuser()
    with open(f.parent / "kill_switch_audit.log", "a", encoding="utf-8") as log:
        log.write(f"{datetime.now(timezone.utc).isoformat()} {'ENGAGED' if engaged else 'RELEASED'} by {who}\n")


def main(argv: list[str]) -> int:
    cmd = (argv[0] if argv else "status").lower()
    if cmd == "on":
        set_kill_switch(True)
    elif cmd == "off":
        set_kill_switch(False)
    elif cmd != "status":
        print("usage: python -m src.killswitch [on|off|status]")
        return 2
    engaged = settings.kill_switch_file.exists() or not settings.auto_respond_enabled
    print("kill switch ENGAGED: all tickets escalate" if engaged else "kill switch off: automatic answering enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
