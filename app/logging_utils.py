import json
import logging
import datetime

logger = logging.getLogger("intake")

def log_event(event: str, level: str = "info", **fields):
    payload = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "level": level,
        "event": event,
        **{k: v for k, v in fields.items() if v is not None},
    }
    log = getattr(logger, level if level in ("debug", "info", "warning", "error", "critical") else "info")
    log(json.dumps(payload, ensure_ascii=False))