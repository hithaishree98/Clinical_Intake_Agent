import json
import datetime

def log_event(event: str, level: str = "info", **fields):
    payload = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "level": level,
        "event": event,
        **{k: v for k, v in fields.items() if v is not None},
    }
    print(json.dumps(payload, ensure_ascii=False))
