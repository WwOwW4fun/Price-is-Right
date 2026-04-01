import json
import logging
import re
import sys
import threading
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.deal_agent_framework import DealAgentFramework


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
AGENT_PATTERN = re.compile(r"\[(.*?)\]")
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000


def strip_ansi(value):
    return ANSI_PATTERN.sub("", value or "")


def serialize_opportunity(opportunity):
    if hasattr(opportunity, "model_dump"):
        raw = opportunity.model_dump()
    else:
        raw = opportunity.dict()

    price = raw["deal"]["price"]
    estimate = raw["estimate"]
    discount = raw["discount"]
    discount_percent = round((discount / estimate) * 100, 1) if estimate else 0.0

    return {
        "product": raw["deal"]["product_description"],
        "price": price,
        "estimate": estimate,
        "discount": discount,
        "discountPercent": discount_percent,
        "url": raw["deal"]["url"],
        "source": urlparse(raw["deal"]["url"]).netloc.replace("www.", "") or "Unknown",
    }


def build_columns(deals):
    recent = list(reversed(deals[-4:]))
    best = sorted(deals, key=lambda deal: deal["discount"], reverse=True)[:4]
    alerted = list(reversed(deals[-4:]))
    return [
        {"title": "Recent Opportunities", "items": recent},
        {"title": "Best Discounts", "items": best},
        {"title": "Alerted Deals", "items": alerted},
    ]


def build_alerts(deals):
    alerts = []
    for deal in list(reversed(deals[-3:])):
        alerts.append(
            {
                "title": f"{deal['source']} alert ready",
                "body": (
                    f"{deal['product'][:110]} now shows an estimated "
                    f"${deal['discount']:.2f} edge against the model price."
                ),
            }
        )
    return alerts


def build_summary(deals, running, last_run_at):
    best_discount = max((deal["discountPercent"] for deal in deals), default=0.0)
    best_deal = max(deals, key=lambda deal: deal["discountPercent"], default=None)
    return [
        {
            "label": "System Status",
            "value": "Running" if running else "Idle",
            "note": "4 agents active" if running else "Waiting for next run",
        },
        {
            "label": "Deals Found",
            "value": str(len(deals)),
            "note": "Saved opportunities in memory",
        },
        {
            "label": "Best Discount",
            "value": f"{best_discount:.1f}%",
            "note": best_deal["source"] if best_deal else "No deal yet",
        },
        {
            "label": "Last Run",
            "value": last_run_at or "Never",
            "note": "Background pipeline",
        },
    ]


def build_timeline(logs):
    return list(reversed(logs))


def classify_level(message):
    lowered = message.lower()
    if "error" in lowered or "failed" in lowered:
        return "highlight"
    if "ready" in lowered or "completed" in lowered or "best deal" in lowered:
        return "active"
    return "normal"


def parse_log(message):
    clean = strip_ansi(message)
    bracketed = AGENT_PATTERN.findall(clean)
    timestamp = bracketed[0] if bracketed else ""
    agent = bracketed[-1] if bracketed else "Agent System"
    description = clean.split(f"[{agent}] ", 1)[-1] if f"[{agent}] " in clean else clean
    short_time = timestamp[11:16] if len(timestamp) >= 16 else "--:--"
    return {
        "time": short_time,
        "agent": agent,
        "message": description.strip(),
        "level": classify_level(description),
    }


class ApiLogHandler(logging.Handler):
    def __init__(self, app_state):
        super().__init__()
        self.app_state = app_state

    def emit(self, record):
        rendered = self.format(record)
        parsed = parse_log(rendered)
        with self.app_state.lock:
            self.app_state.logs.append(parsed)


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.framework = None
        self.running = False
        self.last_run_at = None
        self.error = None
        self.logs = deque(maxlen=18)

    def get_framework(self):
        if self.framework is None:
            self.framework = DealAgentFramework()
        return self.framework

    def snapshot(self):
        framework = self.get_framework()
        deals = [serialize_opportunity(item) for item in framework.memory]
        return {
            "running": self.running,
            "error": self.error,
            "mode": "Autonomous scan",
            "lastRunAt": self.last_run_at,
            "summaryCards": build_summary(deals, self.running, self.last_run_at),
            "dealColumns": build_columns(deals),
            "agentTimeline": build_timeline(list(self.logs)),
            "alerts": build_alerts(deals),
        }

    def run_agents(self):
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.error = None
            self.last_run_at = datetime.now().strftime("%b %d, %I:%M %p")

        def worker():
            try:
                framework = self.get_framework()
                framework.run()
            except Exception as exc:
                logging.exception("Agent run failed")
                with self.lock:
                    self.error = str(exc)
            finally:
                with self.lock:
                    self.running = False

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return True


APP_STATE = AppState()
LOG_HANDLER = ApiLogHandler(APP_STATE)
LOG_HANDLER.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
)
root_logger = logging.getLogger()
if not any(isinstance(handler, ApiLogHandler) for handler in root_logger.handlers):
    root_logger.addHandler(LOG_HANDLER)


class ApiHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self._send_json({}, status=204)

    def do_GET(self):
        if self.path == "/api/status":
            self._send_json(APP_STATE.snapshot())
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path == "/api/run":
            started = APP_STATE.run_agents()
            self._send_json({"started": started, "state": APP_STATE.snapshot()})
            return
        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), ApiHandler)
    print(f"Agent API listening on http://{HTTP_HOST}:{HTTP_PORT}")
    server.serve_forever()
