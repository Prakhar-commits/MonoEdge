"""Sign and send a ticket webhook. Used for demos and the review call.

Example:
  python scripts/send_webhook.py --event-id evt_101 --title "Login fails" --priority high
"""
import argparse
import hashlib
import hmac
import json
import os
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--title", default="Login fails")
    parser.add_argument("--priority", default="high")
    parser.add_argument("--ticket-id", default=None, help="defaults to TCK-<event-id>")
    parser.add_argument("--type", default="ticket.created")
    parser.add_argument("--bad-signature", action="store_true")
    parser.add_argument("--url", default="http://localhost:8000/webhooks/tickets")
    args = parser.parse_args()

    secret = os.environ.get("WEBHOOK_SECRET")
    if not secret:
        print("Set WEBHOOK_SECRET in your environment first.", file=sys.stderr)
        return 1

    body = json.dumps(
        {
            "event_id": args.event_id,
            "type": args.type,
            "data": {
                "ticket_id": args.ticket_id or f"TCK-{args.event_id}",
                "title": args.title,
                "priority": args.priority,
            },
        },
        separators=(",", ":"),
    ).encode()
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if args.bad_signature:
        signature = "0" * 64

    response = httpx.post(
        args.url,
        content=body,  # send the exact bytes we signed
        headers={"Content-Type": "application/json", "X-Signature": f"sha256={signature}"},
    )
    print(response.status_code, response.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
