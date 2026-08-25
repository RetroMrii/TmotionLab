from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError
from urllib.request import (
    Request,
    urlopen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Smoke-test a running TmotionLab API.")
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    return parser.parse_args()


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    data = None

    headers = {
        "Accept": "application/json",
    }

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

        headers["Content-Type"] = "application/json"

    request = Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8")

        raise RuntimeError(f"{method} {url} failed with {exc.code}: {body}") from exc


def main() -> None:
    args = parse_args()

    base = args.base_url.rstrip("/")

    health = request_json(f"{base}/health")

    assert health["status"] == "ok"

    print("[PASS] /health")

    models = request_json(f"{base}/models")

    assert len(models["models"]) == 3

    print("[PASS] /models")

    prediction = request_json(
        f"{base}/predict",
        method="POST",
        payload={"text": ("I feel extremely nervous about tomorrow.")},
    )

    assert len(prediction["top_3"]) == 3

    print("[PASS] /predict")

    comparison = request_json(f"{base}/compare/dair_test_001837")

    assert comparison["example_id"] == "dair_test_001837"

    print("[PASS] /compare")

    interpretation = request_json(
        f"{base}/interpret",
        method="POST",
        payload={
            "text": ("I feel extremely nervous about tomorrow."),
            "top_k": 5,
        },
    )

    assert len(interpretation["top_text_spans"]) <= 5

    print("[PASS] /interpret")

    errors = request_json(f"{base}/errors?limit=5")

    assert errors["total"] == 207

    print("[PASS] /errors")

    confusions = request_json(f"{base}/errors/confusions?limit=5")

    assert len(confusions["items"]) == 5

    print("[PASS] /errors/confusions")

    print("[OK] TmotionLab API smoke test passed")


if __name__ == "__main__":
    main()
