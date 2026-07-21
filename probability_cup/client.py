"""Minimal SportsPredict REST client for the Probability Cup.

The API key is read from the SPORTSPREDICT_KEY environment variable — never
hardcode it. Create a bot key in the SportsPredict app (Profile -> My Bots) and:

    export SPORTSPREDICT_KEY=sp_live_...

Endpoints used: list events/lobbies/matches/markets, join a lobby, batch-submit
and patch predictions, and read settled results (Brier score vs. crowd).
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
import json as _json

BASE_URL = "https://api.sportspredict.com/api/v1"


class SportsPredictClient:
    def __init__(self, key: str | None = None, base_url: str = BASE_URL):
        self.key = key or os.environ.get("SPORTSPREDICT_KEY")
        if not self.key:
            raise RuntimeError(
                "No API key. Set SPORTSPREDICT_KEY=sp_live_... in your environment."
            )
        self.base_url = base_url

    def _request(self, method: str, path: str, body: dict | None = None) -> dict | list:
        data = _json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.base_url}/{path}", data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.key}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req) as resp:
                return _json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code}: {e.read().decode()}") from e

    # --- discovery ---
    def events(self, limit: int = 10) -> list:
        return self._request("GET", f"events?limit={limit}")

    def lobbies(self, event_id: str) -> list:
        return self._request("GET", f"lobbies?event_id={event_id}")

    def join_lobby(self, lobby_id: str) -> dict:
        return self._request("POST", f"lobbies/{lobby_id}/join")

    def matches(self, event_id: str, lobby_id: str) -> list:
        return self._request("GET", f"matches?event_id={event_id}&lobby_id={lobby_id}")

    def markets(self, lobby_id: str, match_id: str) -> list:
        return self._request("GET", f"markets?lobby_id={lobby_id}&match_id={match_id}")

    # --- trading ---
    def submit_batch(self, lobby_id: str, preds: dict[str, int]) -> dict:
        """preds maps market_id -> integer probability (1..99)."""
        body = {"predictions": [
            {"market_id": mid, "lobby_id": lobby_id, "probability": int(p)}
            for mid, p in preds.items()
        ]}
        return self._request("POST", "predictions/batch", body)

    def patch(self, prediction_id: str, probability: int) -> dict:
        return self._request(
            "PATCH", f"predictions/{prediction_id}", {"probability": int(probability)}
        )

    def open_predictions(self, lobby_id: str) -> list:
        return self._request("GET", f"predictions?lobby_id={lobby_id}")

    def results(self, lobby_id: str) -> list:
        """Settled predictions with Brier scores (lower is better)."""
        return self._request("GET", f"results?lobby_id={lobby_id}")
