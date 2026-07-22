"""Minimal SportsPredict REST client for the Probability Cup.

The API key is read from the SPORTSPREDICT_KEY environment variable (never
hardcode it). Create a bot key in the SportsPredict app (Profile -> My Bots) and:

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
    """Thin REST client for the SportsPredict Probability Cup API.

    Wraps the handful of endpoints needed to discover events/lobbies/matches/
    markets and to submit, patch, and read back predictions. Authentication uses
    a bot API key sent as a Bearer token.
    """

    def __init__(self, key: str | None = None, base_url: str = BASE_URL):
        """Initialise the client, resolving the API key from arg or environment.

        Args:
            key: API key to use; falls back to the SPORTSPREDICT_KEY environment
                variable when None.
            base_url: Base URL of the REST API.

        Raises:
            RuntimeError: If no key is provided and none is found in the
                environment.
        """
        self.key = key or os.environ.get("SPORTSPREDICT_KEY")
        if not self.key:
            raise RuntimeError(
                "No API key. Set SPORTSPREDICT_KEY=sp_live_... in your environment."
            )
        self.base_url = base_url

    def _request(self, method: str, path: str, body: dict | None = None) -> dict | list:
        """Send an authenticated HTTP request and return the decoded JSON body.

        Args:
            method: HTTP method (e.g. "GET", "POST", "PATCH").
            path: Path appended to the base URL (no leading slash).
            body: Optional JSON-serialisable request body; sent only when given.

        Returns:
            dict | list: The parsed JSON response.

        Raises:
            RuntimeError: If the server returns an HTTP error status.
        """
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
        """List available events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            list: The event records.
        """
        return self._request("GET", f"events?limit={limit}")

    def lobbies(self, event_id: str) -> list:
        """List the lobbies belonging to an event.

        Args:
            event_id: The event whose lobbies to list.

        Returns:
            list: The lobby records.
        """
        return self._request("GET", f"lobbies?event_id={event_id}")

    def join_lobby(self, lobby_id: str) -> dict:
        """Join a lobby as the authenticated bot.

        Args:
            lobby_id: The lobby to join.

        Returns:
            dict: The join confirmation record.
        """
        return self._request("POST", f"lobbies/{lobby_id}/join")

    def matches(self, event_id: str, lobby_id: str) -> list:
        """List the matches available in a lobby.

        Args:
            event_id: The event the lobby belongs to.
            lobby_id: The lobby whose matches to list.

        Returns:
            list: The match records.
        """
        return self._request("GET", f"matches?event_id={event_id}&lobby_id={lobby_id}")

    def markets(self, lobby_id: str, match_id: str) -> list:
        """List the prediction markets for a match within a lobby.

        Args:
            lobby_id: The lobby the match belongs to.
            match_id: The match whose markets to list.

        Returns:
            list: The market records.
        """
        return self._request("GET", f"markets?lobby_id={lobby_id}&match_id={match_id}")

    # --- trading ---
    def submit_batch(self, lobby_id: str, preds: dict[str, int]) -> dict:
        """Submit a batch of predictions for a lobby.

        Args:
            lobby_id: The lobby to submit predictions into.
            preds: Mapping of market_id to integer probability (1..99).

        Returns:
            dict: The batch-submission response.
        """
        body = {"predictions": [
            {"market_id": mid, "lobby_id": lobby_id, "probability": int(p)}
            for mid, p in preds.items()
        ]}
        return self._request("POST", "predictions/batch", body)

    def patch(self, prediction_id: str, probability: int) -> dict:
        """Update the probability of a single existing prediction.

        Args:
            prediction_id: The prediction to update.
            probability: The new integer probability (1..99).

        Returns:
            dict: The updated prediction record.
        """
        return self._request(
            "PATCH", f"predictions/{prediction_id}", {"probability": int(probability)}
        )

    def open_predictions(self, lobby_id: str) -> list:
        """List the authenticated bot's open (unsettled) predictions in a lobby.

        Args:
            lobby_id: The lobby whose open predictions to list.

        Returns:
            list: The open prediction records.
        """
        return self._request("GET", f"predictions?lobby_id={lobby_id}")

    def results(self, lobby_id: str) -> list:
        """List settled predictions with their Brier scores (lower is better).

        Args:
            lobby_id: The lobby whose settled results to fetch.

        Returns:
            list: The settled prediction records including Brier scores.
        """
        return self._request("GET", f"results?lobby_id={lobby_id}")
