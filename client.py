"""
client.py — HTTP client for the Hospital Triage OpenEnv.

Allows users to interact with the environment programmatically
after installing the package from the HF Space:

    pip install git+https://huggingface.co/spaces/pratham45/hospital-triage-openenv

Usage:
    from client import TriageEnvClient

    client = TriageEnvClient("http://localhost:8004")
    obs = client.reset("easy")
    result = client.step("P001", "immediate", "ICU", "cardiac_protocol")
    state = client.state()
"""

from __future__ import annotations

import requests
from typing import Optional, Dict, Any


class TriageEnvClient:
    """
    HTTP client for the Hospital Triage OpenEnv server.

    Args:
        base_url: Server base URL, e.g. "http://localhost:8004"
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(self, base_url: str = "http://localhost:8004", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make a GET request to the environment server."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to environment at {url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out: GET {url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text[:200]}")

    def _post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a POST request to the environment server."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.post(url, json=data or {}, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to environment at {url}")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out: POST {url}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text[:200]}")

    def health(self) -> Dict[str, Any]:
        """Check if the server is healthy."""
        return self._get("/health")

    def reset(self, task_level: str = "easy") -> Dict[str, Any]:
        """
        Reset the environment and start a new episode.

        Args:
            task_level: One of "easy", "medium", "hard"

        Returns:
            Dict with keys: status, task_level, observation
        """
        return self._post("/reset", {"task_level": task_level})

    def step(
        self,
        patient_id: str,
        priority: str,
        ward: str,
        treatment: str,
        reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a triage action for the current patient.

        Args:
            patient_id: ID of the patient being triaged (e.g. "P001")
            priority:   One of "immediate", "urgent", "non_urgent", "deceased"
            ward:       One of "ICU", "emergency", "general", "waiting"
            treatment:  One of "cardiac_protocol", "trauma_protocol",
                        "respiratory_protocol", "basic_care", "observe_only"
            reasoning:  Optional string explaining the agent's decision

        Returns:
            Dict with keys: status, observation, reward, done, info
        """
        payload: Dict[str, Any] = {
            "patient_id": patient_id,
            "priority": priority,
            "ward": ward,
            "treatment": treatment,
        }
        if reasoning is not None:
            payload["reasoning"] = reasoning
        return self._post("/step", payload)

    def state(self) -> Dict[str, Any]:
        """
        Get the full current environment state.

        Returns:
            Dict with keys: status, state
        """
        return self._get("/state")

    def info(self) -> Dict[str, Any]:
        """Get environment info from root endpoint."""
        return self._get("/")


# ── Convenience alias ─────────────────────────────────────────────────────

HospitalTriageClient = TriageEnvClient


# ── Quick demo when run directly ──────────────────────────────────────────

if __name__ == "__main__":
    import json

    client = TriageEnvClient("http://localhost:8004")

    print("Checking health...")
    print(json.dumps(client.health(), indent=2))

    print("\nResetting (easy)...")
    obs = client.reset("easy")
    print(json.dumps(obs, indent=2))

    patient = obs.get("observation", {}).get("current_patient", {})
    pid = patient.get("id", "P001")

    print(f"\nStepping with patient {pid}...")
    result = client.step(
        patient_id=pid,
        priority="immediate",
        ward="ICU",
        treatment="cardiac_protocol",
        reasoning="Demo action",
    )
    print(json.dumps(result, indent=2))