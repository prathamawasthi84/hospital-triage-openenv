import os
import json
import textwrap
import requests
from typing import List, Optional
from openai import OpenAI

# ──────────────────────────────────────────
# ENV VARS — exactly as per sample
# ──────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL      = os.getenv("ENV_URL") or "http://localhost:8004"
BENCHMARK    = "hospital-triage-openenv"
MAX_STEPS    = 20
TEMPERATURE  = 0.3
MAX_TOKENS   = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ──────────────────────────────────────────
# LOG FUNCTIONS — must match format exactly
# ──────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}",
          flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True,
    )

# ──────────────────────────────────────────
# ENV INTERACTION — calls our FastAPI server
# ──────────────────────────────────────────

def env_reset(task_level: str) -> dict:
    """Call /reset on our environment server"""
    response = requests.post(
        f"{ENV_URL}/reset",
        json={"task_level": task_level},
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def env_step(action: dict) -> dict:
    """Call /step on our environment server"""
    response = requests.post(
        f"{ENV_URL}/step",
        json=action,
        timeout=30
    )
    response.raise_for_status()
    return response.json()

def env_state() -> dict:
    """Call /state on our environment server"""
    response = requests.get(
        f"{ENV_URL}/state",
        timeout=30
    )
    response.raise_for_status()
    return response.json()

# ──────────────────────────────────────────
# SYSTEM PROMPT — tells LLM what to do
# ──────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert emergency room triage doctor.
    
    You will receive information about a patient including:
    - Age, blood pressure, heart rate, oxygen saturation
    - Symptoms they are experiencing
    - Available hospital resources (ICU beds, doctors etc)
    - Current queue of other patients waiting
    
    You must respond with a JSON object containing:
    {
        "patient_id": "<exact patient id from observation>",
        "priority": "<immediate|urgent|non_urgent|deceased>",
        "ward": "<ICU|emergency|general|waiting>",
        "treatment": "<cardiac_protocol|trauma_protocol|respiratory_protocol|basic_care|observe_only>",
        "reasoning": "<brief explanation mentioning patient symptoms>"
    }
    
    Priority guide:
    - immediate: life threatening (chest pain, severe bleeding, no breathing)
    - urgent: serious but stable (broken bones, moderate pain)
    - non_urgent: minor issues (mild headache, small cuts)
    - deceased: no pulse, fixed pupils, no intervention possible
    
    Ward guide:
    - ICU: critical patients needing intensive monitoring
    - emergency: urgent patients needing prompt treatment
    - general: stable patients needing standard care
    - waiting: minor patients who can wait
    
    Treatment guide:
    - cardiac_protocol: chest pain, heart attack, arrhythmia
    - trauma_protocol: injuries, bleeding, fractures
    - respiratory_protocol: breathing problems, low oxygen
    - basic_care: general treatment, monitoring
    - observe_only: deceased or very minor cases
    
    IMPORTANT: Always use the EXACT patient_id from the observation.
    Always include ALL fields: patient_id, priority, ward, treatment, reasoning.
    Always respond with valid JSON only. No extra text.
""").strip()

# ──────────────────────────────────────────
# LLM CALL — get triage decision from model
# ──────────────────────────────────────────

def get_triage_decision(
    client: OpenAI,
    observation: dict,
    step: int,
    history: List[str]
) -> dict:
    """Ask LLM to make triage decision based on observation"""

    patient = observation.get("current_patient", {})
    resources = observation.get("resources", {})
    queue_size = len(observation.get("queue", []))

    user_prompt = textwrap.dedent(f"""
        Step {step} — Triage Decision Required

        CURRENT PATIENT:
        - ID: {patient.get('id', 'unknown')}
        - Age: {patient.get('age', 'unknown')}
        - Blood Pressure: {patient.get('blood_pressure', 'unknown')}
        - Heart Rate: {patient.get('heart_rate', 'unknown')} bpm
        - Oxygen Saturation: {patient.get('oxygen_saturation', 'unknown')}%
        - Symptoms: {', '.join(patient.get('symptoms', []))}
        - Wait Time: {patient.get('wait_time', 0)} minutes
        - Deteriorating: {patient.get('deteriorating', False)}

        RESOURCES AVAILABLE:
        - ICU Beds: {resources.get('ICU_beds', 0)}
        - General Beds: {resources.get('general_beds', 0)}
        - Doctors: {resources.get('doctors', 0)}
        - Blood Supply: {resources.get('blood_supply', 'normal')}

        OTHER PATIENTS WAITING: {queue_size}
        TASK LEVEL: {observation.get('task_level', 'easy')}
        CONTEXT: {observation.get('message', '')}

        Recent history:
        {chr(10).join(history[-3:]) if history else 'None'}

        Respond with JSON triage decision. Use EXACT patient ID: {patient.get('id', 'unknown')}
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        decision = json.loads(text)

        # ✅ FIX: ensure all required fields exist with safe defaults
        decision["patient_id"] = patient.get("id", decision.get("patient_id", "unknown"))
        decision.setdefault("priority",  "urgent")
        decision.setdefault("ward",      "emergency")
        decision.setdefault("treatment", "basic_care")
        decision.setdefault("reasoning", "LLM decision")

        return decision

    except json.JSONDecodeError:
        return {
            "patient_id": patient.get("id", "unknown"),
            "priority":   "urgent",
            "ward":       "emergency",
            "treatment":  "basic_care",
            "reasoning":  "Fallback decision due to parsing error"
        }
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return {
            "patient_id": patient.get("id", "unknown"),
            "priority":   "urgent",
            "ward":       "emergency",
            "treatment":  "basic_care",
            "reasoning":  "Fallback decision due to API error"
        }

# ──────────────────────────────────────────
# RUN ONE TASK EPISODE
# ──────────────────────────────────────────

def run_task(client: OpenAI, task_level: str) -> float:
    """
    Run one full episode for a given task level.
    Returns final score 0.0–1.0
    """
    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(
        task=task_level,
        env=BENCHMARK,
        model=MODEL_NAME
    )

    try:
        # reset environment
        reset_result = env_reset(task_level)
        observation  = reset_result.get("observation", {})

        # ✅ FIX: safety check — verify reset worked
        if not observation:
            print(f"[DEBUG] Reset returned empty observation for {task_level}", flush=True)
            return 0.0

        for step in range(1, MAX_STEPS + 1):

            # check if episode already done
            if not observation.get("current_patient"):
                break

            # get triage decision from LLM
            decision = get_triage_decision(
                client, observation, step, history
            )

            # format action string for log
            action_str = (
                f"triage(id={decision.get('patient_id','')},"
                f"priority={decision.get('priority','')},"
                f"ward={decision.get('ward','')})"
            )

            error = None
            reward = 0.0
            done = False

            try:
                step_result = env_step(decision)
                reward      = step_result.get(
                    "reward", {}
                ).get("total", 0.0)
                done        = step_result.get("done", False)
                observation = step_result.get("observation", {})

                history.append(
                    f"Step {step}: {action_str} "
                    f"-> reward {reward:+.2f}"
                )

            except Exception as e:
                error = str(e)[:100]
                done  = True

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error
            )

            if done:
                break

        # calculate final score
        if rewards:
            score = sum(rewards) / len(rewards)
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards
        )

    return score

# ──────────────────────────────────────────
# MAIN — run all 3 tasks
# ──────────────────────────────────────────

def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    print("[DEBUG] Starting Hospital Triage baseline inference",
          flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Env URL: {ENV_URL}", flush=True)

    task_levels = ["easy", "medium", "hard"]
    all_scores  = {}

    for task_level in task_levels:
        print(f"\n[DEBUG] Running task: {task_level}", flush=True)
        score = run_task(client, task_level)
        all_scores[task_level] = score
        print(f"[DEBUG] {task_level} score: {score:.3f}",
              flush=True)

    print("\n[DEBUG] ── FINAL SCORES ──", flush=True)
    for level, score in all_scores.items():
        print(f"[DEBUG] {level}: {score:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[DEBUG] average: {avg:.3f}", flush=True)

if __name__ == "__main__":
    main()