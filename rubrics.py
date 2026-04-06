"""
rubrics.py — Deterministic graders for Easy / Medium / Hard triage tasks.
P2 deliverable. All functions return a float in [0.0, 1.0].
Same input ALWAYS produces same output — zero randomness.
"""

from typing import Dict, List, Optional

# EASY GRADER
# Scores a single-patient triage action against ground truth.
# Reward breakdown: priority 0.40 | ward 0.30 | treatment 0.30
# Critical miss penalty: -0.50 (clamped so final score >= 0.0)

EASY_PRIORITY_WEIGHTS = {"immediate": 0.40, "urgent": 0.40, "non-urgent": 0.40}
EASY_CRITICAL_SYMPTOMS = {
    "chest pain", "sweating", "cyanosis", "shortness of breath",
    "confusion", "vomiting blood", "severe abdominal pain",
    "sudden severe headache", "facial droop", "slurred speech",
    "left arm numbness", "no pulse", "apneic"
}


def _is_critical_patient(symptoms: List[str]) -> bool:
    """Return True if patient has any critical-flag symptom."""
    lower = {s.lower() for s in symptoms}
    return bool(lower & EASY_CRITICAL_SYMPTOMS)


def grade_easy(action: dict, ground_truth: dict, patient_symptoms: List[str]) -> float:
    """
    Grade a single Easy-task action.

    Args:
        action:          Agent output dict with keys priority, ward, treatment.
        ground_truth:    Ground-truth dict with same keys.
        patient_symptoms: Patient symptom list (for critical-miss check).

    Returns:
        float in [0.0, 1.0]
    """
    score = 0.0
    penalty = 0.0

    # Priority — 0.40
    if action.get("priority") == ground_truth["priority"]:
        score += 0.40
    else:
        # Critical miss: patient needed immediate but agent said non-urgent
        gt_priority = ground_truth["priority"]
        agent_priority = action.get("priority", "")
        if gt_priority == "immediate" and agent_priority == "non-urgent":
            penalty += 0.50
        elif gt_priority == "immediate" and agent_priority != "immediate":
            penalty += 0.25  # urgent instead of immediate — partial penalty

    # Ward — 0.30
    if action.get("ward") == ground_truth["ward"]:
        score += 0.30

    # Treatment — 0.30 (exact match OR semantically close)
    if action.get("treatment") == ground_truth["treatment"]:
        score += 0.30
    else:
        # Partial credit: same category of care
        gt_t = ground_truth["treatment"]
        ag_t = action.get("treatment", "")
        if gt_t and ag_t:
            gt_root = gt_t.split("_")[0]
            ag_root = ag_t.split("_")[0]
            if gt_root == ag_root:
                score += 0.15  # same treatment family

    final = max(0.0, min(1.0, score - penalty))
    return round(final, 4)

# MEDIUM GRADER
# Scores multi-patient queue action: treat_order + allocation.
# Reward breakdown: survival 0.40 | resource efficiency 0.30 | wait time 0.20
# Penalty: -0.10 per missed critical patient in ICU allocation

def _score_treat_order(agent_order: List[str], gt_order: List[str]) -> float:
    """
    Partial-credit scoring for treat order.
    Full credit for correct order. Partial credit based on
    how many top-k positions match (Spearman-like, simplified).
    """
    if not agent_order or not gt_order:
        return 0.0

    n = len(gt_order)
    # Weight earlier positions more heavily
    weights = [1 / (i + 1) for i in range(n)]
    total_weight = sum(weights)
    earned = 0.0

    for i, pid in enumerate(gt_order):
        if i < len(agent_order) and agent_order[i] == pid:
            earned += weights[i]
        elif pid in agent_order:
            # Partial: patient is present but in wrong position
            earned += weights[i] * 0.3

    return earned / total_weight


def _score_allocation(agent_alloc: dict, gt_alloc: dict, critical_ids: List[str]) -> tuple:
    """
    Returns (allocation_score 0–1, penalty_count).
    penalty_count = number of critical patients not given ICU.
    """
    if not gt_alloc:
        return 1.0, 0

    correct = sum(1 for pid, ward in gt_alloc.items()
                  if agent_alloc.get(pid) == ward)
    alloc_score = correct / len(gt_alloc)

    # Count critical patients who needed ICU but didn't get it
    penalties = sum(
        1 for pid in critical_ids
        if gt_alloc.get(pid) == "ICU" and agent_alloc.get(pid) != "ICU"
    )
    return alloc_score, penalties


def grade_medium(action: dict, ground_truth: dict,
                 patient_queue: list, step: int = 0) -> float:
    """
    Grade a Medium-task action.

    Args:
        action:        Agent output with treat_order and allocate.
        ground_truth:  Ground-truth with treat_order and allocate.
        patient_queue: List of Patient objects in the scenario.
        step:          Current env step (unused in base scoring, hook for future).

    Returns:
        float in [0.0, 1.0]
    """
    gt_order = ground_truth.get("treat_order", [])
    gt_alloc = ground_truth.get("allocate", {})

    agent_order = action.get("treat_order", [])
    agent_alloc = action.get("allocate", {})

    critical_ids = [p.id for p in patient_queue if p.severity == "critical"]

    # Survival rate proxy = treat order quality (0.40)
    order_score = _score_treat_order(agent_order, gt_order)
    survival_component = order_score * 0.40

    # Resource efficiency = allocation accuracy (0.30)
    alloc_score, missed_critical = _score_allocation(agent_alloc, gt_alloc, critical_ids)
    resource_component = alloc_score * 0.30

    # Wait time score (0.20): bonus if top-2 treated are both critical
    top2_agent = agent_order[:2] if len(agent_order) >= 2 else agent_order
    top2_critical = sum(1 for pid in top2_agent if pid in critical_ids)
    wait_component = (top2_critical / 2) * 0.20

    # Deterioration penalty (-0.10 per missed critical)
    penalty = missed_critical * 0.10

    raw = survival_component + resource_component + wait_component - penalty
    return round(max(0.0, min(1.0, raw)), 4)

# HARD GRADER
# Scores START triage tag for one patient in a mass-casualty scenario.
# Reward breakdown: tagging accuracy 0.35 | lives saved 0.35 | resource optimality 0.20
# Penalty: -0.10 per unnecessary deferral (if deferral list provided)
# Tags in severity order for partial-credit distance scoring
TAG_ORDER = {"black": 0, "red": 1, "yellow": 2, "green": 3}
TAG_DISTANCE_CREDIT = {0: 1.0, 1: 0.5, 2: 0.0}  # distance → partial credit

# Which tags correspond to "life-saving" interventions
LIFE_SAVING_TAGS = {"red"}

# Resource cost per tag (lower = more efficient under scarcity)
RESOURCE_COST = {"red": 1.0, "yellow": 0.5, "green": 0.1, "black": 0.0}


def _tag_distance_score(agent_tag: str, gt_tag: str) -> float:
    """Partial credit based on distance between tags in severity order."""
    if agent_tag not in TAG_ORDER or gt_tag not in TAG_ORDER:
        return 0.0
    distance = abs(TAG_ORDER[agent_tag] - TAG_ORDER[gt_tag])
    return TAG_DISTANCE_CREDIT.get(distance, 0.0)

def grade_hard_single(patient_id: str, action: dict, ground_truth: dict) -> float:
    """
    Grade a single patient's tag in the Hard task.
    Args:
        patient_id:    e.g. "H002"
        action:        Agent output with 'tag' key (and optionally 'defer')
        ground_truth:  HARD_GROUND_TRUTH dict for this patient_id

    Returns:
        float in [0.0, 1.0]
    """
    gt = ground_truth.get(patient_id)
    if gt is None:
        return 0.0

    gt_tag = gt["tag"]
    agent_tag = action.get("tag", "")

    # Tagging accuracy (0.35) — with partial credit for adjacent tags
    tag_score = _tag_distance_score(agent_tag, gt_tag)
    tagging_component = tag_score * 0.35

    # Lives saved proxy (0.35):
    # Full credit if agent correctly tags a "red" patient as red (saves life)
    # Partial credit for yellow on a red (delayed = risk)
    # Zero credit (or negative) for black on a red (abandons saveable patient)
    lives_component = 0.0
    if gt_tag == "red":
        if agent_tag == "red":
            lives_component = 0.35
        elif agent_tag == "yellow":
            lives_component = 0.15   # delayed but not abandoned
        elif agent_tag == "black":
            lives_component = 0.0    # wrongly abandoned
        else:
            lives_component = 0.10
    elif gt_tag == "black":
        if agent_tag == "black":
            lives_component = 0.35   # correctly didn't waste resources
        elif agent_tag == "red":
            lives_component = 0.10   # over-triaged, wasted resources but no harm to this patient
        else:
            lives_component = 0.20
    else:
        # yellow / green — scale by tag accuracy
        lives_component = tag_score * 0.35

    # Resource optimality (0.20):
    # Under blood_supply=low, over-triaging is costly. Reward correct frugality.
    gt_cost = RESOURCE_COST.get(gt_tag, 0.5)
    agent_cost = RESOURCE_COST.get(agent_tag, 0.5)
    cost_diff = abs(gt_cost - agent_cost)
    resource_component = max(0.0, 0.20 - cost_diff * 0.20)

    # Unnecessary deferral penalty (-0.10 each)
    defer_list = action.get("defer", [])
    penalty = 0.0
    if patient_id in defer_list and gt_tag in ("red",):
        penalty = 0.10  # deferred a patient who needed immediate care

    raw = tagging_component + lives_component + resource_component - penalty
    return round(max(0.0, min(1.0, raw)), 4)

def grade_hard_batch(actions: Dict[str, dict], ground_truth: dict) -> float:
    """
    Grade all 15 patients in one Hard episode.
    Args:
        actions:       {patient_id: action_dict} for all patients
        ground_truth:  HARD_GROUND_TRUTH dict
    Returns:
        float in [0.0, 1.0] — mean score across all patients
    """
    if not actions:
        return 0.0
    scores = [
        grade_hard_single(pid, act, ground_truth)
        for pid, act in actions.items()
    ]
    return round(sum(scores) / len(scores), 4)

