from env import TriageEnv
from models import Action

env = TriageEnv()

# ── TEST EASY ──────────────────────────
print("Testing EASY grader...")
obs = env.reset("easy")
patient = obs.current_patient
truth = env._get_ground_truth(patient.id)

# Test 1 — perfect action = 1.0
perfect = Action(
    patient_id=patient.id,
    priority=truth["correct_priority"],
    ward=truth["correct_ward"],
    treatment=truth["correct_treatment"]
)
result = env.step(perfect)
assert 0.0 <= result.reward.total <= 1.0, "Score out of range!"
assert result.reward.total == 1.0, f"Perfect action should be 1.0, got {result.reward.total}"
print(f"  Perfect action: {result.reward.total} ✅")

# Test 2 — wrong action = low score
obs = env.reset("easy")
patient = obs.current_patient
wrong = Action(
    patient_id=patient.id,
    priority="non_urgent",
    ward="waiting",
    treatment="observe_only"
)
result = env.step(wrong)
assert 0.0 <= result.reward.total <= 1.0, "Score out of range!"
print(f"  Wrong action: {result.reward.total} ✅")

# ── TEST MEDIUM ────────────────────────
print("\nTesting MEDIUM grader...")
obs = env.reset("medium")
patient = obs.current_patient
truth = env._get_ground_truth(patient.id)

perfect = Action(
    patient_id=patient.id,
    priority=truth["correct_priority"],
    ward=truth["correct_ward"],
    treatment=truth.get("correct_treatment", "basic_care")
)
result = env.step(perfect)
assert 0.0 <= result.reward.total <= 1.0, "Score out of range!"
print(f"  Perfect action: {result.reward.total} ✅")

# ── TEST HARD ──────────────────────────
print("\nTesting HARD grader...")
obs = env.reset("hard")
patient = obs.current_patient
truth = env._get_ground_truth(patient.id)

# map correct_tag to priority
tag_to_priority = {
    "red":    "immediate",
    "yellow": "urgent",
    "green":  "non_urgent",
    "black":  "deceased"
}
priority = tag_to_priority.get(
    truth.get("correct_tag", "yellow"), "urgent"
)
perfect = Action(
    patient_id=patient.id,
    priority=priority,
    ward=truth.get("correct_ward", "emergency"),
    treatment="basic_care",
    reasoning=" ".join(patient.symptoms)
)
result = env.step(perfect)
assert 0.0 <= result.reward.total <= 1.0, "Score out of range!"
print(f"  Perfect action: {result.reward.total} ✅")

# ── DETERMINISM TEST ───────────────────
print("\nTesting determinism...")
obs = env.reset("easy")
patient = obs.current_patient
truth = env._get_ground_truth(patient.id)
action = Action(
    patient_id=patient.id,
    priority=truth["correct_priority"],
    ward=truth["correct_ward"],
    treatment=truth["correct_treatment"]
)
r1 = env.step(action)

obs = env.reset("easy")
# find same patient
while obs.current_patient.id != patient.id:
    obs = env.reset("easy")
r2 = env.step(action)
assert r1.reward.total == r2.reward.total, "Not deterministic!"
print(f"  Score 1: {r1.reward.total} | Score 2: {r2.reward.total} ✅")

print("\n✅ ALL GRADER TESTS PASSED!")
print("✅ All scores in 0.0–1.0 range")
print("✅ Graders are deterministic")