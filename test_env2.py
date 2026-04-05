from env import TriageEnv
from models import Action

env = TriageEnv()
obs = env.reset("easy")
patient = obs.current_patient
print("Reset OK:", patient.id)
print("Symptoms:", patient.symptoms)

# get ground truth from task data
truth = env._get_ground_truth(patient.id)

# use correct action
action = Action(
    patient_id=patient.id,
    priority=truth["correct_priority"],
    ward=truth["correct_ward"],
    treatment=truth["correct_treatment"]
)
result = env.step(action)
print("Reward:", result.reward.total)
print("Feedback:", result.reward.feedback)
print("Done:", result.done)