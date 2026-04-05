from models import (
    Patient, Observation, Action, Reward,
    EnvState, StepResponse, Resources,
    Priority, Ward, Treatment, Severity,
    TaskLevel, RewardBreakdown
)
from tasks import get_task
from typing import List, Optional, Dict, Any
import copy

class TriageEnv:

    def __init__(self):
        self.queue: List[Patient] = []
        self.resources: Optional[Resources] = None
        self.step_count: int = 0
        self.max_steps: int = 20
        self.total_reward: float = 0.0
        self.task_level: TaskLevel = TaskLevel.EASY
        self.task_data: Dict = {}
        self.done: bool = False
        self.patients_treated: int = 0
        self.patients_deteriorated: int = 0
        self.patients_deceased: int = 0
        self._patient_index: int = 0  # for easy task cycling

    # ──────────────────────────────────────
    # RESET — start a fresh episode
    # ──────────────────────────────────────

    def reset(self, task_level: str = "easy") -> Observation:
        """
        Resets environment and returns first observation.
        Called at start of every episode.
        """
        # reset all internal state
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.patients_treated = 0
        self.patients_deteriorated = 0
        self.patients_deceased = 0
        self._patient_index = 0
        self.task_level = TaskLevel(task_level)

        # load task scenario from tasks.py
        self.task_data = get_task(self.task_level)

        # build patient queue based on task level
        self._build_queue()

        # set resources based on task level
        self._set_resources()

        # set max steps based on difficulty
        self.max_steps = {
            TaskLevel.EASY:   10,
            TaskLevel.MEDIUM: 15,
            TaskLevel.HARD:   20,
        }[self.task_level]

        return self._get_observation()

    def _build_queue(self):
        """Build patient queue from task data"""
        self.queue = []
        if self.task_level == TaskLevel.EASY:
            for p_data in self.task_data["patients"]:
                self.queue.append(self._dict_to_patient(p_data))
        elif self.task_level in [TaskLevel.MEDIUM, TaskLevel.HARD]:
            for p_data in self.task_data["patients"]:
                self.queue.append(self._dict_to_patient(p_data))

    def _set_resources(self):
        """Set resources from task data"""
        if "resources" in self.task_data:
            r = self.task_data["resources"]
        else:
            # default resources for easy task
            r = {
                "ICU_beds": 5,
                "general_beds": 10,
                "doctors": 5,
                "OR_rooms": 2,
                "blood_supply": "normal"
            }
        self.resources = Resources(**r)

    def _dict_to_patient(self, p_data: Dict) -> Patient:
        """Convert raw dict from tasks.py to Patient model"""
        return Patient(
            id=p_data["id"],
            age=p_data["age"],
            blood_pressure=p_data["blood_pressure"],
            heart_rate=p_data["heart_rate"],
            oxygen_saturation=p_data["oxygen_saturation"],
            symptoms=p_data["symptoms"],
            severity=p_data.get("severity"),
            deteriorating=p_data.get("deteriorating", False),
            wait_time=0,
        )

    # ──────────────────────────────────────
    # STEP — process one agent action
    # ──────────────────────────────────────

    def step(self, action: Action) -> StepResponse:
        """
        Process agent action and return next observation,
        reward, done flag and info dict.
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        # find patient agent is acting on
        patient = self._find_patient(action.patient_id)

        if patient is None:
            # penalise agent for invalid patient ID
            reward = Reward(
                total=0.0,
                breakdown=RewardBreakdown(),
                feedback="Invalid patient ID provided."
            )
        else:
            # grade action using rubrics
            reward = self._grade_action(action, patient)

            # update patient status
            patient.assigned_priority = action.priority
            patient.assigned_ward = action.ward

            # consume resources
            self._consume_resources(action.ward)

            # mark patient as treated
            self.patients_treated += 1

            # remove treated patient from queue
            # for easy task — move to next patient
            if self.task_level == TaskLevel.EASY:
                self.queue = [p for p in self.queue
                            if p.id != action.patient_id]

        # increment step
        self.step_count += 1
        self.total_reward += reward.total

        # apply deterioration every 3 steps
        self._apply_deterioration()

        # increase wait time for remaining patients
        self._update_wait_times()

        # check if episode is done
        self.done = self._check_done()

        return StepResponse(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info={
                "step": self.step_count,
                "total_reward": round(self.total_reward, 3),
                "patients_remaining": len(self.queue),
                "patients_treated": self.patients_treated,
                "patients_deteriorated": self.patients_deteriorated,
            }
        )

    # ──────────────────────────────────────
    # STATE — return full env snapshot
    # ──────────────────────────────────────

    def state(self) -> EnvState:
        """
        Returns complete environment state.
        Can be called anytime during episode.
        """
        return EnvState(
            queue=self.queue,
            resources=self.resources,
            step=self.step_count,
            max_steps=self.max_steps,
            total_reward=round(self.total_reward, 3),
            task_level=self.task_level,
            done=self.done,
            patients_treated=self.patients_treated,
            patients_deteriorated=self.patients_deteriorated,
            patients_deceased=self.patients_deceased,
        )

    # ──────────────────────────────────────
    # GRADING — score the agent's action
    # ──────────────────────────────────────

    def _grade_action(self, action: Action,
                    patient: Patient) -> Reward:
        """Grade action against ground truth from tasks.py"""

        if self.task_level == TaskLevel.EASY:
            return self._grade_easy(action, patient)
        elif self.task_level == TaskLevel.MEDIUM:
            return self._grade_medium(action, patient)
        elif self.task_level == TaskLevel.HARD:
            return self._grade_hard(action, patient)

    def _grade_easy(self, action: Action,
                    patient: Patient) -> Reward:
        """Easy grader — priority + ward + treatment"""

        # find ground truth from task data
        ground_truth = self._get_ground_truth(patient.id)

        breakdown = RewardBreakdown()
        feedback_parts = []

        # priority score (40%)
        if action.priority == ground_truth.get("correct_priority"):
            breakdown.priority_score = 0.40
            feedback_parts.append("Priority correct ✓")
        else:
            breakdown.priority_score = 0.0
            feedback_parts.append(
                f"Priority wrong — expected "
                f"{ground_truth.get('correct_priority')}"
            )
            # heavy penalty for missing critical patient
            if ground_truth.get("severity") == Severity.CRITICAL:
                breakdown.deterioration_penalty = -0.30
                feedback_parts.append("Critical patient missed! -0.30")

        # ward score (30%)
        if action.ward == ground_truth.get("correct_ward"):
            breakdown.ward_score = 0.30
            feedback_parts.append("Ward correct ✓")
        else:
            breakdown.ward_score = 0.0
            feedback_parts.append(
                f"Ward wrong — expected "
                f"{ground_truth.get('correct_ward')}"
            )

        # treatment score (30%)
        if action.treatment == ground_truth.get("correct_treatment"):
            breakdown.treatment_score = 0.30
            feedback_parts.append("Treatment correct ✓")
        else:
            breakdown.treatment_score = 0.0
            feedback_parts.append(
                f"Treatment wrong — expected "
                f"{ground_truth.get('correct_treatment')}"
            )

        # calculate total — clamp to 0.0–1.0
        total = max(0.0, min(1.0,
            breakdown.priority_score +
            breakdown.ward_score +
            breakdown.treatment_score +
            breakdown.deterioration_penalty
        ))

        return Reward(
            total=round(total, 3),
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts)
        )

    def _grade_medium(self, action: Action,
                    patient: Patient) -> Reward:
        """Medium grader — survival + resources + wait time"""

        ground_truth = self._get_ground_truth(patient.id)
        breakdown = RewardBreakdown()
        feedback_parts = []

        # priority correct? (40%)
        if action.priority == ground_truth.get("correct_priority"):
            breakdown.priority_score = 0.40
            feedback_parts.append("Priority correct ✓")
        else:
            breakdown.priority_score = 0.0
            feedback_parts.append("Priority wrong ✗")

        # resource efficiency — did agent use ICU wisely? (30%)
        if action.ward == ground_truth.get("correct_ward"):
            breakdown.ward_score = 0.30
            feedback_parts.append("Ward correct ✓")
        else:
            # penalise wasting ICU on non-critical
            if (action.ward == Ward.ICU and
                ground_truth.get("severity") != Severity.CRITICAL):
                breakdown.ward_score = -0.10
                feedback_parts.append("Wasted ICU bed! -0.10")
            else:
                breakdown.ward_score = 0.0
                feedback_parts.append("Ward wrong ✗")

        # wait time score (20%)
        if patient.wait_time < 5:
            breakdown.treatment_score = 0.20
            feedback_parts.append("Treated promptly ✓")
        elif patient.wait_time < 10:
            breakdown.treatment_score = 0.10
            feedback_parts.append("Slight delay")
        else:
            breakdown.treatment_score = 0.0
            breakdown.wait_time_penalty = -0.10
            feedback_parts.append("Long wait time ✗")

        # deterioration penalty
        if patient.deteriorating:
            breakdown.deterioration_penalty -= 0.10
            feedback_parts.append("Patient was deteriorating! -0.10")

        total = max(0.0, min(1.0,
            breakdown.priority_score +
            breakdown.ward_score +
            breakdown.treatment_score +
            breakdown.wait_time_penalty +
            breakdown.deterioration_penalty
        ))

        return Reward(
            total=round(total, 3),
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts)
        )

    def _grade_hard(self, action: Action,
                    patient: Patient) -> Reward:
        """Hard grader — START triage tagging accuracy"""

        ground_truth = self._get_ground_truth(patient.id)
        breakdown = RewardBreakdown()
        feedback_parts = []

        # tagging accuracy (35%)
        correct_tag = ground_truth.get("correct_tag")
        action_tag = action.priority

        tag_map = {
            Priority.IMMEDIATE:  "red",
            Priority.URGENT:     "yellow",
            Priority.NON_URGENT: "green",
            Priority.DECEASED:   "black",
        }
        agent_tag = tag_map.get(action_tag, "green")

        if agent_tag == correct_tag:
            breakdown.priority_score = 0.35
            feedback_parts.append(f"Tag correct: {correct_tag} ✓")
        else:
            breakdown.priority_score = 0.0
            feedback_parts.append(
                f"Tag wrong — expected {correct_tag}, "
                f"got {agent_tag} ✗"
            )
            # critical mistake — tagged black when saveable
            if correct_tag == "red" and agent_tag == "black":
                breakdown.deterioration_penalty = -0.20
                feedback_parts.append("Fatal error: gave up on saveable patient! -0.20")

        # reasoning quality (40%) — keyword matching
        reasoning = (action.reasoning or "").lower()
        key_symptoms = [s.lower() for s in patient.symptoms]
        keywords_used = sum(
            1 for kw in key_symptoms if kw in reasoning
        )
        reasoning_score = min(0.40,
            (keywords_used / max(len(key_symptoms), 1)) * 0.40
        )
        breakdown.ward_score = round(reasoning_score, 3)
        feedback_parts.append(
            f"Reasoning score: {round(reasoning_score, 3)}"
        )

        # resource optimality (20%)
        if action.ward == ground_truth.get("correct_ward",Ward.WAITING):
            breakdown.treatment_score = 0.20
            feedback_parts.append("Resource use optimal ✓")
        else:
            breakdown.treatment_score = 0.0
            feedback_parts.append("Resource use suboptimal ✗")

        total = max(0.0, min(1.0,
            breakdown.priority_score +
            breakdown.ward_score +
            breakdown.treatment_score +
            breakdown.deterioration_penalty
        ))

        return Reward(
            total=round(total, 3),
            breakdown=breakdown,
            feedback=" | ".join(feedback_parts)
        )

    # ──────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────

    def _get_observation(self) -> Observation:
        """Build observation from current state"""
        # current patient = first in queue
        current = self.queue[0] if self.queue else None

        # hide severity from agent
        if current:
            visible = current.copy()
            visible.severity = None
        else:
            visible = None

        # hide severity from queue too
        visible_queue = []
        for p in self.queue[1:]:
            vp = p.copy()
            vp.severity = None
            visible_queue.append(vp)

        return Observation(
            current_patient=visible,
            queue=visible_queue,
            resources=self.resources,
            step=self.step_count,
            max_steps=self.max_steps,
            task_level=self.task_level,
            message=self._get_context_message(),
        )

    def _get_context_message(self) -> str:
        """Provide context to agent"""
        if self.task_level == TaskLevel.EASY:
            return "Triage this patient. Assign priority, ward and treatment."
        elif self.task_level == TaskLevel.MEDIUM:
            return (f"Manage the ER queue. "
                    f"{len(self.queue)} patients waiting. "
                    f"ICU beds: {self.resources.ICU_beds}.")
        elif self.task_level == TaskLevel.HARD:
            deteriorating = sum(
                1 for p in self.queue if p.deteriorating
            )
            return (f"MASS CASUALTY. {len(self.queue)} casualties. "
                    f"{deteriorating} deteriorating. "
                    f"Resources critically low.")
        return ""

    def _find_patient(self,
                    patient_id: str) -> Optional[Patient]:
        """Find patient by ID in queue"""
        for p in self.queue:
            if p.id == patient_id:
                return p
        return None

    def _get_ground_truth(self, patient_id: str) -> Dict:
        """Get correct answer for a patient from task data"""
        patients = self.task_data.get("patients", [])
        for p in patients:
            if p["id"] == patient_id:
                return p
        return {}

    def _consume_resources(self, ward: Ward):
        """Reduce resource count when patient assigned"""
        if ward == Ward.ICU:
            self.resources.ICU_beds = max(
                0, self.resources.ICU_beds - 1
            )
        elif ward == Ward.GENERAL:
            self.resources.general_beds = max(
                0, self.resources.general_beds - 1
            )

    def _apply_deterioration(self):
        """Every 3 steps — critical patients deteriorate"""
        if self.step_count % 3 == 0:
            for patient in self.queue:
