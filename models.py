from dataclasses import Field

from pydantic import BaseModel, Field
from typing import List,Optional,Dict,Any
from enum import Enum

class Priority(str,Enum):
    IMMEDIATE = 'immediate' #life-threatening
    URGENT='urgent' #serious but stable
    NON_URGENT='non_urgent' #not serious
    DECEASED='deceased' #deceased patient, mrtak
    
class Ward(str,Enum):
    ICU="icu" #intensive or deep care
    GENERAL='general' #general care
    EMERGENCY='emergency' #emergency room
    WAITING='waiting' #waiting area

class Treatment(str,Enum):
    CARDIAC ="cardiac_protocol"
    TRAUMA = 'trauma_protocol'
    RESPIRATORY = "respiratory_protocol"
    BASIC ="basic_protocol"
    OBSERVE="observe_protocol"

class Severity(str,Enum):
    CRITICAL="critical" #immediate risk of death
    SERIOUS='serious'# serious  urgent attention needed
    MODERATE='moderate'#needs care, stable
    MINOR='minor' #not urgent
    
class TaskLevel(str,Enum):
    EASY="easy"
    MEDIUM='medium'
    HARD='hard'
#patient model
class Patient(BaseModel):
    id:str=field(...,description="Unique patient ID e.g.P001")  
    age:int=field(...,ge=0,le=120)
    blood_pressure:str=field(...,description='blood pressure reading e.g.120/180')
    heart_rate:int=field(...,ge=0,le=300)
    oxygen_saturation:float=field(...,ge=0.0,le=100.0)
    symptoms: List[str] = Field(..., description="List of reported symptoms")
    severity: Optional[Severity] = Field(None, description="Ground truth severity — hidden from agent")
    wait_time: int = Field(default=0, description="Minutes patient has been waiting")
    deteriorating: bool = Field(default=False, description="Is patient getting worse each step")
    assigned_ward: Optional[Ward] = Field(None, description="Ward assigned by agent")
    assigned_priority: Optional[Priority] = Field(None, description="Priority assigned by agent")
    

## OBSERVATION — what agent sees  

class Resources(BaseModel):
    ICU_beds: int = Field(..., ge=0)
    general_beds: int = Field(..., ge=0)
    doctors: int = Field(..., ge=0)
    OR_rooms: int = Field(..., ge=0)
    blood_supply: str = Field(default="normal", description="normal / low / critical")

class Observation(BaseModel):
    current_patient: Optional[Patient] = Field(None, description="Patient agent must triage now")
    queue: List[Patient] = Field(default=[], description="Visible waiting patients")
    resources: Resources = Field(..., description="Currently available hospital resources")
    step: int = Field(default=0, description="Current step number in episode")
    max_steps: int = Field(default=20, description="Total steps in this episode")
    task_level: TaskLevel = Field(..., description="easy / medium / hard")
    message: str = Field(default="", description="Optional context message for agent")

    class Config:
        use_enum_values = True

# ACTION — what agent decides

class Action(BaseModel):
    patient_id: str = Field(..., description="ID of patient being triaged")
    priority: Priority = Field(..., description="Triage priority assigned")
    ward: Ward = Field(..., description="Ward to send patient to")
    treatment: Treatment = Field(..., description="Treatment protocol to apply")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning — used in hard task scoring")

    class Config:
        use_enum_values = True
    
# REWARD — score breakdown

class RewardBreakdown(BaseModel):
    priority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ward_score: float = Field(default=0.0, ge=0.0, le=1.0)
    treatment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    wait_time_penalty: float = Field(default=0.0, ge=-1.0, le=0.0)
    deterioration_penalty: float = Field(default=0.0, ge=-1.0, le=0.0)

class Reward(BaseModel):
    total: float = Field(..., ge=0.0, le=1.0, description="Final reward score 0.0 to 1.0")
    breakdown: RewardBreakdown = Field(..., description="Per-component score breakdown")
    feedback: str = Field(default="", description="Human readable feedback for agent")

# ENV STATE — full environment snapshot


class EnvState(BaseModel):
    queue: List[Patient] = Field(default=[], description="All patients in the system")
    resources: Resources = Field(..., description="Current resource availability")
    step: int = Field(default=0)
    max_steps: int = Field(default=20)
    total_reward: float = Field(default=0.0, description="Cumulative reward so far")
    task_level: TaskLevel = Field(..., description="Current task difficulty")
    done: bool = Field(default=False, description="Is the episode finished")
    patients_treated: int = Field(default=0)
    patients_deteriorated: int = Field(default=0)
    patients_deceased: int = Field(default=0)

    class Config:
        use_enum_values = True

# STEP RESPONSE — what env.step() returns
class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default={})