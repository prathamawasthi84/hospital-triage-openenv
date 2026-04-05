from models import (
    Patient, Resources, Observation,
    Severity, Priority, Ward, Treatment, TaskLevel
    )
from typing import List, Dict, Any
import random
import copy

EASY_PATIENTS: List[Dict] = [
    {
        "id": "P001",
        "age": 67,
        "blood_pressure": "180/110",
        "heart_rate": 115,
        "oxygen_saturation": 94.0,
        "symptoms": ["chest pain", "sweating", "left arm pain"],
        "severity": Severity.CRITICAL,
        "correct_priority": Priority.IMMEDIATE,
        "correct_ward": Ward.ICU,
        "correct_treatment": Treatment.CARDIAC,
    },
    {
        "id": "P002",
        "age": 23,
        "blood_pressure": "120/80",
        "heart_rate": 72,
        "oxygen_saturation": 98.5,
        "symptoms": ["mild headache", "fatigue"],
        "severity": Severity.MINOR,
        "correct_priority": Priority.NON_URGENT,
        "correct_ward": Ward.WAITING,
        "correct_treatment": Treatment.OBSERVE,
    },
    {
        "id": "P003",
        "age": 45,
        "blood_pressure": "90/60",
        "heart_rate": 130,
        "oxygen_saturation": 91.0,
        "symptoms": ["difficulty breathing", "blue lips", "confusion"],
        "severity": Severity.CRITICAL,
        "correct_priority": Priority.IMMEDIATE,
        "correct_ward": Ward.ICU,
        "correct_treatment": Treatment.RESPIRATORY,
    },
    {
        "id": "P004",
        "age": 34,
        "blood_pressure": "130/85",
        "heart_rate": 88,
        "oxygen_saturation": 97.0,
        "symptoms": ["broken arm", "moderate pain"],
        "severity": Severity.MODERATE,
        "correct_priority": Priority.URGENT,
        "correct_ward": Ward.EMERGENCY,
        "correct_treatment": Treatment.BASIC,
    },
    {
        "id": "P005",
        "age": 78,
        "blood_pressure": "160/100",
        "heart_rate": 95,
        "oxygen_saturation": 93.0,
        "symptoms": ["sudden severe headache", "vomiting", "vision loss"],
        "severity": Severity.CRITICAL,
        "correct_priority": Priority.IMMEDIATE,
        "correct_ward": Ward.ICU,
        "correct_treatment": Treatment.CARDIAC,
    },
    {
        "id": "P006",
        "age": 19,
        "blood_pressure": "118/76",
        "heart_rate": 68,
        "oxygen_saturation": 99.0,
        "symptoms": ["sore throat", "mild fever"],
        "severity": Severity.MINOR,
        "correct_priority": Priority.NON_URGENT,
        "correct_ward": Ward.WAITING,
        "correct_treatment": Treatment.BASIC,
    },
    {
        "id": "P007",
        "age": 55,
        "blood_pressure": "85/55",
        "heart_rate": 140,
        "oxygen_saturation": 89.0,
        "symptoms": ["stab wound abdomen", "heavy bleeding", "pale skin"],
        "severity": Severity.CRITICAL,
        "correct_priority": Priority.IMMEDIATE,
        "correct_ward": Ward.ICU,
        "correct_treatment": Treatment.TRAUMA,
    },
    {
        "id": "P008",
        "age": 62,
        "blood_pressure": "145/92",
        "heart_rate": 82,
        "oxygen_saturation": 95.0,
        "symptoms": ["chest tightness", "mild shortness of breath"],
        "severity": Severity.SERIOUS,
        "correct_priority": Priority.URGENT,
        "correct_ward": Ward.EMERGENCY,
        "correct_treatment": Treatment.CARDIAC,
    },
    {
        "id": "P009",
        "age": 8,
        "blood_pressure": "100/65",
        "heart_rate": 110,
        "oxygen_saturation": 92.0,
        "symptoms": ["high fever 104F", "seizure", "stiff neck"],
        "severity": Severity.CRITICAL,
        "correct_priority": Priority.IMMEDIATE,
        "correct_ward": Ward.ICU,
        "correct_treatment": Treatment.BASIC,
    },
    {
        "id": "P010",
        "age": 40,
        "blood_pressure": "122/78",
        "heart_rate": 76,
        "oxygen_saturation": 98.0,
        "symptoms": ["sprained ankle", "mild swelling"],
        "severity": Severity.MINOR,
        "correct_priority": Priority.NON_URGENT,
        "correct_ward": Ward.WAITING,
        "correct_treatment": Treatment.BASIC,
    },
]

# MEDIUM_TASK 
MEDIUM_SCENARIOS: List[Dict] = [
    {
        "scenario_id": "M001",
        "patients": [
            {
                "id": "M001_P1", "age": 70,
                "blood_pressure": "190/120", "heart_rate": 120,
                "oxygen_saturation": 91.0,
                "symptoms": ["chest pain", "sweating"],
                "severity": Severity.CRITICAL,
                "correct_priority": Priority.IMMEDIATE,
                "correct_ward": Ward.ICU,
                "correct_treatment": Treatment.CARDIAC,
                "deteriorating": True,
            },
            {
                "id": "M001_P2", "age": 25,
                "blood_pressure": "115/75", "heart_rate": 70,
                "oxygen_saturation": 99.0,
                "symptoms": ["mild sprain", "bruising"],
                "severity": Severity.MINOR,
                "correct_priority": Priority.NON_URGENT,
                "correct_ward": Ward.WAITING,
                "correct_treatment": Treatment.BASIC,
                "deteriorating": False,
            },
            {
                "id": "M001_P3", "age": 55,
                "blood_pressure": "88/58", "heart_rate": 135,
                "oxygen_saturation": 90.0,
                "symptoms": ["gunshot wound leg", "bleeding"],
                "severity": Severity.SERIOUS,
                "correct_priority": Priority.URGENT,
                "correct_ward": Ward.EMERGENCY,
                "correct_treatment": Treatment.TRAUMA,
                "deteriorating": True,
            },
            {
                "id": "M001_P4", "age": 44,
                "blood_pressure": "130/88", "heart_rate": 90,
                "oxygen_saturation": 96.0,
                "symptoms": ["broken wrist", "moderate pain"],
                "severity": Severity.MODERATE,
                "correct_priority": Priority.URGENT,
                "correct_ward": Ward.EMERGENCY,
                "correct_treatment": Treatment.BASIC,
                "deteriorating": False,
            },
            {
                "id": "M001_P5", "age": 3,
                "blood_pressure": "95/60", "heart_rate": 160,
                "oxygen_saturation": 88.0,
                "symptoms": ["choking", "turning blue", "unresponsive"],
                "severity": Severity.CRITICAL,
                "correct_priority": Priority.IMMEDIATE,
                "correct_ward": Ward.ICU,
                "correct_treatment": Treatment.RESPIRATORY,
                "deteriorating": True,
            },
        ],
        "resources": {
            "ICU_beds": 2,
            "general_beds": 3,
            "doctors": 3,
            "OR_rooms": 1,
            "blood_supply": "normal"
        },
        # correct_order_by_priority_(most_critical_first)
        "correct_treat_order": ["M001_P1", "M001_P5", "M001_P3", "M001_P4", "M001_P2"],
        "correct_ICU_allocation": ["M001_P1", "M001_P5"],
    },
    {
        "scenario_id": "M002",
        "patients": [
            {
                "id": "M002_P1", "age": 60,
                "blood_pressure": "170/105", "heart_rate": 100,
                "oxygen_saturation": 93.0,
                "symptoms": ["stroke symptoms", "face drooping", "arm weakness"],
                "severity": Severity.CRITICAL,
                "correct_priority": Priority.IMMEDIATE,
                "correct_ward": Ward.ICU,
                "correct_treatment": Treatment.CARDIAC,
                "deteriorating": True,
            },
            {
                "id": "M002_P2", "age": 30,
                "blood_pressure": "125/82", "heart_rate": 78,
                "oxygen_saturation": 97.0,
                "symptoms": ["allergic reaction", "hives", "mild swelling"],
                "severity": Severity.MODERATE,
                "correct_priority": Priority.URGENT,
                "correct_ward": Ward.EMERGENCY,
                "correct_treatment": Treatment.BASIC,
                "deteriorating": False,
            },
            {
                "id": "M002_P3", "age": 50,
                "blood_pressure": "80/50", "heart_rate": 145,
                "oxygen_saturation": 87.0,
                "symptoms": ["severe burns 60 percent body", "shock"],
                "severity": Severity.CRITICAL,
                "correct_priority": Priority.IMMEDIATE,
                "correct_ward": Ward.ICU,
                "correct_treatment": Treatment.TRAUMA,
                "deteriorating": True,
            },
            {
                "id": "M002_P4", "age": 22,
                "blood_pressure": "118/76", "heart_rate": 72,
                "oxygen_saturation": 98.5,
                "symptoms": ["cut finger", "minor bleeding"],
                "severity": Severity.MINOR,
                "correct_priority": Priority.NON_URGENT,
                "correct_ward": Ward.WAITING,
                "correct_treatment": Treatment.BASIC,
                "deteriorating": False,
            },
            {
                "id": "M002_P5", "age": 67,
                "blood_pressure": "155/98", "heart_rate": 92,
                "oxygen_saturation": 94.0,
                "symptoms": ["diabetic emergency", "unconscious", "low blood sugar"],
                "severity": Severity.SERIOUS,
                "correct_priority": Priority.URGENT,
                "correct_ward": Ward.EMERGENCY,
                "correct_treatment": Treatment.BASIC,
                "deteriorating": True,
            },
        ],
        "resources": {
            "ICU_beds": 2,
            "general_beds": 2,
            "doctors": 2,
            "OR_rooms": 1,
            "blood_supply": "low"
        },
        "correct_treat_order": ["M002_P1", "M002_P3", "M002_P5", "M002_P2", "M002_P4"],
        "correct_ICU_allocation": ["M002_P1", "M002_P3"],
    },
]

# HARD_TASK 
HARD_SCENARIOS: List[Dict] = [
    {
        "scenario_id": "H001",
        "description": "Building collapse — 15 casualties",
        "patients": [
            {"id":"H001_P01","age":35,"blood_pressure":"70/40","heart_rate":150,"oxygen_saturation":82.0,"symptoms":["crushed chest","internal bleeding","unconscious"],"severity":Severity.CRITICAL,"correct_tag":"red","deteriorating":True},
            {"id":"H001_P02","age":28,"blood_pressure":"110/70","heart_rate":95,"oxygen_saturation":96.0,"symptoms":["broken leg","moderate pain","alert"],"severity":Severity.MODERATE,"correct_tag":"yellow","deteriorating":False},
            {"id":"H001_P03","age":72,"blood_pressure":"60/30","heart_rate":160,"oxygen_saturation":75.0,"symptoms":["massive head trauma","unresponsive","no pulse"],"severity":Severity.CRITICAL,"correct_tag":"black","deteriorating":False},
            {"id":"H001_P04","age":19,"blood_pressure":"125/80","heart_rate":78,"oxygen_saturation":98.0,"symptoms":["minor cuts","anxiety","walking"],"severity":Severity.MINOR,"correct_tag":"green","deteriorating":False},
            {"id":"H001_P05","age":45,"blood_pressure":"85/55","heart_rate":138,"oxygen_saturation":88.0,"symptoms":["tension pneumothorax","severe breathing difficulty"],"severity":Severity.CRITICAL,"correct_tag":"red","deteriorating":True},
            {"id":"H001_P06","age":60,"blood_pressure":"100/65","heart_rate":110,"oxygen_saturation":91.0,"symptoms":["spinal injury","paralysis below waist"],"severity":Severity.SERIOUS,"correct_tag":"yellow","deteriorating":False},
            {"id":"H001_P07","age":8,"blood_pressure":"95/58","heart_rate":155,"oxygen_saturation":86.0,"symptoms":["severe burns face and arms","shock"],"severity":Severity.CRITICAL,"correct_tag":"red","deteriorating":True},
            {"id":"H001_P08","age":33,"blood_pressure":"120/78","heart_rate":82,"oxygen_saturation":97.5,"symptoms":["dislocated shoulder","walking wounded"],"severity":Severity.MINOR,"correct_tag":"green","deteriorating":False},
            {"id":"H001_P09","age":55,"blood_pressure":"75/45","heart_rate":148,"oxygen_saturation":84.0,"symptoms":["penetrating abdominal wound","evisceration"],"severity":Severity.CRITICAL,"correct_tag":"red","deteriorating":True},
            {"id":"H001_P10","age":42,"blood_pressure":"130/85","heart_rate":88,"oxygen_saturation":95.0,"symptoms":["fractured ribs","painful breathing"],"severity":Severity.SERIOUS,"correct_tag":"yellow","deteriorating":False},
            {"id":"H001_P11","age":90,"blood_pressure":"55/30","heart_rate":0,"oxygen_saturation":0.0,"symptoms":["no pulse","no breathing","fixed pupils"],"severity":Severity.CRITICAL,"correct_tag":"black","deteriorating":False},
            {"id":"H001_P12","age":25,"blood_pressure":"118/76","heart_rate":74,"oxygen_saturation":98.0,"symptoms":["dust inhalation","mild cough","walking"],"severity":Severity.MINOR,"correct_tag":"green","deteriorating":False},
            {"id":"H001_P13","age":50,"blood_pressure":"80/50","heart_rate":142,"oxygen_saturation":85.0,"symptoms":["crush syndrome both legs","renal failure risk"],"severity":Severity.CRITICAL,"correct_tag":"red","deteriorating":True},
            {"id":"H001_P14","age":38,"blood_pressure":"128/84","heart_rate":90,"oxygen_saturation":94.0,"symptoms":["head laceration","mild concussion","responsive"],"severity":Severity.MODERATE,"correct_tag":"yellow","deteriorating":False},
            {"id":"H001_P15","age":65,"blood_pressure":"88/52","heart_rate":136,"oxygen_saturation":83.0,"symptoms":["cardiac arrest","CPR in progress"],"severity":Severity.CRITICAL,"correct_tag":"red","deteriorating":True},
        ],
        "resources": {
            "ICU_beds": 3,
            "general_beds": 4,
            "doctors": 2,
            "OR_rooms": 1,
            "blood_supply": "critical"
        },
        # START_triage_ground_truth_counts
        "correct_tags": {
            "black": ["H001_P03", "H001_P11"],           
            "red":   ["H001_P01","H001_P05","H001_P07",
                      "H001_P09","H001_P13","H001_P15"], 
            "yellow":["H001_P02","H001_P06","H001_P10",
                      "H001_P14"],                        
            "green": ["H001_P04","H001_P08","H001_P12"],  
        },
    },
]

# called_by_env.py_to_get_scenarios
def get_easy_patients(shuffle: bool = True) -> List[Dict]:
    """Returns list of easy patients — shuffled by default"""
    patients = copy.deepcopy(EASY_PATIENTS)
    if shuffle:
        random.shuffle(patients)
    return patients

def get_medium_scenario(scenario_id: str = None) -> Dict:
    """Returns a medium scenario — random if no ID given"""
    if scenario_id:
        for s in MEDIUM_SCENARIOS:
            if s["scenario_id"] == scenario_id:
                return copy.deepcopy(s)
    return copy.deepcopy(random.choice(MEDIUM_SCENARIOS))

def get_hard_scenario(scenario_id: str = None) -> Dict:
    """Returns a hard scenario — random if no ID given"""
    if scenario_id:
        for s in HARD_SCENARIOS:
            if s["scenario_id"] == scenario_id:
                return copy.deepcopy(s)
    return copy.deepcopy(random.choice(HARD_SCENARIOS))

def get_task(level: TaskLevel, scenario_id: str = None) -> Dict:
    """Main entry point — env.py calls this"""
    if level == TaskLevel.EASY:
        return {"patients": get_easy_patients(), "level": TaskLevel.EASY}
    elif level == TaskLevel.MEDIUM:
        return get_medium_scenario(scenario_id)
    elif level == TaskLevel.HARD:
        return get_hard_scenario(scenario_id)
    else:
        raise ValueError(f"Unknown task level: {level}")  
