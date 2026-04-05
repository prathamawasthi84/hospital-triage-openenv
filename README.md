---
title: Hospital Triage OpenEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8004
tags:
  - openenv
pinned: false
---

# 🏥 Hospital Emergency Triage — OpenEnv

A real-world OpenEnv environment where an AI agent learns
to triage emergency patients under dynamic conditions,
limited resources, and time pressure.

Built for the **Meta × Scaler OpenEnv Hackathon**.

---

## 🌍 Real-World Motivation

Over **250,000 patients are misTriaged** in emergency rooms
globally every year. Triage nurses make life-or-death
decisions under extreme stress and fatigue.

This environment simulates that exact problem — giving AI
agents a safe place to learn optimal triage policies before
any real patient is at risk.

---

## 🎯 Environment Overview

| Property | Value |
|---|---|
| Environment Name | hospital-triage-openenv |
| Task Type | Real-world ER Triage Simulation |
| Action Space | Priority + Ward + Treatment |
| Observation Space | Patient vitals, queue, resources |
| Reward Range | 0.0 – 1.0 |
| Episode Length | 10–20 steps |
| Difficulty Levels | Easy / Medium / Hard |

---

## 🔁 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /reset | Start new episode, returns Observation |
| POST | /step | Submit action, returns obs+reward+done |
| GET | /state | Get full current environment state |
| GET | /docs | Interactive API documentation |

---

## 👁️ Observation Space

What the agent sees at each step:
```json
{
  "current_patient": {
    "id": "P001",
    "age": 67,
    "blood_pressure": "180/110",
    "heart_rate": 115,
    "oxygen_saturation": 94.0,
    "symptoms": ["chest pain", "sweating", "left arm pain"],
    "wait_time": 0,
    "deteriorating": false
  },
  "queue": [...],
  "resources": {
    "ICU_beds": 2,
    "general_beds": 3,
    "doctors": 3,
    "OR_rooms": 1,
    "blood_supply": "normal"
  },
  "step": 4,
  "max_steps": 20,
  "task_level": "easy"
}
```

---

## ⚡ Action Space

What the agent must return at each step:
```json
{
  "patient_id": "P001",
  "priority": "immediate",
  "ward": "ICU",
  "treatment": "cardiac_protocol",
  "reasoning": "Chest pain + elevated HR + low O2 indicates cardiac emergency"
}
```

### Priority Options
| Value | Meaning |
|---|---|
| immediate | Life threatening — treat now |
| urgent | Serious but stable |
| non_urgent | Minor — can wait |
| deceased | No intervention possible |

### Ward Options
| Value | Meaning |
|---|---|
| ICU | Intensive care unit |
| emergency | Emergency room |
| general | General ward |
| waiting | Waiting area |

### Treatment Options
| Value | Meaning |
|---|---|
| cardiac_protocol | Heart related emergency |
| trauma_protocol | Physical trauma/injury |
| respiratory_protocol | Breathing emergency |
| basic_care | Standard treatment |
| observe_only | Monitor only |

---

## 📊 Reward Function

Rewards are given at **every step** — not just episode end.

### Easy Task
| Component | Weight |
|---|---|
| Priority correct | +0.40 |
| Ward correct | +0.30 |
| Treatment correct | +0.30 |
| Wrong priority on critical patient | -0.50 penalty |

### Medium Task
| Component | Weight |
|---|---|
| Survival rate of queue | +0.40 |
| Resource efficiency | +0.30 |
| Wait time score | +0.20 |
| Deterioration penalty | -0.10 per missed critical |

### Hard Task
| Component | Weight |
|---|---|
| Tagging accuracy | +0.35 |
| Lives saved score | +0.35 |
| Resource optimality | +0.20 |
| Unnecessary deferrals | -0.10 per mistake |

---

## 🎮 3 Task Levels

### ✅ Easy — Single Patient Triage
- 1 patient at a time
- Clear vitals and symptoms
- No resource constraints
- 10 steps per episode

### ⚠️ Medium — Multi-Patient Queue
- 5 patients simultaneously
- Only 2 ICU beds available
- Deterioration mechanic active
- 15 steps per episode

### 🔴 Hard — Mass Casualty Incident
- 15 casualties from disaster
- Critically scarce resources
- START triage tags required
- Re-triage deteriorating patients
- 20 steps per episode

---

## 🚀 Quick Start

### Run with Docker
```bash
docker build -t hospital-triage-openenv .
docker run -p 8004:8004 hospital-triage-openenv
```

### Test the API
```bash
# Reset environment
curl -X POST http://localhost:8004/reset

# Submit an action
curl -X POST http://localhost:8004/step \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","priority":"immediate","ward":"ICU","treatment":"cardiac_protocol"}'

# Get current state
curl http://localhost:8004/state
```

---

## 🤖 Run Baseline Inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_hf_token

python inference.py
```

---

## 📈 Baseline Scores

Scores produced by GPT-4o-mini baseline agent:

| Task | Score |
|---|---|
| Easy | TBD |
| Medium | TBD |
| Hard | TBD |

*(Updated after baseline run on Day 3)*

---

## 📁 Project Structure
hospital-triage-openenv/
├── env.py           # Core environment
├── models.py        # Pydantic typed models
├── tasks.py         # Patient scenario datasets
├── rubrics.py       # Grader functions
├── server.py        # FastAPI server
├── inference.py     # Baseline agent script
├── openenv.yaml     # Environment config
├── Dockerfile       # Container setup
└── README.md        # This file

---

## ⚙️ Environment Variables

| Variable | Description |
|---|---|
| API_BASE_URL | LLM API endpoint |
| MODEL_NAME | Model identifier |
| HF_TOKEN | HuggingFace API token |

---

## 👥 Team

Built with ❤️ for the Meta × Scaler OpenEnv Hackathon
