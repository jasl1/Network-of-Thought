# 🧠 Network-of-Thought: Multi-turn Jailbreak Framework for LLMs

**Network-of-Thought** is a universal, multi-turn, conversation-based jailbreak framework targeting cutting-edge Large Language Models (LLMs). It explores the adversarial space as a **"network of thought"**—an interconnected graph of real-world topics and entities—to systematically generate, refine, and execute jailbreak attacks in a controlled, research-focused setting.

---

## 🚀 Overview

This framework captures and simulates how a malicious user might steer a conversation toward harmful outputs, even when interacting with safeguarded LLMs. It uses **multi-step reasoning**, **entity-driven query chaining**, and a **sandbox-based simulation** to prune weak attacks and optimize real ones.

---

## 🔬 Methodology

1. **Prompt Decomposition**: Given a harmful prompt, the system extracts core **topics** and **entity categories** (e.g., chemicals, weapons).
2. **Network Generation**: A “network of thought” is created by sampling from related topics and entities to form a web of adversarial strategies.
3. **Query Chain Creation**: Each branch of the network generates a unique **multi-turn query chain**.
4. **Simulation Phase**: These chains are run through a **sandbox** (`simulation.py`) to refine and filter out ineffective prompts.
5. **Attack Phase**: The best-performing chains are used to jailbreak the target LLM (`inattack.py`).
6. **Judging**: Final responses are judged (`judge.py`) to assess if a successful jailbreak occurred.

---

## 📂 File Descriptions

| File             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `main.py`        | Entry point to run the full attack pipeline                                 |
| `simulation.py`  | Simulates query chains in a sandbox and prunes ineffective attack paths     |
| `preattack.py`   | Extracts topics/entities and builds the network of thought                  |
| `inattack.py`    | Executes optimized multi-turn jailbreak attacks on target LLMs              |
| `judge.py`       | Evaluates the success/failure of generated outputs                          |
| `utils.py`       | Utility functions used across modules (e.g., prompt formatting, logging)    |
| `data/harmbench.csv` | CSV containing harmful prompt seeds and behavioral targets            |

---

## ⚙️ How to Run

Run a simple attack test with 3 questions from `harmbench.csv` using GPT-4o:

```bash
python3 main.py --questions 3 \
                --behavior ./data/harmbench.csv \
                --attack_model_name gpt-4o \
                --target_model_name gpt-4o
```

- `--questions`: Number of harmful prompts to test
- `--behavior`: Path to behavior/prompt dataset
- `--attack_model_name`: Model used to generate attack chains
- `--target_model_name`: Model being attacked

---

## 🧪 Use Cases

- Research on LLM vulnerabilities
- Evaluation of jailbreak defenses
- Study of multi-turn adversarial behavior
- Curriculum design for Secure & Private AI courses

---

## 🛡️ Motivation

While many LLMs are trained to reject harmful prompts, **multi-turn adversarial conversations** can still bypass safety filters. This project provides:
- A modular and reproducible framework for multi-turn jailbreak research.
- Insight into how attackers could think and plan.
- A testbed for evaluating next-generation defenses.

---

## 📌 Citation (Coming Soon)

This framework is part of an upcoming academic publication. Please stay tuned for a citation if you plan to use this in your work.
