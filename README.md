# Equation-2-RL (`eq2rl`)

> **Clean RL implementations in PyTorch & JAX for Study/Revision**
> `#come-study-with-me` `#basics` `#rl` `#marl` `#harl`

Hi! When I first started studying Reinforcement Learning, I desperately wanted a single, go-to repository with clean implementations in PyTorch and JAX; something that mapped directly to the math in the actual papers. Instead, I (more often than not) found myself digging through thousands of lines of production framework code.

This repo is strictly a study guide. We replicate Deep RL algorithms by stripping away all the engineering bloat (no heavy parallelization, no complex wrappers). Everything is implemented purely so you can see exactly how paper equations translate into clean, execution-ready code.

## Philosophy
* **Zero Bloat:** No vectorized environments, no complex multiprocessing frameworks, no hardware-specific optimization hacks. If it’s not in the original paper's pseudo-code, it doesn't go in here.
* **Conceptual Parity, Not Line-by-Line:** We don't force JAX to look like PyTorch. PyTorch scripts are naturally stateful, object-oriented, and imperative. JAX scripts are naturally stateless, functional, and transformed (`jit`, `grad`). We embrace the unique flavor of both.
* **100% Self-Contained:** No shared `utils.py` file or custom abstract base classes. If a script needs a replay buffer or a helper function, it's written right there inside the file. You can grab a single file, run it, and walk away.

## Flat Directory Structure

To keep things completely scannable, we avoid nested folder rabbit holes. Each algorithm gets exactly one folder with exactly three files:

```text
eq2rl/
├── README.md               # You are here!
│
├── dqn/
│   ├── math.md             # The paper math, directly referencing the paper and decoded into plain English
│   ├── pytorch_dqn.py      # Standalone PyTorch code
│   └── jax_dqn.py          # Standalone JAX code
│
└── ppo/
    ├── math.md
    ├── pytorch_ppo.py
    └── jax_ppo.py

```

---

## Algorithms

Here is our current progress checklist. Feel free to jump into any of the completed modules or grab a `WIP` one to study and finish it up!

| Algorithm | Type | Target Env | Theory (`math.md`) | PyTorch Status | JAX Status |
| --- | --- | --- | --- | --- | --- |
| **DQN** | Single-Agent (Value) | `CartPole-v1` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **REINFORCE** | Single-Agent (Policy) | `CartPole-v1` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **A2C** | Single-Agent (Actor-Critic) | `CartPole-v1` / `Pendulum-v1` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **PPO** | Single-Agent (Policy) | `Pendulum-v1` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **DDPG** | Single-Agent (Continuous) | `Pendulum-v1` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **SAC** | Single-Agent (Continuous) | `Pendulum-v1` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **Distributed DQN** | Single-Agent (Distributed) | `CartPole-v1` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **MADDPG** | Multi-Agent | `simple_spread` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |
| **MAPPO** | Multi-Agent | `simple_spread` | 🔴 Not Started | 🔴 Not Started | 🔴 Not Started |

*Legend: 🟢 Complete | 🟡 Work in Progress / Open PR | 🔴 Not Started*

---

## 🛠️ How to Use This Repo

1. Pick an algorithm you want to study (e.g., `dqn`).
2. Read the `math.md` inside that folder first. It starts with the intuition from the original paper and breaks down the math/algorithm into plain language.
3. Open `pytorch_dqn.py` and `jax_dqn.py` side-by-side.
4. Look at how PyTorch uses objects and backward passes while JAX uses stateless state dictionaries and functional loss math.

### Quick Start (Dependencies)

We keep dependencies completely minimal. You just need standard `gymnasium` (plus `PettingZoo` for multi-agent settings), `torch`, `jax`, and `optax`.

```bash
pip install gymnasium pettingzoo torch jax optax

```

To run any script, just execute it directly:

```bash
python dqn/pytorch_dqn.py

```
