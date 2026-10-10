# Track 1 Reference Guide: Quantum Games & Interactive Puzzles
**Qiskit Fall Fest 2026 — IIIT Kottayam**

Welcome to **Track 1**! Your goal is to design and prototype an interactive game, puzzle, or gamified demonstration where **quantum mechanical principles** are integral gameplay mechanics.

---

## 1. What Makes a Game "Quantum"?

A game is considered a true quantum game when quantum phenomena directly drive game states, player decisions, or win/loss resolution:

1. **Superposition ($H, RX, RY$ gates):**
   * *Game Concept:* A game tile, hidden card, or target ship exists in a probabilistic superposition of multiple states (e.g., neither pure hit nor pure miss) until a player performs a scan or attack.
2. **Entanglement ($CNOT, CZ$ gates):**
   * *Game Concept:* Two or more game pieces, board squares, or players are non-locally linked. Revealing or changing one piece instantly determines the state of its entangled partner.
3. **Measurement Collapse (`measure` operation):**
   * *Game Concept:* Game uncertainty and probabilistic branches are collapsed into definite classical outcomes upon player observation, resolving battles or triggering traps.
4. **Interference & Phase ($Z, S, T$ gates):**
   * *Game Concept:* Players use strategic gates to constructively interfere their own winning branches while destructively interfering their opponent's chances (e.g., Meyer's Quantum Penny Flip).

---

## 2. Project Architecture & Workflow

A typical quantum game project follows this simple architecture:

```
┌─────────────────────────────────┐
│     Player Input / Decision     │  (e.g., Choose Move, Fire, Entangle)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│   Qiskit Circuit Construction   │  (Apply H, X, CNOT, Phase gates)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│   Execution via Sampler / Aer   │  (Sample measurement outcomes)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  Game State Update & Display    │  (Update scores, board state, UI)
└─────────────────────────────────┘
```

You can build the user interface using:
- **Interactive Jupyter Notebook** (using Python `input()` loops or `ipywidgets`)
- **Terminal CLI** (simple colored text outputs)
- **Web UI** (using lightweight frameworks like **Streamlit** or **Gradio**)

---

## 3. Concrete Project Ideas & Inspiration

Here are a few proven ideas you can adapt, expand, or reinvent:

### Idea 1: Quantum Battleship / Fog-of-War
* **Mechanic:** A grid where tiles represent qubits. Ships are placed across multiple tiles using superposition ($H$ or $RY(\theta)$ rotations) or entangled across coordinates ($CNOT$).
* **Gameplay:** When an opponent fires a missile, the targeted qubit is measured. If the outcome is $|1\rangle$, it's a Hit; if $|0\rangle$, it's a Miss. Players can use "radar" scans or phase shields to manipulate probabilities.

### Idea 2: Quantum Tic-Tac-Toe
* **Mechanic:** In classic Quantum Tic-Tac-Toe (invented by Allan Goff), players place "quantum marks" across two squares simultaneously (entangled state).
* **Gameplay:** Squares can hold multiple marks. When a cycle of entanglement is formed, the board collapses into classical marks, determining the winner.

### Idea 3: Quantum Dungeon / Trapdoor Escape
* **Mechanic:** A player navigates a maze of rooms. Trapdoors and treasure chests are entangled.
* **Gameplay:** Opening Chest A reveals whether Door B is unlocked or booby-trapped. Players collect "Quantum Keys" (gates: $X$, $H$, $Z$) to modify the circuit before stepping into unknown rooms.

### Idea 4: Quantum Monty Hall / Card Duel
* **Mechanic:** A 3-door Monty Hall puzzle or card game where the prize is in a superposition across doors.
* **Gameplay:** Players choose classical doors or apply quantum operations. Show how a quantum contestant using entanglement or interference beats the standard classical 66% winning probability.

---

## 4. Starter Code & Resources

* **Starter Notebook:** See [`track1_game_starter.ipynb`](./track1_game_starter.ipynb) in this folder for working implementations of:
  * *Meyer's Quantum Penny Flip* (Demonstrating how superposition guarantees a 100% win rate against classical players).
  * *Entangled Trapdoor Mini-Game* (Demonstrating turn-based loops with Bell pairs and Qiskit's `StatevectorSampler`).

### Recommended External References:
* [Qiskit Documentation: Building Circuits & Measurement](https://docs.quantum.ibm.com/guides)
* [David A. Meyer: Quantum Strategies (Penny Flip Game)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.82.1052)
* [Allan Goff: Quantum Tic-Tac-Toe Paper](https://arxiv.org/abs/quant-ph/0202151)
* [Qiskit Community Awesome Quantum Games](https://github.com/qiskit-community/awesome-quantum-games)
* [Streamlit Documentation (for rapid game UIs)](https://docs.streamlit.io/)
