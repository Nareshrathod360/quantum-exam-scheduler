# Qiskit Fall Fest 2026 — Project Submission Template
**IIIT Kottayam Quantum Hackathon (24-Hour Sprint)**

> ⚠️ **CRITICAL INSTRUCTIONS FOR PARTICIPANTS:**
> 1. **Do NOT rename or delete the section headings** below. The automated AI evaluation and filtering system relies on this exact Markdown structure to parse and grade your project.
> 2. Replace all placeholder text enclosed in `[brackets]` with your project's details.
> 3. **Commit History Audit:** All project commits must be timestamped strictly within the 24-hour hackathon window (between **October 10, 2026 at 7:05 PM** and **October 11, 2026 at 7:05 PM**). Commits observed outside this window will result in **immediate disqualification**.
> 4. Ensure all Jupyter Notebooks have their cells **run and outputs displayed** before pushing.
> 5. Starter resources and reference guides are located in the [`references/`](./references/) folder.
> 6. **Tech Stack Flexibility:** You are free to use any external frameworks/technologies (e.g., Streamlit, React, Pygame, FastAPI, NetworkX) for your frontend, app logic, or classical baselines. However, **Qiskit MUST power the quantum core** of your project.
> 7. **Full Event Guide & Rules:** For full schedule details, track inspirations, submission steps, and grading focus areas, refer to [`HACKATHON_GUIDE.md`](./HACKATHON_GUIDE.md).

*(You may delete the instructions box above when preparing your final repository submission)*

---

# Project Title

## Project Metadata

* **Project Title:** `[Your Project Title]`
* **Team Name:** `[Your Team Name]`
* **Chosen Track:** `[Track 1: Quantum Games & Interactive Puzzles OR Track 2: Open-Domain Optimization & Real-World Solvers]`
* **Video Presentation Link:** `[Insert YouTube Unlisted / Google Drive / Loom URL here]`
  *(⚠️ Crucial: Ensure link access is set to "Anyone with the link can view" on Google Drive or "Unlisted / Public" on YouTube. Locked/inaccessible links cannot be scored!)*
* **Live Demo Timestamp in Video:** `[MM:SS]` *(e.g., "Demo starts at 03:45")*

### Team Members (Max 4 members)
| Name | Email | GitHub Username |
| :--- | :--- | :--- |
| `[Member 1 Name]` | `[Email]` | `[@username]` |
| `[Member 2 Name]` | `[Email]` | `[@username]` |
| `[Member 3 Name]` | `[Email]` | `[@username]` |
| `[Member 4 Name]` | `[Email]` | `[@username]` |

---

## 1. Project Overview & Motivation

* **Summary (3–4 sentences):**  
  `[Provide a concise executive summary of what you built. What is the game or optimization dilemma?]`

* **Why this problem / game?**  
  `[Explain why you chose this specific topic. Why is it engaging, practically relevant, or pedagogically meaningful?]`

---

## 2. Quantum Architecture & Mechanics

* **Core Quantum Principles Used:**  
  `[Specify the exact quantum principles at play: e.g., Superposition (H gate), Entanglement (Bell states / CNOT), Measurement Collapse, Interference, or QAOA Cost Hamiltonian.]`

* **How the Quantum Circuit is Integrated:**  
  `[Explain how quantum states map to your game board, strategy, or optimization problem. How does your Qiskit circuit interact with player input or the optimization loop?]`

* **Circuit Details:**  
  * **Number of Qubits:** `[e.g., 4 qubits]`
  * **Key Gates Used:** `[e.g., H, CX, RZ, Measurement]`
  * **Primitives Used:** `[e.g., StatevectorSampler, StatevectorEstimator, or Qiskit Aer simulator]`
  * **Other Technologies Used:** `[e.g., Streamlit for UI, NetworkX for graph generation, Pygame, etc. (Note: Qiskit must power the quantum logic)]`

```python
# Optional: Paste a representative snippet of your core Qiskit circuit code here
```

---

## 3. Repository Structure & How to Run

### Project Structure
```text
├── README.md               # Project documentation (this file)
├── requirements.txt        # Python dependencies
├── src/                    # Source code scripts or game application
│   └── main.py             # Main entry point (or streamlit app)
├── notebooks/              # Jupyter notebooks containing experiments
│   └── solution.ipynb      # Main runnable notebook with outputs
└── references/             # Hackathon reference guides (Track 1 & Track 2)
```

### Installation & Execution Instructions

> 💡 **Environment Recommendation:** To avoid package conflicts, we strongly suggest using a fresh Python virtual environment or Google Colab:
> ```bash
> python -m venv venv
> source venv/bin/activate    # On Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

1. **Clone the repository:**
   ```bash
   git clone [your-repo-url]
   cd [your-repo-folder]
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the project / demo:**
   ```bash
   # If running a script or CLI game:
   python src/main.py

   # If running a Streamlit web app:
   streamlit run src/app.py

   # Or open the Jupyter Notebook:
   jupyter notebook notebooks/solution.ipynb
   ```

---

## 4. Implementation Walkthrough & Live Demo

* **Walkthrough of Gameplay / Optimization Pipeline:**  
  `[Describe the user experience or algorithmic workflow step-by-step. What inputs does the user provide, what happens inside the quantum circuit, and what output is produced?]`

* **Key Screenshots / Terminal Output:**  
  `[Paste sample terminal output, game screenshots, or optimization convergence plots here to demonstrate that the project runs.]`

---

## 5. Classical vs. Quantum Comparison

* **Classical Baseline / Counterpart:**  
  `[What classical baseline did you compare against? (e.g., a classical player, a random guesser, a brute-force classical solver, or a naive greedy heuristic)]`

* **Comparative Analysis:**  
  `[How did the quantum approach compare to the classical baseline? For Track 1: Did quantum strategy give an unfair advantage or unique gameplay behavior? For Track 2: What was the approximation ratio (C_quantum / C_optimal) and feasibility yield?]`

---

## 6. Performance, Results & Limitations

* **Key Results & Observations:**  
  `[Summarize the main results from your experiments, simulations, or gameplay tests.]`

* **Critical Reflection on Limitations:**  
  `[Be honest and rigorous: What happens if this is scaled to larger qubit counts? How would physical quantum noise, gate errors, and decoherence affect your solution on real NISQ hardware? Acknowledging limitations demonstrates true scientific understanding!]`

---

## 7. References & Acknowledgments

* `[Cite any open-source libraries, GitHub repositories, or research papers you referenced during the hackathon.]`
* `[Acknowledge team contributions and tools used.]`
