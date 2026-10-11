# Quantum Exam Scheduler Benchmark Summary

**Random-guess feasibility = 30/32768 (0.0916%)**

## Performance Comparison Table

| experiment       | instance | method               | reps | penalty | runs | n_valid_runs | valid_run_rate | mean_feasibility_yield | median_feasibility_yield | optimum_found_rate | mean_optimality_gap | mean_slots_used | median_runtime_s | min_runtime_s | max_runtime_s |
| ---------------- | -------- | -------------------- | ---- | ------- | ---- | ------------ | -------------- | ---------------------- | ------------------------ | ------------------ | ------------------- | --------------- | ---------------- | ------------- | ------------- |
| E1_headline      | sample   | exact                | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 100.0%             | 0.0%                | 3.0             | 0.000147         | 0.000147      | 0.000147      |
| E1_headline      | sample   | greedy-DSATUR        | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 0.0%               | 14.3%               | 3.0             | 0.000155         | 0.000155      | 0.000155      |
| E1_headline      | sample   | greedy-largest_first | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 0.0%               | 14.3%               | 3.0             | 0.000318         | 0.000318      | 0.000318      |
| E1_headline      | sample   | qaoa-p1              | 1    | 1.5     | 10   | 10           | 100.0%         | 1.36%                  | 1.64%                    | 60.0%              | 15.7%               | 3.0             | 8.14             | 4.62          | 11.6          |
| E1_headline      | sample   | qaoa-p2              | 2    | 1.5     | 3    | 3            | 100.0%         | 4.20%                  | 0.54%                    | 66.7%              | 4.8%                | 3.0             | 61.8             | 54.3          | 64.6          |
| E1_headline      | sample   | random-sampling      | -    | -       | 20   | 19           | 95.0%          | 0.08%                  | 0.09%                    | 25.0%              | 33.8%               | 3.0             | 0.000229         | 0.000205      | 0.000375      |
| E2_penalty_sweep | sample   | qaoa-p1              | 1    | 1.0     | 3    | 3            | 100.0%         | 0.68%                  | 0.93%                    | 66.7%              | 17.5%               | 3.0             | 9.95             | 7.36          | 10.2          |
| E2_penalty_sweep | sample   | qaoa-p1              | 1    | 1.1     | 3    | 3            | 100.0%         | 0.48%                  | 0.15%                    | 33.3%              | 12.7%               | 3.0             | 9.82             | 9.58          | 10.9          |
| E2_penalty_sweep | sample   | qaoa-p1              | 1    | 1.5     | 3    | 3            | 100.0%         | 1.28%                  | 1.51%                    | 66.7%              | 7.9%                | 3.0             | 9.16             | 8.16          | 9.57          |
| E2_penalty_sweep | sample   | qaoa-p1              | 1    | 2.0     | 3    | 3            | 100.0%         | 0.83%                  | 0.12%                    | 33.3%              | 22.2%               | 3.0             | 11.0             | 8.47          | 12.7          |
| E2_penalty_sweep | sample   | qaoa-p1              | 1    | 3.0     | 3    | 3            | 100.0%         | 1.07%                  | 0.39%                    | 33.3%              | 34.9%               | 3.0             | 15.0             | 12.8          | 24.5          |
| E3_scaling       | ring_3   | exact                | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 100.0%             | 0.0%                | 3.0             | 0.0000457        | 0.0000457     | 0.0000457     |
| E3_scaling       | ring_3   | greedy-largest_first | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 100.0%             | 0.0%                | 3.0             | 0.000533         | 0.000533      | 0.000533      |
| E3_scaling       | ring_3   | qaoa-p1              | 1    | 1.5     | 3    | 3            | 100.0%         | 5.62%                  | 6.76%                    | 66.7%              | 13.3%               | 3.0             | 1.12             | 1.01          | 1.23          |
| E3_scaling       | ring_3   | random-sampling      | -    | -       | 20   | 20           | 100.0%         | 1.10%                  | 1.10%                    | 100.0%             | 0.0%                | 3.0             | 0.000340         | 0.000230      | 0.000985      |
| E3_scaling       | ring_4   | exact                | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 100.0%             | 0.0%                | 2.0             | 0.000121         | 0.000121      | 0.000121      |
| E3_scaling       | ring_4   | greedy-largest_first | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 100.0%             | 0.0%                | 2.0             | 0.000124         | 0.000124      | 0.000124      |
| E3_scaling       | ring_4   | qaoa-p1              | 1    | 1.5     | 3    | 3            | 100.0%         | 5.44%                  | 4.61%                    | 66.7%              | 51.5%               | 2.33            | 1.97             | 1.58          | 2.09          |
| E3_scaling       | ring_4   | random-sampling      | -    | -       | 20   | 20           | 100.0%         | 0.47%                  | 0.45%                    | 65.0%              | 20.3%               | 2.25            | 0.000342         | 0.000280      | 0.000475      |
| E3_scaling       | ring_5   | exact                | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 100.0%             | 0.0%                | 3.0             | 0.000249         | 0.000249      | 0.000249      |
| E3_scaling       | ring_5   | greedy-largest_first | -    | -       | 1    | 1            | 100.0%         | -                      | -                        | 100.0%             | 0.0%                | 3.0             | 0.000115         | 0.000115      | 0.000115      |
| E3_scaling       | ring_5   | qaoa-p1              | 1    | 1.5     | 3    | 3            | 100.0%         | 1.15%                  | 1.61%                    | 66.7%              | 28.4%               | 3.0             | 9.40             | 9.26          | 9.90          |
| E3_scaling       | ring_5   | random-sampling      | -    | -       | 20   | 19           | 95.0%          | 0.08%                  | 0.09%                    | 15.0%              | 37.3%               | 3.0             | 0.000345         | 0.000307      | 0.000432      |

### Experiment Notes
- **E1 (Headline):** Compares exact search, greedy heuristics, classical random sampling, and QAOA on the benchmark problem to evaluate solution quality and feasibility yield.
- **E2 (Penalty sweep):** Analyzes how the QUBO constraint penalty parameter affects ground-state validity and QAOA sampling success.
- **E3 (Scaling):** Assesses solver feasibility yield and runtime scaling as the number of courses (qubit count) increases.

### Random-Guess Feasibility by Problem Size (E3)
- **ring_3** (3 courses, 3 slots, 9 qubits): 6/512 (1.1719%)
- **ring_4** (4 courses, 3 slots, 12 qubits): 18/4096 (0.4395%)
- **ring_5** (5 courses, 3 slots, 15 qubits): 30/32768 (0.0916%)
