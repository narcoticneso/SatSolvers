# SAT Solvers: DPLL and Simulated Annealing

This project implements two different algorithms for solving the Boolean Satisfiability Problem (SAT): a complete backtracking solver based on DPLL and a heuristic solver based on Simulated Annealing. The goal is to compare a guaranteed but potentially slow method with a faster, approximate approach that scales better to large problems.

---

## What is SAT?

The Boolean Satisfiability Problem (SAT) asks whether there exists an assignment of True or False values to variables such that a Boolean formula evaluates to True. SAT was the first problem proven to be NP-complete and plays a central role in theoretical computer science and many real-world applications.

### Example (Satisfiable)

Formula:
```
(x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
```

One satisfying assignment is:
```
x1 = True, x2 = False, x3 = True
```

All clauses evaluate to True under this assignment.

### Example (Unsatisfiable)

Formula:
```
(x1) AND (NOT x1)
```

No assignment can satisfy both clauses simultaneously, so the formula is unsatisfiable.

---

## Project Overview

This project explores two different approaches to solving SAT problems:

- **DPLL (Davis-Putnam-Logemann-Loveland)**  
  A complete, systematic search algorithm that guarantees a correct result.

- **Simulated Annealing**  
  A probabilistic metaheuristic that trades guaranteed correctness for speed on large instances.

A Genetic Algorithm was also implemented during development but was significantly slower than the other approaches and is not included in the final evaluation.

---

## Algorithms

### DPLL Solver

DPLL is a backtracking-based SAT solver that explores variable assignments while aggressively pruning the search space.

**Key techniques used:**
- Unit propagation  
- Pure literal elimination  
- Jeroslow-Wang heuristic for variable selection  
- Recursive backtracking with conflict detection  

**Strengths:**
- Always finds a solution if one exists  
- Can prove that a formula is unsatisfiable  
- Works well on small to medium-sized problems  

**Limitations:**
- Exponential worst-case runtime  
- Performance degrades on large or difficult instances  

---

### Simulated Annealing Solver

Simulated Annealing is a stochastic optimization technique inspired by physical annealing processes. It searches for assignments that maximize the number of satisfied clauses.

**How it works:**
- Start with a random assignment  
- Flip variable values to improve clause satisfaction  
- Occasionally accept worse solutions to escape local optima  
- Gradually reduce randomness using a temperature schedule  

**Key features:**
- Configurable temperature and cooling rate  
- Fitness function based on satisfied clauses  
- Incremental updates for efficiency  

**Strengths:**
- Fast on large problem instances  
- Uses little memory  
- Produces high-quality solutions in practice  

**Limitations:**
- Not guaranteed to find a satisfying assignment  
- Cannot prove unsatisfiability  
- Sensitive to parameter tuning  

---

## Installation

```bash
git clone https://github.com/narcoticneso/SatSolvers.git
cd SatSolvers
```

No external dependencies are required. The project uses only the Python standard library.

---

## Usage

### Running the solvers

The main script evaluates both algorithms on benchmark SAT problems:

```bash
python SatSolver.py
```

This will:
- Run DPLL on all benchmark instances  
- Run Simulated Annealing multiple times per problem  
- Output results to CSV files for analysis  

---

### Input format

SAT problems are provided in DIMACS CNF format:

```
c comment
p cnf 3 3
1 2 0
-1 3 0
-2 -3 0
```

- `p cnf` specifies the number of variables and clauses  
- Each clause ends with `0`  
- Positive integers represent positive literals  
- Negative integers represent negated literals  

---

### Programmatic usage

```python
from SatSolver import *

numVariables, clauses = parseFile("problem.cnf")

# DPLL solver
result = DPLL(clauses, numVariables)
print(f"Satisfiable: {result}")

# Simulated Annealing solver
posOcc, negOcc = buildOccurrenceLists(clauses, numVariables)
sa = simulatedAnnealing(
    clauses=clauses,
    numVariables=numVariables,
    posOccurrences=posOcc,
    negOccurrences=negOcc,
    maxSteps=10000,
    initialTemp=100.0,
    coolingRate=0.95
)

solution = sa.solve()
numSatisfied, _, _ = fitness(clauses, solution)
print(f"Satisfied {numSatisfied}/{len(clauses)} clauses")
```

---

## Project Structure

```
SatSolver/
├── SatSolver.py
├── README.md
├── LICENSE
└── PA3_Benchmarks/
    ├── CNF Formulas/
    └── HARD CNF Formulas/
```

---

## Performance Summary

Performance depends heavily on problem size and difficulty.

### DPLL

| Variables | Difficulty | Typical Runtime |
|----------|------------|-----------------|
| < 50     | Regular    | < 0.1s          |
| 50–100  | Regular    | 0.1s–2s         |
| 100+    | Hard       | 2s–60s+         |

### Simulated Annealing

| Variables | Difficulty | Runtime | Clause Satisfaction |
|----------|------------|---------|---------------------|
| < 100    | Regular    | < 0.5s  | 95–100%             |
| 100–500  | Regular    | 0.5s–3s | 90–100%             |
| 500+     | Hard       | 3s–10s  | 85–95%              |

DPLL guarantees correctness, while Simulated Annealing prioritizes speed and scalability.

---

## Key Components

### Parsing and Evaluation
- `parseFile`
- `fitness`
- `buildOccurrenceLists`

### DPLL Implementation
- `DPLL`
- `dpllRecursive`
- `findUnitClause`
- `findPureLiteral`
- `chooseLiteral`
- `simplifyClauses`

### Simulated Annealing
- `simulatedAnnealing` class
- `solve`
- `flipVariable`
- `fitnessDelta`

---

## Output

Results are written to CSV files.

**DPLLResults.csv**
- Problem name  
- Difficulty tier  
- SAT result  
- Runtime  

**SimulatedAnnealingResults.csv**
- Problem name  
- Difficulty tier  
- Run number  
- Runtime  
- Satisfied clauses  
- Total clauses  
- SAT label  

---

## Applications

SAT solvers are widely used in:
- Hardware and software verification  
- Constraint satisfaction and scheduling  
- Cryptographic analysis  
- Bioinformatics  
- Configuration and planning systems  
