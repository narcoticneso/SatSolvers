# SAT Solvers: DPLL and Simulated Annealing

This project implements multiple algorithms for solving the Boolean Satisfiability Problem (SAT), with a primary focus on a complete DPLL-based solver and a heuristic Simulated Annealing solver. The project compares a guaranteed but potentially expensive search algorithm with faster stochastic approaches that scale better to large problem instances.

---

## What is SAT?

The Boolean Satisfiability Problem (SAT) asks whether there exists an assignment of True or False values to variables such that a Boolean formula evaluates to True. SAT was the first problem proven to be NP-complete and is a foundational problem in computer science with many practical applications.

### Example (Satisfiable)

Formula:
```
(x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
```

One satisfying assignment is:
```
x1 = True, x2 = False, x3 = True
```

### Example (Unsatisfiable)

Formula:
```
(x1) AND (NOT x1)
```

No assignment can satisfy both clauses simultaneously, so the formula is unsatisfiable.

---

## Project Overview

This project implements and evaluates the following SAT-solving approaches:

- **DPLL (Davis-Putnam-Logemann-Loveland)**  
  A complete backtracking-based algorithm that guarantees correctness.

- **Simulated Annealing**  
  A stochastic local-search algorithm that efficiently finds high-quality solutions for large instances.

- **Genetic Algorithm**  
  Implemented as an alternative heuristic approach, but not run by default due to slower performance compared to Simulated Annealing.

---

## Algorithms

### DPLL Solver

The DPLL solver systematically explores variable assignments while aggressively pruning the search space.

**Techniques used:**
- Unit propagation
- Pure literal elimination
- Jeroslow–Wang branching heuristic
- Recursive backtracking with conflict detection

**Strengths:**
- Guarantees a solution if one exists
- Can prove unsatisfiability
- Effective for small to medium-sized problems

**Limitations:**
- Exponential worst-case runtime
- Slower on large or difficult instances

---

### Simulated Annealing Solver

The Simulated Annealing solver is implemented as a function (`simmulatedAnnealing`) that performs probabilistic local search.

**How it works:**
- Starts from a random assignment
- Iteratively flips variables to improve clause satisfaction
- Accepts worse moves with probability `exp(ΔE / T)`
- Gradually reduces temperature using geometric cooling

**Key characteristics:**
- Fitness defined as number of satisfied clauses
- Incremental updates for efficiency
- Designed for scalability on large instances

**Limitations:**
- Not guaranteed to find a satisfying assignment
- Cannot prove unsatisfiability
- Sensitive to parameter choices

---

### Genetic Algorithm

A population-based Genetic Algorithm is implemented using crossover, mutation, and fitness-based selection. While functional, it was observed to be significantly slower than the other approaches and is therefore excluded from default experiments.

---

## Installation

```bash
git clone https://github.com/narcoticneso/SatSolvers.git
cd SatSolvers
```

This project uses only the Python standard library.

---

## Usage

### Running experiments

Executing the main file will run both DPLL and Simulated Annealing benchmarks:

```bash
python SatSolver.py
```

This will:
- Test DPLL on all benchmark CNF files
- Test Simulated Annealing with multiple runs per problem
- Generate CSV result files

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
is_sat = DPLL(clauses, numVariables)
print(f"Satisfiable: {is_sat}")

# Simulated Annealing solver
solution = simmulatedAnnealing(
    clauses=clauses,
    numVariables=numVariables,
    initialTemp=10,
    coolingRate=0.95
)

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

## Output Files

Results are written to CSV files:

**DPLLResults.csv**
- Problem name
- Difficulty tier
- SAT result
- CPU time

**SA_Results.csv**
- Problem name
- Difficulty tier
- Run number
- CPU time
- Satisfied clauses
- Total clauses
- SAT label

---

## Key Functions

### Parsing and Evaluation
- `parseFile`
- `fitness`
- `buildOccurrenceLists`

### DPLL
- `DPLL`
- `dpllRecursive`
- `findUnitClause`
- `findPureLiteral`
- `chooseLiteral`
- `simplifyClauses`

### Simulated Annealing
- `simmulatedAnnealing`
- `flipVariable`
- `fitnessDelta`

### Genetic Algorithm
- `geneticAlgorithm`
- `testGeneticAlgorithm`

---

## Applications

SAT solvers are widely used in:
- Hardware and software verification
- Constraint satisfaction and scheduling
- Cryptographic analysis
- Bioinformatics
- Configuration and planning systems
