# SAT Solvers: DPLL and Simulated Annealing

Implementation of two SAT (Boolean Satisfiability) solving algorithms: the classical DPLL algorithm and a metaheuristic Simulated Annealing approach.

## What is SAT?

The Boolean Satisfiability Problem (SAT) is the problem of determining if there exists an assignment of truth values (True/False) to variables that makes a given Boolean formula evaluate to True. SAT is a fundamental problem in computer science and was the first problem proven to be NP-complete.

### Example (Satisfiable):
Given the formula: `(x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)`

Is there an assignment of True/False to x1, x2, x3 that makes the entire formula True?

Answer: **Yes!** Setting x1=True, x2=False, x3=True satisfies all clauses.

### Example (Unsatisfiable):
Given the formula: `(x1) AND (NOT x1)`

Is there an assignment that makes this formula True?

Answer: **No!** The first clause requires x1=True, but the second clause requires x1=False. There is no assignment that satisfies both clauses simultaneously. This formula is unsatisfiable.

## Project Overview

This project implements two different approaches to solving SAT problems:

1. **DPLL (Davis-Putnam-Logemann-Loveland)**: A complete, systematic algorithm that guarantees finding a solution if one exists
2. **Simulated Annealing**: A probabilistic metaheuristic that can efficiently find solutions for large instances

*Note: A Genetic Algorithm approach was also implemented but proved to be significantly slower than the other methods and is not recommended for use.*

## Algorithms

### DPLL Algorithm

DPLL is a complete backtracking-based search algorithm that systematically explores the space of possible variable assignments.

**Key Features:**
- **Unit Propagation**: Automatically assigns variables that appear alone in a clause
- **Pure Literal Elimination**: Assigns variables that appear with only one polarity
- **Jeroslow-Wang Heuristic**: Intelligently chooses which variable to branch on next
- **Backtracking**: Explores alternative assignments when conflicts are detected

**Advantages:**
- Guaranteed to find a solution if one exists
- Proves unsatisfiability when no solution exists
- Efficient for small to medium-sized problems

**Limitations:**
- Exponential worst-case time complexity
- Can be slow on large, difficult instances

### Simulated Annealing

Simulated Annealing is a metaheuristic optimization algorithm inspired by the physical process of annealing in metallurgy.

**How It Works:**
1. Start with a random variable assignment
2. Iteratively flip variables to improve the number of satisfied clauses
3. Accept worse solutions with a probability that decreases over time (controlled by "temperature")
4. Gradually "cool down" to converge on a high-quality solution

**Key Features:**
- **Temperature Schedule**: Controls exploration vs. exploitation trade-off
- **Fitness Function**: Counts the number of satisfied clauses
- **Incremental Updates**: Efficiently computes fitness changes after variable flips
- **Escape Local Optima**: Can accept worse solutions to avoid getting stuck

**Advantages:**
- Fast on large problem instances
- Good solution quality in practice
- Memory efficient

**Limitations:**
- Not guaranteed to find optimal solution
- Cannot prove unsatisfiability
- Requires parameter tuning (temperature, cooling rate)

## Installation

```bash
# Clone the repository
git clone https://github.com/Sixteen1-6/SatSolver.git
cd SatSolver

# No external dependencies required - uses only Python standard library
```

## Usage

### Running the Solvers

The main script tests both algorithms on benchmark SAT problems:

```bash
python SatSolver.py
```

This will:
1. Run DPLL on all benchmark problems
2. Run Simulated Annealing with 10 test runs per problem
3. Generate CSV files with results

### Input Format

SAT problems are provided in DIMACS CNF format:

```
c This is a comment
p cnf 3 3
1 2 0
-1 3 0
-2 -3 0
```

- `p cnf [variables] [clauses]`: Problem line specifying number of variables and clauses
- Each line ending in `0` represents a clause
- Positive integers are positive literals, negative integers are negative literals

### Custom Usage

```python
from SatSolver import *

# Parse a CNF file
numVariables, clauses = parseFile("problem.cnf")

# Solve with DPLL
satisfiable = DPLL(clauses, numVariables)
print(f"Satisfiable: {satisfiable}")

# Solve with Simulated Annealing
posOccurrences, negOccurrences = buildOccurrenceLists(clauses, numVariables)
sa = simulatedAnnealing(
    clauses=clauses,
    numVariables=numVariables,
    posOccurrences=posOccurrences,
    negOccurrences=negOccurrences,
    maxSteps=10000,
    initialTemp=100.0,
    coolingRate=0.95
)
solution = sa.solve()
numSatisfied, _, _ = fitness(clauses, solution)
print(f"Satisfied {numSatisfied}/{len(clauses)} clauses")
```

## Project Structure

```
SatSolver/
├── SatSolver.py           # Main implementation
├── README.md              # This file
├── LICENSE                # License file
└── PA3_Benchmarks/        # Benchmark CNF problems
    ├── CNF Formulas/      # Regular difficulty problems
    └── HARD CNF Formulas/ # Hard difficulty problems
```

## Algorithm Performance

Performance varies significantly based on problem characteristics:

### DPLL
| Problem Size | Difficulty | Typical Time |
|--------------|-----------|--------------|
| < 50 vars    | Regular   | < 0.1s       |
| 50-100 vars  | Regular   | 0.1s - 2s    |
| 100+ vars    | Hard      | 2s - 60s+    |

### Simulated Annealing
| Problem Size | Difficulty | Typical Time | Solution Quality |
|--------------|-----------|--------------|------------------|
| < 100 vars   | Regular   | < 0.5s       | 95-100%          |
| 100-500 vars | Regular   | 0.5s - 3s    | 90-100%          |
| 500+ vars    | Hard      | 3s - 10s     | 85-95%           |

*Note: DPLL guarantees optimal solutions, while Simulated Annealing may not always find perfect solutions but runs faster on large instances.*

## Key Functions

### Parsing and Evaluation
- `parseFile(filePath)`: Parse DIMACS CNF format file
- `fitness(clauses, assignment)`: Evaluate number of satisfied clauses
- `buildOccurrenceLists(clauses, numVariables)`: Precompute clause-variable relationships

### DPLL Algorithm
- `DPLL(clauses, numVariables)`: Main DPLL solver
- `dpllRecursive(clauses, assignment)`: Recursive DPLL with unit propagation and pure literal elimination
- `findUnitClause(clauses)`: Find clauses with single literal
- `findPureLiteral(clauses)`: Find variables appearing with only one polarity
- `chooseLiteral(clauses)`: Select branching variable using Jeroslow-Wang heuristic
- `simplifyClauses(clauses, literal)`: Simplify formula given an assignment

### Simulated Annealing
- `simulatedAnnealing`: Class implementing the SA algorithm
- `solve()`: Run the simulated annealing process
- `flipVariable()`: Efficiently update state after flipping a variable
- `fitnessDelta()`: Compute fitness change without full re-evaluation

### Testing
- `testDPLL()`: Run DPLL on benchmark problems
- `testSimulatedAnnealing()`: Run SA on benchmark problems with multiple trials

## Output

Both testing functions generate CSV files with results:

**DPLLResults.csv:**
- Problem name
- Difficulty tier
- SAT result (True/False)
- CPU time

**SimulatedAnnealingResults.csv:**
- Problem name
- Difficulty tier
- Run number
- CPU time
- Number of satisfied clauses
- Total clauses
- SAT label (SAT/NOTSAT)

## Applications of SAT Solving

SAT solvers have numerous real-world applications:

- **Hardware Verification**: Verifying correctness of circuit designs
- **Software Verification**: Proving program properties and finding bugs
- **Planning and Scheduling**: Solving constraint satisfaction problems
- **Cryptography**: Breaking encryption schemes and analyzing protocols
- **Bioinformatics**: Phylogenetic tree reconstruction and gene regulation analysis
- **Configuration Management**: Finding valid system configurations