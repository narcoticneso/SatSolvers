import random
import math
def parseFile(filePath): # parsing the file
  numVariables = 0
  clauses = []
  currentClause = []
  with open(filePath, 'r') as file: # read mode
    for line in file:
      line = line.strip() # remove whitespace
      if not line or line.startswith('c') or line.startswith('%'):
        continue
      if line.startswith('p'):
        parts = line.split() # makes each part a word
        numVariables = int(parts[2]) # the third word is the amount of variables
        continue
      numbers = line.split()
      for num in numbers:
        if num == '0': # means the end of the line
          if currentClause:
            clauses.append(currentClause)
            currentClause = []
        else: # not 0 so add to claause
          currentClause.append(int(num))
  return numVariables, clauses

def buildOccurrenceLists(clauses, numVariables):
  posOccurrences = [[] for _ in range(numVariables)] # list of lists for positive occurrences
  negOccurrences = [[] for _ in range(numVariables)] # list of lists for negative occurrences
  for clauseIndex, clause in enumerate(clauses): # enumerate to get index and clause
    for literal in clause:
      varIndex = abs(literal) - 1 # variable index (0-based)
      if literal > 0:
        posOccurrences[varIndex].append(clauseIndex) # add clause index to positive occurrences
      else:
        negOccurrences[varIndex].append(clauseIndex) # add clause index to negative occurrences
  return posOccurrences, negOccurrences

def createRandomAssignment(numVariables):
  return [random.choice([True, False]) for _ in range(numVariables)] # random true/false assignment

def fitness(clauses, assignment):
    numTrueLiterals = [] # number of true literals in each clause
    unsatisfiedClauses = set() # set of unsatisfied clause indices
    numOfSatisfiedClauses = 0

    for clauseIndex, clause in enumerate(clauses):
        trueCount = 0
        for literal in clause:
            varIndex = abs(literal) - 1 # 0 based
            if literal > 0 and assignment[varIndex]: # positive literal
                trueCount += 1
            elif literal < 0 and not assignment[varIndex]: # negative literal
                trueCount += 1

        numTrueLiterals.append(trueCount)
        if trueCount == 0:
            unsatisfiedClauses.add(clauseIndex) # add index of unsatisfied clause
        else:
            numOfSatisfiedClauses += 1
    return numOfSatisfiedClauses, numTrueLiterals, unsatisfiedClauses

def flipVariable(varIndex, assignment, numTrueLiterals, unsatisfiedClauses, numSatisfied, posOccurrences, negOccurrences):
    oldValue = assignment[varIndex] # store old value
    assignment[varIndex] = not oldValue # flip the variable
    if oldValue: # was True, now False
        clausesGainingTrue = negOccurrences[varIndex] # clauses that gain a true literal
        clausesLosingTrue = posOccurrences[varIndex] # clauses that lose a true literal
    else: # was False, now True
        clausesGainingTrue = posOccurrences[varIndex]
        clausesLosingTrue = negOccurrences[varIndex]

    # Update clauses gaining a true literal
    for clauseIndex in clausesLosingTrue:
        numTrueLiterals[clauseIndex] -= 1
        if numTrueLiterals[clauseIndex] == 0: # check if clause became unsatisfied
            unsatisfiedClauses.add(clauseIndex) # mark as unsatisfied
            numSatisfied -= 1 # decrement satisfied count
    for clauseIndex in clausesGainingTrue: # clause gains a true literal
        if numTrueLiterals[clauseIndex] == 0: # was unsatisfied
            unsatisfiedClauses.discard(clauseIndex) # now satisfied
            numSatisfied += 1 # increment satisfied count
        numTrueLiterals[clauseIndex] += 1 # increment true literal count
    return numSatisfied

def fitnessDelta(varIndex, assignment, numTrueLiterals, posOccurrences, negOccurrences):
    '''
    posDelta means flip impoves fitness
    negDelta means flip worsens fitness
    zeroDelta means flip has no effect on fitness
    '''
    if assignment[varIndex]: # currently True, flipping to False
       clausesGainingTrue = negOccurrences[varIndex] # clauses that gain a true literal
       clausesLosingTrue = posOccurrences[varIndex] # clauses that lose a true literal
    else: # currently False, flipping to True
        clausesGainingTrue = posOccurrences[varIndex]
        clausesLosingTrue = negOccurrences[varIndex]


    #counting unsatified clauses with changes
    numBreakingClauses = 0

    for clauseIndex in clausesLosingTrue:
        if numTrueLiterals[clauseIndex] == 1: # would become unsatisfied since it has only one true literal
            numBreakingClauses += 1
    numFixingClauses = 0 # clauses that would become satisfied
    for clauseIndex in clausesGainingTrue:
        if numTrueLiterals[clauseIndex] == 0: # would become satisfied
            numFixingClauses += 1
    return numFixingClauses - numBreakingClauses

# Genetic Algorithm
class geneticAlgorithm:
  # parameters
  populationSize = 200
  evolveGen = 20 # no. of generations to evolve (run length)
  crossoverRate = 0.8
  mutationRate = 0.02

  def __init__(self, formula, numVariables):  # called immediately
    self.formula = formula
    self.numVariables = numVariables
# GENERATE SOLUTIONS
  def initializePopulation(self):
    population = []
    for i in range(self.populationSize):
      population.append(createRandomAssignment(self.numVariables))    # creates initial population with random assignments
    return population

    # CROSSOVER
  def parentSelector(self, population, fitnesses):  # roulette wheel for selecting parents based on fitness
    totalFitness = sum(fitnesses)

    if totalFitness == 0:                     # edge case
      return random.choice(population)

    # number line
    cumulativeProbs = []
    cumulativeSum = 0
    for fitness in fitnesses:
      cumulativeSum += fitness / totalFitness
      cumulativeProbs.append(cumulativeSum)

    spin = random.random()

    for i,prob in enumerate(cumulativeProbs):   # find which segment the spin landed on
      if spin <= prob:
        return population[i]
    return population[-1]   # SHOULD NOT REACH HERE... just in case

  def crossover(self, parent1, parent2):  # randomly select crossover point and swap genetic material
    crossoverPoint = random.randint(1, self.numVariables - 1)
    # :crossover point -> traits up to this index ; crossoverPoint: traits after this index
    offspring1 = parent1[:crossoverPoint] + parent2[crossoverPoint:]
    offspring2 = parent2[:crossoverPoint] + parent1[crossoverPoint:]
    return offspring1, offspring2
  # MUTATE
  def mutate(self, assignment):     # randomly creating mutations by flipping variables based on mutationRate
    for i in range(len(assignment)):
      # flip selected variable with probability mutationRate
      if random.random() < self.mutationRate:     # random floating point number comparison
        assignment[i] = not assignment[i]

  # KILL roulette wheel
  def calculateKill(self, fitnesses):     # fitnesses: list of fitness values
  # probability of death is inversely proportionate to fitness (survival of the fittest)
    killProbabilities = []
    for fitness in fitnesses:
      if fitness == 0:
        killProbabilities.append(1.0)   # low fitness high kill chance
      else:
        killProbabilities.append(1.0/ fitness)
    return killProbabilities

  def whichToPick(self, population, fitnesses):
    killProbabilities = self.calculateKill(fitnesses)
    totalWeight = sum(killProbabilities)
    if totalWeight == 0:
      return random.randint(0, len(population)-1)
    cumulativeProb = []
    cumulativeSum = 0
    for prob in killProbabilities:
      cumulativeSum += prob / totalWeight
      cumulativeProb.append(cumulativeSum)
    spin = random.random()        # spin weighted wheel
    for i, prob in enumerate(cumulativeProb):   # find which to pick
      if spin <= prob:
        return i      # return index of individual

  def purgeNight(self, population, fitnesses):
    while(len(population) > self.populationSize):   # keep going until population is back to original
      indexToKill = self.whichToPick(population, fitnesses)
      population.pop(indexToKill)       # remove chosen one
      fitnesses.pop(indexToKill)

    return population

  def solve(self):
    population = self.initializePopulation()    # random population

    bestAssignment = None
    bestFitness = 0
    totalClauses = len(self.formula)

    for generation in range(self.evolveGen):    # loop through every generation
      fitnesses = []
      for assignment in population:       # find fitnesses for all individuals
        numSatisfied, _, _ = fitness(self.formula, assignment)
        fitnesses.append(numSatisfied)

      maxFitness = fitnesses.index(max(fitnesses))    # this uses index
      if fitnesses[maxFitness] > bestFitness:
        bestFitness = fitnesses[maxFitness]
        bestAssignment = population[maxFitness][:]    # copies the list (independent copy)

      if bestFitness == totalClauses:       # check if we've found perfect soln
        break
      offspring = []
      for i in range(self.populationSize // 2):     # create offspring
        parent1 = self.parentSelector(population, fitnesses)
        parent2 = self.parentSelector(population, fitnesses)
        child1,child2 = self.crossover(parent1, parent2)    # returns (offspring1, offspring2) a tuple with two lists
        offspring.extend([child1,child2])   # adds both lists to the offspring list

      for child in offspring:
        self.mutate(child)
      combinedPop = population + offspring
      combinedFitness = []
      for assignment in combinedPop:        # calculate fitnesses for new pop
        numSatisfied, _, _ = fitness(self.formula, assignment)
        combinedFitness.append(numSatisfied)
      population = self.purgeNight(combinedPop, combinedFitness) # purge the weak based on new fitnesses

    return bestAssignment

def simmulatedAnnealing(clauses, numVariables, initialTemp=10, coolingRate=0.95, stepsPerTemp=1000,minTemp=1e-3):
    posOccurrences, negOccurrences = buildOccurrenceLists(clauses, numVariables) # O(1) clause lookup
    totalClauses = len(clauses)

    # current <- makeNode(problem.initialState)
    currentAssignment = createRandomAssignment(numVariables) # random initial assignment
    numSatisfied, numTrueLiterals, unsatisfiedClauses = fitness(clauses, currentAssignment)

    bestAssignment = currentAssignment.copy() # best soltuion
    bestSatisfied = numSatisfied # best solution fitness

    currentTemp = initialTemp # temperature is the probability of accepting worse solutions

    while currentTemp > minTemp:
       for step in range(stepsPerTemp): # inner loop for each temperature
        if numSatisfied == totalClauses:
            return currentAssignment # found satisfying assignment
        if not unsatisfiedClauses: # no unsatisfied clauses sanity check
            return currentAssignment

        clauseIndex = random.choice(list(unsatisfiedClauses)) # pick a random unsatisfied clause
        literal = random.choice(clauses[clauseIndex]) # pick a random literal from that clause
        varIndex = abs(literal) - 1 # variabel index (0-based) to flip

        # delta e is next.value - current value
        deltaE = fitnessDelta(varIndex, currentAssignment, numTrueLiterals, posOccurrences, negOccurrences)
        if deltaE > 0: # flipping improves fitness / hill climbing
            numSatisfied = flipVariable(varIndex, currentAssignment, numTrueLiterals, unsatisfiedClauses, numSatisfied, posOccurrences, negOccurrences)
            if numSatisfied > bestSatisfied: # update best solution
                bestSatisfied = numSatisfied
                bestAssignment = currentAssignment.copy()
        else: # flipping worsens or does not change fitness , use  e^(deltaE/temp) to determine if we should accept the worse solution
            acceptanceProb = math.exp(deltaE / currentTemp) # acceptance probability
            spin = random.random() # random float [0.0, 1.0)
            if spin < acceptanceProb: # accept worse solution with some probability
                numSatisfied = flipVariable(varIndex, currentAssignment, numTrueLiterals, unsatisfiedClauses, numSatisfied, posOccurrences, negOccurrences)
            #else reject the worse solution and do nothing

       currentTemp *= coolingRate # cool down / T<- schedule(t)  and geometric cooling
    return bestAssignment # return best found assignment even if not satisfying

def testSimmulatedAnnealing(FoldertoUse = "PA3_Benchmarks", numTests = 10):
    '''
    Test the simmulated annealing multiple times and collects CPU time and number of satisfied clauses
    '''

    import time
    import os
    import csv


    regularFolder = os.path.join(FoldertoUse, "CNF Formulas")
    hardFolder = os.path.join(FoldertoUse, "HARD CNF Formulas")

    allFiles = []

    if os.path.exists(regularFolder):
        for file in os.listdir(regularFolder):
            if file.endswith(".cnf") and "rcnf" not in file.lower(): # only .cnf files and not ones without comments
                filePath = os.path.join(regularFolder, file) # full path
                allFiles.append(("REGULAR", filePath)) # prepend REGULAR or HARD to distinguish later

    if os.path.exists(hardFolder):
        for file in os.listdir(hardFolder):
            if file.endswith(".cnf") and "rcnf" not in file.lower(): # only .cnf files and not ones without comments
                filePath = os.path.join(hardFolder, file)
                allFiles.append(("HARD", filePath))


    print(f"There are {len(allFiles)} files to test in {FoldertoUse}")

    with open("SA_Results.csv", mode='w', newline='') as csvfile:
       writer = csv.writer(csvfile)
       writer.writerow(["Problem","Tier","Run","CPU Time", "bestSatisifiedClauses", "Total Clauses","SAT"])


       for tier, filePath in allFiles:
            numVariables, clauses = parseFile(filePath)
            buildOccurrenceLists(clauses, numVariables) # precompute occurrence lists
            totalClauses = len(clauses)
            print(f"Testing file: {filePath} ({tier})")
            for testRun in range(numTests):
                startTime = time.process_time() # CPU time
                resultAssignment = simmulatedAnnealing(clauses, numVariables)
                endTime = time.process_time()
                cpuTime = endTime - startTime
                BestNumSatisifed, _, _ = fitness(clauses, resultAssignment)
                satLabel = "SAT" if BestNumSatisifed == totalClauses else "NOTSAT"
                print(f"Run {testRun}: CPU Time: {cpuTime:.4f} seconds, Satisfied Clauses: {BestNumSatisifed}/{totalClauses}")
                writer.writerow([os.path.basename(filePath), tier, testRun, f"{cpuTime:.4f}", BestNumSatisifed, totalClauses,satLabel])


def testGeneticAlgorithm(FoldertoUse = "PA3_Benchmarks", numTests = 10):
    '''
    Test the genetic algorithm multiple times and collects CPU time and number of satisfied clauses
    '''

    import time
    import os
    import csv


    regularFolder = os.path.join(FoldertoUse, "CNF Formulas")
    hardFolder = os.path.join(FoldertoUse, "HARD CNF Formulas")

    allFiles = []
    '''
    if os.path.exists(regularFolder):
        for file in os.listdir(regularFolder):
            if file.endswith(".cnf") and "rcnf" not in file.lower(): # only .cnf files and not ones without comments
                filePath = os.path.join(regularFolder, file) # full path
                allFiles.append(("REGULAR", filePath)) # prepend REGULAR or HARD to distinguish later
    '''
    if os.path.exists(hardFolder):
        for file in os.listdir(hardFolder):
            if file.endswith(".cnf") and "rcnf" not in file.lower(): # only .cnf files and not ones without comments
                filePath = os.path.join(hardFolder, file)
                allFiles.append(("HARD", filePath))


    print(f"There are {len(allFiles)} files to test in {FoldertoUse}")

    with open("GA_Results.csv", mode='w', newline='') as csvfile:
       writer = csv.writer(csvfile)
       writer.writerow(["Problem","Tier","Run","CPU Time", "bestSatisifiedClauses", "Total Clauses","SAT"])


       for tier, filePath in allFiles:
            numVariables, clauses = parseFile(filePath)
            buildOccurrenceLists(clauses, numVariables) # precompute occurrence lists
            totalClauses = len(clauses)
            print(f"Testing file: {filePath} ({tier})")
            for testRun in range(numTests):
                startTime = time.process_time() # CPU time
                ga = geneticAlgorithm(clauses, numVariables)
                resultAssignment = ga.solve()
                endTime = time.process_time()
                cpuTime = endTime - startTime
                BestNumSatisifed, _, _ = fitness(clauses, resultAssignment)
                satLabel = "SAT" if BestNumSatisifed == totalClauses else "NOTSAT"
                print(f"Run {testRun}: CPU Time: {cpuTime:.4f} seconds, Satisfied Clauses: {BestNumSatisifed}/{totalClauses}")
                writer.writerow([os.path.basename(filePath), tier, testRun, f"{cpuTime:.4f}", BestNumSatisifed, totalClauses,satLabel])

def DPLL (clauses, numVariables):
  clausesCopy = [clause[:] for clause in clauses]
  satisfiable,_ = dpllRecursive(clausesCopy, {})
  return satisfiable

def dpllRecursive(clauses, assignment):
  # uses current CNF formula and disctionary mapping
    while True: # apply unit propigation and pure literal elimination
        # unit propoigation is finding a clause with one literal
        unitLiteral = findUnitClause(clauses)
        if unitLiteral is not None:
            varNum = abs(unitLiteral) # variable number (1-based)
            value = unitLiteral > 0 # True if positive literal and False when negative
            assignment[varNum] = value # update assignment

            clauses = simplifyClauses(clauses, unitLiteral) # simplify clauses
            if [] in clauses: # conflict detected
                return False, {} # unsatisfiable
            continue # continue to apply unit propigation until no more unit clauses

        # pure literal elimination is finding a literal that appears only positively or only negatively
        pureLiteral = findPureLiteral(clauses)
        if pureLiteral is not None:
            varNum = abs(pureLiteral)
            value = pureLiteral > 0
            assignment[varNum] = value # update the assignment

            clauses = simplifyClauses(clauses, pureLiteral) # simplify clauses
            if [] in clauses: # conflict detected
                return False, {} # unsatisfiable
            continue # continue to apply pure literal elimination until no more pure literals
        break # no more unit clauses or pure literals
        # check base cases
    if not clauses: # all clauses satisfied
     return True, assignment
    # empty clause conflict
    if [] in clauses: # empty clause found
      return False, assignment # unsatisfiable

    # choose a variable to branch on
    literal = chooseLiteral(clauses)
    varNum = abs(literal)

    # try assigning True
    simplifiedClauses = simplifyClauses(clauses, literal) # simplify clauses with literal set to True
    if [] not in simplifiedClauses: # no conflict
        newAssignment = assignment.copy()
        newAssignment[varNum] = (literal > 0) # assign True
        satisfiable, finalAssignment = dpllRecursive(simplifiedClauses, newAssignment)
        if satisfiable:
            return True, finalAssignment # found satisfying assignment
    # try assigning False
    simplifiedClauses = simplifyClauses(clauses, -literal) # simplify clauses with literal set to False
    if [] not in simplifiedClauses: # no conflict
        newAssignment = assignment.copy() #  make a copy of the assignment
        newAssignment[varNum] = (literal < 0) # assign False
        satisfiable, finalAssignment = dpllRecursive(simplifiedClauses, newAssignment)
        if satisfiable:
            return True, finalAssignment # found satisfying assignment
    return False, assignment # unsatisfiable

def simplifyClauses(clauses, literal):
    # remove any clause satisfied by the literal and remove negated literal from other clauses
    # if an empty clause is created, return None to indicate conflict
    newClauses = []
    negLiteral = -literal
    for clause in clauses:
        if literal in clause: # clause satisfied
            continue # skip this clause
        if negLiteral in clause: # remove negated literal from clause
            newClause = []
            for lit in clause:
                if lit != negLiteral:
                    newClause.append(lit)
            if not newClause: # empty clause created
                return [[]] # conflict detected
            newClauses.append(newClause)
        else:
            newClauses.append(clause) # keep clause unchanged
    return newClauses

def findUnitClause(clauses):
    for clause in clauses:
      if len(clause) == 1: # unit clause found
        return clause[0] # return the single literal
    return None # no unit clause found

def findPureLiteral(clauses): # pure literal is one that appears with only one polarity
    allLiterals = set()
    for clause in clauses:
        for literal in clause:
          allLiterals.add(literal) # add literal to the set
    for literal in allLiterals:
        if -literal not in allLiterals: # pure literal found
            return literal
    return None # no pure literal found

def chooseLiteral(clauses):
   # use Jeroslow-Wang heuristic to choose literal
   # It bascially gives higher weight to literals in shorter clauses and picks the one with the highest score
    literalScores = {}
    for clause in clauses:
        weight = 2 ** (-len(clause)) # weight based on clause length
        for literal in clause:
            literalScores[literal] = literalScores.get(literal, 0) + weight # accumulate score
    # choose literal with highest score
    if literalScores:
        return max(literalScores, key=literalScores.get) # literal with max score

import time
def testDPLL(FoldertoUse = "PA3_Benchmarks"):
    '''
    Test the DPLL and collects CPU time and number of satisfied clauses
    '''
    import os
    import csv

    regularFolder = os.path.join(FoldertoUse, "CNF Formulas")
    hardFolder = os.path.join(FoldertoUse, "HARD CNF Formulas")

    allFiles = []

    if os.path.exists(regularFolder):
        for file in os.listdir(regularFolder):
            if file.endswith(".cnf") and "rcnf" not in file.lower(): # only .cnf files and not ones without comments
                filePath = os.path.join(regularFolder, file) # full path
                allFiles.append(("REGULAR", filePath)) # prepend REGULAR or HARD to distinguish later

    if os.path.exists(hardFolder):
        for file in os.listdir(hardFolder):
            if file.endswith(".cnf") and "rcnf" not in file.lower(): # only .cnf files and not ones without comments
                filePath = os.path.join(hardFolder, file)
                allFiles.append(("HARD", filePath))


    print(f"There are {len(allFiles)} files to test in {FoldertoUse}")
    with open("DPLLResults.csv", mode='w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["Problem","Tier","SAT","CPUtime"])
      for tier, filePath in allFiles:
          numVariables, clauses = parseFile(filePath)
          startTime = time.process_time()
          sat = DPLL(clauses, numVariables)
          endTime = time.process_time()
          cpuTime = endTime - startTime
          writer.writerow([os.path.basename(filePath), tier, sat,cpuTime])
          print(f"File: {filePath} ({tier}), SAT: {sat}, Time: {cpuTime}")

if __name__ == "__main__":


    testDPLL(FoldertoUse="PA3_Benchmarks")

    testSimmulatedAnnealing(
            FoldertoUse="PA3_Benchmarks",
            numTests=10,
        )



 