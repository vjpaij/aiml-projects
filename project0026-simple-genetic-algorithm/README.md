### Description:

A genetic algorithm (GA) is an optimization technique inspired by natural selection and genetics. It evolves a population of candidate solutions over generations using operators like selection, crossover, and mutation. In this project, we implement a basic GA to maximize a mathematical function — a classic optimization problem.

- Maximizes the function f(x) = x * sin(x) over the interval [0, 10]
- Evolves a population over multiple generations
- Applies selection, crossover, and mutation
- Tracks and prints the best solution each generation

## Genetic Algorithm to Maximize f(x) = x \* sin(x)

This script implements a basic Genetic Algorithm (GA) using NumPy to find the maximum of the function:

```
f(x) = x * sin(x)
```

within the range `x ∈ [0, 10]`.

### Code Explanation

```python
import numpy as np
```

* Imports NumPy for numerical operations.

```python
def fitness_function(x):
    return x * np.sin(x)
```

* Defines the **objective (fitness) function** we want to maximize.

### Parameters

```python
population_size = 10
```

* Number of individuals in each generation.

```python
generations = 30
```

* Total number of generations the algorithm will run.

```python
mutation_rate = 0.1
```

* Probability that an individual will mutate (undergo a small random change).

```python
x_bounds = (0, 10)
```

* Lower and upper bounds for values of `x`.

### Initial Population

```python
population = np.random.uniform(low=x_bounds[0], high=x_bounds[1], size=(population_size,))
```

* Randomly initializes a population of real-valued individuals in the range `[0, 10]`.

### Genetic Algorithm Main Loop

```python
for generation in range(generations):
```

* Runs the evolutionary loop for a fixed number of generations.

#### Step 1: Evaluate Fitness

```python
fitness = fitness_function(population)
```

* Calculates the fitness for each individual.

#### Step 2: Selection

```python
sorted_indices = np.argsort(fitness)[::-1]
parents = population[sorted_indices[:population_size // 2]]
```

* Selects the top 50% of individuals (by fitness) to be parents using tournament selection.

#### Step 3: Crossover

```python
offspring = []
while len(offspring) < population_size // 2:
    p1, p2 = np.random.choice(parents, 2, replace=False)
    child = (p1 + p2) / 2
    offspring.append(child)
```

* Creates offspring by averaging two randomly selected parents.

#### Step 4: Mutation

```python
for i in range(len(offspring)):
    if np.random.rand() < mutation_rate:
        offspring[i] += np.random.normal(0, 0.5)
        offspring[i] = np.clip(offspring[i], *x_bounds)
```

* With a small probability, adds Gaussian noise to offspring to maintain diversity. Ensures values stay within bounds.

#### Step 5: Create New Population

```python
population = np.concatenate([parents, offspring])
```

* Combines parents and offspring to form the new generation.

#### Step 6: Print Best Solution So Far

```python
best_fitness = np.max(fitness_function(population))
best_individual = population[np.argmax(fitness_function(population))]
print(f"Generation {generation+1}: Best x = {best_individual:.4f}, f(x) = {best_fitness:.4f}")
```

* Tracks and displays the best individual and its fitness at each generation.

### Final Output

```python
print("\nFinal Best Solution:")
print(f"  x = {best_individual:.4f}")
print(f"  f(x) = {best_fitness:.4f}")
```

* Displays the best solution found after all generations.

---

This Genetic Algorithm is a simple yet effective technique for optimization problems where derivative-based methods may not apply. You can tune parameters such as population size, mutation rate, and number of generations to improve performance depending on the complexity of the fitness landscape.
