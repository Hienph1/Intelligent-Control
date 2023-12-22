import numpy as np
import random
import math

# Function to maximize
# Fitness
def func(x):
  return 6.8 + x*np.sin(8*np.pi*x)

# Genetic algorithm parameters
pop_size = 60
breed_prob = 0.28
mutate_prob = 0.01
# Range of x
bounds = (-1,3)
# Accuracy desired
accuracy = 0.0001
length_gen = int(math.ceil(math.log2((bounds[1] - bounds[0]) / accuracy)))

# Create population
def create_population(pop_size = pop_size, length_gen = length_gen):
  pop = []
  for idx in range(pop_size):
    binary_array = [random.randint(0,1) for j in range(length_gen)]
    pop.append(binary_array)
  pop = np.array(pop)
  return pop

# Breeding at a random point
def crossover(p1, p2):
	crossover_point = random.randint(0, length_gen-1)
	child1 = np.zeros(length_gen)
	child2 = np.zeros(length_gen)
	child1[:crossover_point] = p1[:crossover_point]; child1[crossover_point:] = p2[crossover_point:]
	child2[:crossover_point] = p2[:crossover_point]; child2[crossover_point:] = p1[crossover_point:]
	return child1, child2

# Roulette Wheel Selection
def select_samples_rws(parents, percent = 0.28):

  total_fitness = len(parents)
  fitness = [1] * total_fitness

  selected = []
  num_to_select = int(total_fitness * percent)

  for _ in range(num_to_select):
    index = random.randint(0, total_fitness-1)
    selected.append(parents[index])

  return selected

# Selection random 1 pair
def get_random_numbers(max_value):

  num1 = random.randint(0, max_value)
  while True:
    num2 = random.randint(0, max_value)
    if num2 != num1:
      break

  return (num1, num2)

def convert_to_float(pop):
  float_pop = []

  for gen in pop:
    decimal = 0
    for bit in gen:
      decimal = decimal * 2 + bit
    float_num = decimal / 65535 * 4 - 1
    float_pop.append(float_num)

  return float_pop

def Breeding(selected_parents):
  children = []

  while len(children) < len(selected_parents) * 2:

    # Select scalar values from parents
    idx1, idx2 = get_random_numbers(len(selected_parents) - 1)
    child1, child2 = crossover(selected_parents[idx1], selected_parents[idx2])

    children.append(child1)
    children.append(child2)
  return children

# Run genetic algorithm
# Create new population
pop = create_population()
best_score_prev = 0

for iter in range(50):

  # Select parents
  selected_parents = select_samples_rws(pop, 0.28)

  # Breeding
  children = Breeding(selected_parents)

  new_pop = np.concatenate((pop, children), axis = 0)

  for i in range(len(new_pop)):
    if random.random() < mutate_prob:
      idx = random.randint(0, length_gen-1)
      new_pop[i][idx] = 1 - int(new_pop[i][idx])

  float_pop = convert_to_float(new_pop)
  fitness = [func(x) for x in float_pop]

  # Evolution
  the_chosen = new_pop[np.argmax(fitness)]

  evo_pop = [the_chosen]

  for i in range(len(new_pop) - 1):
    random_index = random.randint(0, len(new_pop) - 1)
    evo_pop.append(new_pop[random_index])
    # new_pop.remove(new_pop[random_index])

  pop = evo_pop

  # Check convergence
  best_score = max(fitness)

  print(f'Iteration {iter}: Best = {best_score}')

  if iter > 10:
    if abs(best_score_prev - best_score) < 0.0001:
      print('Converged!')
      break

  best_score_prev = best_score

print(f'Maximum is {best_score} at {float_pop[fitness.index(best_score)]}')
