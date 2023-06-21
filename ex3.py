import random
import sys

def sigmoid(x):
    return 1 / (1 + pow(2.71828, -x))

class Network:
    def __init__(self, e=None):
        self.__fitness = None
        layers = [16, 8, 4, 2, 1]
        if e:
            self.__edges = e
        else:
            edges = []
            for i in layers[:-2]:
                col = []
                for j in range(i):
                    col.append(random.uniform(-1, 1))
                net = []
                end = int(i/2)
                for j in range(end):
                    col_copy = col.copy()
                    col_copy[j] = 0
                    col_copy[(j + end)] = 0
                    net.append(col_copy)
                edges.append(net)
            edges.append([[random.uniform(-1, 1), random.uniform(-1, 1)]])
            self.__edges = edges
        self.__layers = layers

            
    def predict_layer(self, seq, layer, f=sigmoid):
        end = int(len(seq) / 2)
        result = [0.0] * end
        for i in range(end):
            for j in range(end + end):
                try:
                    result[i] += layer[i][j] * f(seq[j])
                except TypeError:
                    print(layer)
                    print("ERROR")
        return result


    def predict(self, seq, f=sigmoid):
        output = [float(x) for x in seq]
        for i in range(len(self.__layers) - 1):
            output = self.predict_layer(output, self.__edges[i], f=f)
        score = f(output[0])
        if score > 0.5:
            return 1
        return 0
    
    def crossover(self, other):
        layer_index = random.randint(1, int(len(self.__layers) - 2))
        next_edges = self.__edges[:layer_index]
        next_edges += other.get_edges()[layer_index:]
        return Network(e=next_edges)
        
        

    def get_edges(self):
        return self.__edges
    
    ## make a mutation
    def mutate(self):
        ## pick random cell
        layer_index = random.randint(0, int(len(self.__layers) - 2))
        layer = self.__edges[layer_index]
        edge_index = random.randint(0, int(len(layer) - 1))
        sub_edge = layer[edge_index] 
        sub_edge_index = random.randint(0, int(len(sub_edge) - 1))
        ## assign random value
        sub_edge[sub_edge_index] = random.uniform(-1, 1)
            

    ## calculate fitness score
    def fitness(self, test : dict(), override=False):
        if self.__fitness and not override:
            return self.__fitness
        correct = 0
        for key, tag in test.items():
            if int(tag) == self.predict(seq=key):
                correct += 1
        if len(test) > 0:
            self.__fitness = correct / len(test)
            return self.__fitness
        self.__fitness = 0
        return 0


    def print(self):
        print(self.__edges)


## split solutions to pairs
def pair_solutions(solutions):
    ans = []
    random.shuffle(solutions)
    size = len(solutions)
    last = int(size / 2)
    for i in range(0, last):
        if 2 * i + 1 >= len(solutions):
            ans.append((solutions[2 * i], solutions[2 * i]))
        else:
            ans.append((solutions[2 * i], solutions[2 * i + 1]))
    return ans


## random crossover
def crossover1(solutions):
    ## odds for a parent to stay for the next generation
    stay_odds = 0.35
    ## odds for a mutation
    mu_odds = 0.01
    next_gen = []
    for p in solutions:
        children = random.randint(1, 2)
        sol1 = p[0]
        sol2 = p[1]
        for _ in range(0, children):
            off1 = sol1.crossover(sol2)
            next_gen.append(off1)
            off2 = sol2.crossover(sol1)
            next_gen.append(off2)

            if random.random() <= mu_odds:
                    ## mutation
                off1.mutate()
            if random.random() <= mu_odds:
                    ## mutation
                off2.mutate()

        if random.random() <= stay_odds:
            next_gen.append(sol1)
        if random.random() <= stay_odds:
            next_gen.append(sol2)

    return next_gen

## select which solution to take to the crossover part
def select_next(options, practice):
    best = None
    score = float('-inf')
    for s in options:
        temp = s.fitness(practice)
        if temp > score:
            score = temp
            best = s
    return best

def genetic(practice, population_size=22, max_gen=800, max_con=12):
    solutions = [Network() for _ in range(population_size)]
    best_sol = None
    best_score = -1
    gen = 0
    con = 0
    slice_size = 0

    while gen < max_gen and con < max_con and len(solutions) > slice_size:
        if len(solutions) > 10000:
            slice_size = 4
        else:
            slice_size = 2
        ## mix
        random.shuffle(solutions)
        ## keep for next generation
        end = int(0.15 * len(solutions))
        elite = solutions[:end]
        solutions = solutions[end:]
        ## split to slices and choose who continues on
        solutions = [select_next(sub_solution, practice=practice) for sub_solution in [solutions[i:i+slice_size] for i in range(0, len(solutions), slice_size)]]



        ## make pairs
        if len(solutions) % 2 != 0 and len(solutions) > 1:
            random.shuffle(solutions)
            s1 = solutions.pop()
            s2 = solutions.pop()
            solutions.append(select_next([s1, s2], practice=practice))
        
        pairs = pair_solutions(solutions)
        
        ## crossover
        next_gen = crossover1(solutions=pairs)
        

        solutions = next_gen + elite
        ## find best
        next_sol = select_next(solutions, practice=practice)

        ## update best
        if next_sol is not None and next_sol.fitness(practice) > best_score:
            best_score = next_sol.fitness(practice)
            best_sol = next_sol
            con = 0
        else:
            con += 1
        gen += 1
        print(best_score)

    print("Best fit score: " + str(best_score))
    print("Generation: " + str(gen))
    return best_sol




def parse(f):
    practice = dict()
    test = dict()
    # Read the data from the original file
    with open(f, 'r') as file:
        data = file.readlines()

    # Shuffle the data randomly
    random.shuffle(data)

    # Determine the split index based on the desired percentage
    split_index = int(0.75 * len(data))

    # Split the data into two parts
    part1 = data[:split_index]
    part2 = data[split_index:]

    for line in part1:
        values = line.split()
        practice[values[0]] = values[1]

    for line in part2:
        values = line.split()
        test[values[0]] = values[1]

    return practice, test


def main():
    # if len(sys.argv) < 2:
    #     return
    # args = sys.argv[1]
    practice, test = parse("/home/haim/Desktop/ex3/nn0.txt")
    print("Now it runs, sit patiently dumbass")
    ans = genetic(practice=practice)
    x = ans.fitness(test, override=True)
    print(x)

if __name__=="__main__":
    main()
