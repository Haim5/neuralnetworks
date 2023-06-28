import random
import sys
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dot_product(weights, seq):
    print(weights)
    print(seq)
    if len(weights) != len(seq):
        raise ValueError("Lengths of weights and seq must match.")
    result = 0
    for w, x in zip(weights, seq):
        result += w * x
    return result

def select_value(n1, n2):
    if random.randint(0,1) == 1:
        return n1
    return n2

def avg_value(n1, n2):
    return (n1 + n2) / 2

def fit_sort(a):
    return a.get_fitness()

class Network:
    # def __init__(self, e=None):
    #     self.__fitness = None
    #     layers = [16, 2, 1]
    #     if e:
    #         self.__edges = e
    #     else:
    #         edges = []
    #         for i in layers[:-2]:
    #             net = []
    #             end = int(i/2)
    #             for j in range(end):
    #                 col = []
    #                 for k in range(i):
    #                     col.append(random.uniform(-1, 1))
    #                 net.append(col)
    #             edges.append(net)
    #         edges.append([[random.uniform(-1, 1), random.uniform(-1, 1)]])
    #         self.__edges = edges
    #     self.__layers = layers
    #     self.print_weights()

    def __init__(self, e=None):
        self.num_nodes = [17, 2, 1]
        random.seed(a=None, version=2)
        if e:
            self.__edges = e
        else:
            self.__edges = self.initialize_weights()
        ##self.__layers = layers
        self.__fitness = None
        self.__layers = self.num_nodes
        # self.print_weights()

    def initialize_weights(self):
        weights = []
        for i in range(len(self.num_nodes) - 2):
            layer_weights = self.generate_weights(self.num_nodes[i], self.num_nodes[i + 1])
            weights.append(layer_weights)
        weights.append([[1], [-1], [1.5]])
        return weights

    @staticmethod
    def generate_weights(input_nodes, output_nodes):
        weights = []
        
        for _ in range(input_nodes):
            layer_weights = []
            w = random.uniform(-1, 1)
            for _ in range(output_nodes):
                layer_weights.append(w)
                w = -w
            # layer_weights = [(w,-w) for _ in range(output_nodes)]
            weights.append(layer_weights)
        return weights
    
    def print_weights(self):
        for i, layer_weights in enumerate(self.__edges):
            print(f"Weights for layer {i + 1}:")
            for weights in layer_weights:
                print(weights)
            print()


    # def predict(self, seq):
    #     output = [int(c) for c in seq]
    #     for i, layer_weights in enumerate(self.__edges):
    #         layer_output = []
    #         for weights in layer_weights:
    #             neuron_output = sum(w * x for w, x in zip(weights, output))
    #             neuron_output = sigmoid(neuron_output)
    #             layer_output.append(neuron_output)
    #         output = layer_output
    #     print(output)
    #     return output

            
    def predict_layer(self, seq, layer, f=sigmoid):
        end = len(layer)
        result = [0.0] * len(layer[0])
        for j in range(len(layer[0])):
            for i in range(end):
                try:
                    result[j] += layer[i][j] * seq[i]
                except TypeError:
                    print(layer)
                    print("ERROR")
            result[j] = f(result[j])
        return result


    def predict(self, values):
        y = sum(self.__layers)
        for k in range(y):
            e = self.__edges[k]
            for j in range(y):
                values[j]
                e[j]
                values[k]
                values[j] += e[j] * values[k]
        if values[-1] > 2:
            return 1
        return 0
    
    def crossover(self, other, f):
        # layer_index = random.randint(1, int(len(self.__layers) - 2))
        # next_edges = self.__edges[:layer_index]
        # next_edges += other.get_edges()[layer_index:]
        # return Network(e=next_edges)
        next_edges = []
        for i in range(int(sum(self.__layers))):
            t1 = []
            for j in range(int(sum(self.__layers))):
                t1.append(f(other.get_edges()[i][j], self.__edges[i][j])) 
            next_edges.append(t1)
        return Network(e=next_edges)
        
        

    def get_edges(self):
        return self.__edges
    
    def get_fitness(self):
        if(self.__fitness):
            return self.__fitness
        else:
            return 0
    
    ## make a mutation
    def mutate(self):
        x = random.randint(1, 3)
        for _ in range(x):
            i = random.randint(0, int(len(self.__edges))-1)
            e = self.__edges[i]
            j = random.randint(0, int(len(e))-1)
            e[j] = random.uniform(-1, 1)

    ## calculate fitness score
    def fitness(self, test : dict(), override=False):
        if self.__fitness and not override:
            return self.__fitness
        correct = 0
        for key, tag in test.items():
            v = [float(s) for s in key]
            v.append(0.0)
            v.append(0.0)
            v.append(0.0)
            if int(tag) == self.predict(values=v):
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
    stay_odds = 1
    ## odds for a mutation
    mu_odds = 0.05
    next_gen = []
    for p in solutions:
        children = random.randint(1,2)
        sol1 = p[0]
        sol2 = p[1]
        for _ in range(0, children):
            ## create new
            off1 = sol2.crossover(sol1, f=select_value)
            off2 = sol1.crossover(sol2, f=avg_value)

            if random.random() <= mu_odds:
                    ## mutation
                off1.mutate()
            if random.random() <= mu_odds:
                    ## mutation
                off2.mutate()

            next_gen.append(off1)
            next_gen.append(off2)

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

def genetic(practice, population_size=100, max_gen=800, max_con=12):
    solutions = [Network() for _ in range(population_size)]
    best_sol = None
    best_score = -1
    gen = 0
    con = 0
    slice_size = 0

    while gen < max_gen and con < max_con and len(solutions) > slice_size:
        if len(solutions) > 10000:
            slice_size = 3
        else:
            slice_size = 2
        ## mix
        random.shuffle(solutions)
        ## keep for next generation
        solutions.sort(key=fit_sort)
        end = int(0.15 * len(solutions))
        elite = solutions[-20:]
        elite_pairs = pair_solutions(elite)
        elite_cross = crossover1(elite_pairs)
        solutions = [x for x in solutions if x not in elite]
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
        

        solutions = elite + elite_cross
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
    practice, test = parse("nn0.txt")
    print("Runing, please wait")
    ans = genetic(practice=practice)
    x = ans.fitness(test, override=True)
    print(x)

if __name__=="__main__":
    main()
