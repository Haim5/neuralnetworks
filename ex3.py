import random
import sys

def sigmoid(x):
    return 1 / (1 + pow(2.71828, -x))

def select_value(n1, n2):
    if random.randint(0,1) == 1:
        return n1
    return n2

def avg_value(n1, n2):
    return (n1 + n2) / 2

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
                    col_copy[j+1] = 0
                    col_copy[(j + end)] = 0
                    col_copy[(j + end - 1)] = 0
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
                    ##print(layer)
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
    
    def crossover(self, other, f):
        next_edges = []
        for i in range(int(len(self.__edges))):
            t1 = []
            for j in range(int(len(self.__edges[i]))):
                t2 = []
                for k in range(int(len(self.__edges[i][j]))):
                    t2.append(f(other.get_edges()[i][j][k], self.__edges[i][j][k])) 
                t1.append(t2)
            next_edges.append(t1)
        return Network(e=next_edges)

        
        

    def get_edges(self):
        return self.__edges
    
    ## make a mutation
    def mutate(self):
        ## pick random cell
        times = random.randint(1,2)
        for _ in range(times):
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
    stay_odds = 0.45
    ## odds for a mutation
    mu_odds = 0.02
    next_gen = []
    for p in solutions:
        children = random.randint(1, 3)
        sol1 = p[0]
        sol2 = p[1]
        for _ in range(children):
            ## create new
            off1 = sol2.crossover(sol1, f=select_value)
            off2 = sol1.crossover(sol2, f=avg_value)
            
            ## mutation
            if random.random() <= mu_odds:
                off1.mutate()
            if random.random() <= mu_odds:
                off2.mutate()
            
            ## add
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

def genetic(practice, population_size=22, max_gen=800, max_con=12):
    solutions = [Network() for _ in range(population_size)]
    best_sol = None
    best_score = -1
    gen = 0
    con = 0
    slice_size = 0

    while gen < max_gen and con < max_con and len(solutions) > slice_size:
        if len(solutions) > 50:
            slice_size = 3
        else:
            slice_size = 2
        ## mix
        random.shuffle(solutions)
        ## keep for next generation
        end = int(0.1 * len(solutions))
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

    print("Best practice fit score: " + str(best_score))
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

    practice, test = parse("/home/haim/Desktop/ex3/nn0.txt")
    print("Now it runs, sit patiently dumbass")
    ans = genetic(practice=practice)
    ans.print()
    x = ans.fitness(test, override=True)
    print("Test result:" + str(x))

if __name__=="__main__":
    main()
