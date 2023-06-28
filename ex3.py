import random
import sys

def select_value(n1, n2):
    if random.randint(0,1) == 1:
        return n1
    return n2

def avg_value(n1, n2):
    return (n1 + n2) / 2


class Network:
    def __init__(self, e=None):
        self.num_nodes = [16, 2, 1]
        if not e:
            x1 = sum(self.num_nodes)
            edges = [[0.0] * x1 for _ in range(x1)]
            for i in range(self.num_nodes[0]):
                e = edges[i]
                x2 = random.uniform(0, 1)
                e[self.num_nodes[0]] = x2
                e[self.num_nodes[0] + 1] = -x2
            y = self.num_nodes[0]
            for i in range(y, y + self.num_nodes[1]):
                e = edges[i]
                e[self.num_nodes[0] + self.num_nodes[1]] = 1.5
            self.__edges = edges
        else:
            self.__edges = e
        self.__fitness = None
        self.__layers = self.num_nodes
      

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
    
    def crossover(self, other, f, mu_odds=0.01):
        next_edges = []
        for i in range(int(sum(self.__layers))):
            t1 = []
            for j in range(int(sum(self.__layers))):
                t1.append(f(other.get_edges()[i][j], self.__edges[i][j])) 
            next_edges.append(t1)
        off = Network(e=next_edges)
        if random.random() <= mu_odds:
            off.mutate()
        return off
        

    def get_edges(self):
        return self.__edges
    
    
    ## make a mutation
    def mutate(self):
        x = random.randint(1, 5)
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

    def parse_out(self):
        return '\n'.join([' '.join(map(str, sublist)) for sublist in self.__edges])
    


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
def crossover1(solutions, mu_odds=0.01, stay_odds=0.35):
    ## odds for a parent to stay for the next generation
    next_gen = []
    for p in solutions:
        children = random.randint(1, 3)
        sol1 = p[0]
        sol2 = p[1]

        next_gen.append(sol1.crossover(sol2, f=avg_value, mu_odds=mu_odds))
        next_gen += [sol1.crossover(sol2, f=select_value, mu_odds=mu_odds) for _ in range(children)]
            
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

def genetic(practice, population_size=100, max_gen=100, max_con=20):
    solutions = [Network() for _ in range(population_size)]
    best_sol = None
    best_score = -1
    gen = 0
    con = 0
    slice_size = 0
    text = ""
    while gen < max_gen and con < max_con and len(solutions) > slice_size:
        status_bar(status=float(gen) / float(max_gen), text=text)
        if len(solutions) > 300:
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
        text = str(best_score)
    status_bar(status=1.0, text=text)
    print("\nBest fit score: " + str(best_score))
    print("Generation: " + str(gen))
    return best_sol


def status_bar(status, text, ratio=30):
    current_status = int(status * ratio)
    text += "   [" + "#" * current_status + " " * (ratio - current_status) + "] " + f"{status*100:.2f}" + "%"
  
    print("\r" + text, end="", flush=True)


def parse_in(f):
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

def parse_out(ans, dest):
    with open(dest, 'w') as file:
        file.write(ans.parse_out())

def main():
    if len(sys.argv) < 2:
        print("Error: Missing argument.")
    else:
        n = sys.argv[1]
        data = "nn" + n + ".txt"
        dest = "wnet" + n + ".txt"
        practice, test = parse_in(data)
        print("Running, please wait...")
        ans = genetic(practice=practice)
        x = ans.fitness(test, override=True)
        print("Test score: " + str(x))
        print("Saving network...")
        parse_out(ans=ans, dest=dest)
        print("DONE!")

if __name__=="__main__":
    main()
