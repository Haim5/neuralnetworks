
import sys


class Network:
    def __init__(self, e):
        self.__edges = e
        self.__layers = [16, 2, 1]

    ## predict the tag
    def predict(self, key):
        values = [float(s) for s in key]
        values.append(0.0)
        values.append(0.0)
        values.append(0.0)
        y = sum(self.__layers)
        for k in range(y):
            e = self.__edges[k]
            for j in range(y):
                values[j] += e[j] * values[k]
        if values[-1] > 2:
            return 1
        return 0

## parse test data to list
def parse_in(f):
    # Read the data from the original file
    with open(f, 'r') as file:
        return file.readlines()

## parse wnet file to a Network object
def parse_net(f):
    ans = None
    with open(f, 'r') as file:
        e = []
        net = file.readlines()
        for line in net:
            l2 = line.split()
            row = [float(value) for value in l2]
            e.append(row)
        ans = Network(e=e)
    return ans

## parse predictions to a txt file
def parse_out(net, data, dest):
    with open(dest, 'w') as file:
        for d in data:
            d = d.rstrip('\n')
            res = str(int(net.predict(d)))
            file.write(d + " " + res + "\n")

def main():
    n = '1'
    if len(sys.argv) < 3:
        print("No files given. Assuming wnet and testnet exist in folder...")
        wnet = "wnet" + n + ".txt"
        data = "testnet" + n + ".txt"
    else:
        print("Running, please wait...")
        wnet = sys.argv[1]
        data = sys.argv[2]

    dest = "result" + n + ".txt"
    print("Loading data")
    test = parse_in(data)
    print("Loading net...")
    net = parse_net(f=wnet)
    print("Saving results...")
    parse_out(net=net, data=test, dest=dest)
    print("DONE!")

if __name__=="__main__":
    main()