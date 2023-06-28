    # def __init__(self, num_nodes):
    #     self.num_nodes = num_nodes
    #     self.weights = self.initialize_weights()
    #     self.__fitness = None
    #     self.__layers = num_nodes
    #     self.print_weights()

    # def initialize_weights(self):
    #     weights = []
    #     for i in range(len(self.num_nodes) - 1):
    #         layer_weights = self.generate_weights(self.num_nodes[i], self.num_nodes[i + 1])
    #         weights.append(layer_weights)
    #     return weights

    # @staticmethod
    # def generate_weights(input_nodes, output_nodes):
    #     weights = []
    #     for _ in range(input_nodes):
    #         layer_weights = [random.uniform(0, 1) for _ in range(output_nodes)]
    #         weights.append(layer_weights)
    #     return weights
    
    # def print_weights(self):
    #     for i, layer_weights in enumerate(self.weights):
    #         print(f"Weights for layer {i + 1}:")
    #         for weights in layer_weights:
    #             print(weights)
    #         print()


    # def predict(self, seq):
    #     output = seq
    #     for i, layer_weights in enumerate(self.weights):
    #         layer_output = []
    #         for weights in layer_weights:
    #             neuron_output = sum(w * x for w, x in zip(weights, output))
    #             neuron_output = sigmoid(neuron_output)
    #             layer_output.append(neuron_output)
    #         output = layer_output
    #     print(output)
    #     return output