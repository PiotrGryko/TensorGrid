from app.micrograd.nn import CustomMLP


class NamesGeneratorMLP(CustomMLP):

    def __init__(self, inputs_per_neuron, list_of_layers_sizes):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        super().__init__(inputs_per_neuron, list_of_layers_sizes)

    def init_dataset(self):
        # Load the training data
        words = open('names.txt', 'r').read().splitlines()
        print(len(words))
        print(min([len(w) for w in words]))
        print(max([len(w) for w in words]))

        for w in words[:10]:
            for ch1, ch2 in zip(w,w[1:]):
                print(ch1, ch2)

    def get_train_x(self):
        # normalize x
        return self.train_x

    def get_train_y(self):
        # normalize y
        return self.train_y


def create_mlp():
    input_size = 784
    mlp = NamesGeneratorMLP(input_size, [32, 10,5, 1])
    mlp.init_dataset()
    return mlp

create_mlp()