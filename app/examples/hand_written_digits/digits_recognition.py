import gzip

import numpy as np

from app.micrograd.nn import MLP, CustomMLP


class HandWrittenDigitsMLP(CustomMLP):

    def __init__(self, inputs_per_neuron, list_of_layers_sizes):
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.data_path = "/Users/v-piotr.gryko/develop/python/AiPlayground/app/examples/hand_written_digits/data"
        super().__init__(inputs_per_neuron, list_of_layers_sizes)

    def load_mnist(self, path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = f'{path}/{kind}-labels-idx1-ubyte.gz'
        images_path = f'{path}/{kind}-images-idx3-ubyte.gz'

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

        return images, labels

    def init_dataset(self):
        # Load the training data
        self.train_x, self.train_y = self.load_mnist(self.data_path, kind='train')
        # Load the testing data
        self.test_x, self.test_y = self.load_mnist(self.data_path, kind='t10k')

    def get_train_x(self):
        # normalize x
        x_train = self.train_x / 255
        xs = x_train[-10:]
        return xs

    def get_train_y(self):
        # normalize y
        y_train = self.train_y / 10
        ys = y_train[-10:]
        return ys


def create_mlp():
    input_size = 784
    mlp = HandWrittenDigitsMLP(input_size, [32, 10,5, 1])
    mlp.init_dataset()
    return mlp


def main():
    mlp = create_mlp()
    mlp.generate_custom()
    mlp.train_custom(50)
    mlp.generate_custom()


if __name__ == "__main__":
    main()

# size = len(xs[0])
#
# size = len(xs[0])
#
# if __name__ == "__main__":
#     input_size = 784
#     mlp = HandWrittenDigitsMLP(input_size, [10, 10, 1])
#     mlp.print_parameters()
#     mlp.generate(xs, ys)
#     mlp.train(10, xs, ys)
#     mlp.generate(xs, ys)
