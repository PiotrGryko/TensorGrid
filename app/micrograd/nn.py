import random

from app.micrograd.micrograd import Value


class Neuron:
    def __init__(self, number_of_inputs, neuron_index, layer_index):
        self.label = f"neuron:{neuron_index} layer:{layer_index}"
        self.weights = [Value(random.uniform(-1, 1), label=f"weight x{_} {self.label}") for
                        _
                        in range(number_of_inputs)]
        self.bias = Value(random.uniform(-1, 1), label=f"bias neuron:{neuron_index} layer:{layer_index}")
        self.input_layer = layer_index == 0
        self.neuron_index = neuron_index
        self.layer_index = layer_index

    def __call__(self, x):
        # w * x + b
        act = sum(wi * xi for wi, xi in zip(self.weights, x)) + self.bias
        out = act if self.input_layer else act.tanh()
        return out

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, number_of_inputs_per_neuron, neurons_count, layer_index):
        self.layer_index = layer_index
        self.neurons = [Neuron(number_of_inputs=number_of_inputs_per_neuron,
                               layer_index=layer_index,
                               neuron_index=index)
                        for index in range(neurons_count)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:  # Multi layer perceptron
    def __init__(self, inputs_per_neuron, list_of_layers_sizes):
        sz = [inputs_per_neuron] + list_of_layers_sizes
        self.inputs_per_neuron = inputs_per_neuron
        self.layers = [Layer(sz[i], sz[i + 1], layer_index=i) for i in
                       range(len(list_of_layers_sizes))]

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def generate(self, xs, ys):
        result = []
        y_preds = [self(xi) for xi in xs]
        for pred, y in zip(y_preds, ys):
            msg = f"prediction {pred * 10} expected {y * 10} loss {(y - pred) ** 2}"
            result.append(msg)

        final_loss = sum((iy - ip) ** 2 for ip, iy in zip(y_preds, ys))
        result.append(f"final loss {final_loss}")
        for r in result:
            print(r)
        return "\n".join(result)

    def train(self, steps, xs, ys, on_tick=None):
        print("Training..")
        for k in range(0, steps):
            # forward pass
            y_preds = [self(xi) for xi in xs]
            loss = sum((iy - ip) ** 2 for ip, iy in zip(y_preds, ys))
            # backward pass
            loss.grad = 1.0
            loss.label = "loss"
            # IMPORTANT STEP !
            for p in self.parameters():
                p.grad = 0.0
            loss.backward()
            # update
            for p in self.parameters():
                p.data += -0.01 * p.grad
            print(k, loss.data)
            if on_tick:
                on_tick(loss.data, self)


class CustomMLP(MLP):
    def init_dataset(self):
        pass

    def get_train_x(self):
        return []

    def get_train_y(self):
        return []

    def generate_custom(self):
        return self.generate(self.get_train_x(), self.get_train_y())

    def train_custom(self, steps, on_tick=None):
        return self.train(steps, self.get_train_x(), self.get_train_y(), on_tick)

    def print_parameters(self):
        parameters = self.parameters()
        print(f"parameters len: {len(parameters)}")
        print("parameters:")
        for p in parameters:
            print(p)
