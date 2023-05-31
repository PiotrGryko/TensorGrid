import numpy as np


class Value:
    def __init__(self, data, children=(), operator='', label=''):
        self.data = data
        # derivative calculated in backward pass using chain rule
        # Indicates the impact of this value on the entire equation
        self.grad = 0.0
        self.children = children
        self.label = label
        self.operator = operator
        self._backward = None

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            if v._backward:
                v._backward()

    def tanh(self):
        x = self.data
        t = np.tanh(x)  # (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            x = self.data
            # print(x)
            coshx = np.cosh(x)  # 1 + math.exp(-2 * x)) / (2 * math.exp(-x))
            cosh2x = coshx ** 2
            self.grad += (1 / cosh2x) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            # print("addition grad {0} {1} {2}".format(self.data, other.data, out.grad))

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            # print("multiplication grad {0} {1} {2}".format(self.data, other.data, out.grad))

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), f"**{other.data}")

        def _backward():
            self.grad += other.data * self.data ** (other.data - 1) * out.grad

        out._backward = _backward

        return out

    def __gt__(self, other):
        return self.data > other.data

    def __repr__(self):
        return f"{self.label}Value({self.data})"
        # if self.operator == "tanh":
        #     return f"Value({self.data}) created from tanh({self.children[0].data})"
        # else:
        #     return f"Value({self.data}) created from {self.children[0].data} {self.operator} {self.children[1].data}"

