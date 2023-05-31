from app.micrograd.nn import CustomMLP


class MicrogradTestMLP(CustomMLP):

    def get_train_x(self):
        return [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0]
        ]

    def get_train_y(self):
        return [1.0, -1.0, -1.0, 1.0]

def create_mlp():
    return MicrogradTestMLP(3, [4, 4, 1])

def main():
    mlp = create_mlp()
    mlp.print_parameters()

    mlp.generate_custom()
    mlp.train_custom(100)
    mlp.generate_custom()


if __name__ == "__main__":
    main()
