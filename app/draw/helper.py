node_color = (119, 159, 161, 255)
node_size = 20
red_color = (255, 0, 0, 255)


line_color = (45, 127, 179, 255)
label_color = (0, 0, 0, 255)

node_bias_color = (217, 3, 104, 255)
hover_color = (0, 255, 0, 255)
selected_color = (0, 25, 255, 255)
pinned_color = (255, 122, 122, 255)

def generate_color(weight, default_color, target_color):
    color = [int(d + (t - d) * abs(weight)) for d, t in zip(default_color, target_color)]
    color[3] = target_color[3]
    return tuple(color)


def generate_line_color(weight):
    default_color = list(line_color)
    target_color = list(node_bias_color)
    return generate_color(weight, default_color, target_color)


def generate_node_color_w(weights):
    if len(weights)==0:
        return line_color
    default_color = list(node_color)
    target_color = list(node_bias_color)
    average = sum(abs(num) for num in weights) / len(weights)
    return generate_color(average, default_color, target_color)

def generate_node_color(meta):
    if len(meta.weights)==0:
        return line_color
    default_color = list(node_color)
    target_color = list(node_bias_color)
    average = sum(abs(num) for num in list(meta.weights.values())) / len(meta.weights)
    return generate_color(average, default_color, target_color)


