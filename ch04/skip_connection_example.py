# # Execution of skip / shortcut / residual connection by an example

import pdb
import torch
import torch.nn as nn

from skip_connection import ExampleDeepNeuralNetwork




def print_gradients(model, x):

    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculates loss based on how close the target and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")




def main():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[-1., 0., 1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print(f"Without skip connection\n---------------------------")
    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print(f"With skip connection\n---------------------------")
    print_gradients(model_with_shortcut, sample_input)





if __name__ == "__main__":
    main()

