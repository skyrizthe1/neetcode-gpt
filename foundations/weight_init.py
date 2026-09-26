"""
Weight Initialization
======================
Xavier (Glorot) init, Kaiming (He) init, and a diagnostic that shows why
initialization matters by tracking activation std across stacked layers.
"""

import torch
import numpy as np


class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        """
        Xavier/Glorot normal initialization (designed for sigmoid/tanh).

        std = sqrt(2 / (fan_in + fan_out))

        Averaging fan_in and fan_out keeps variance stable in BOTH the
        forward pass (which cares about fan_in) and the backward pass
        (which cares about fan_out).

        Returns a (fan_out x fan_in) weight matrix as a plain nested
        Python list (matching nn.Linear's weight shape convention).
        """
        torch.manual_seed(0)
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        return (torch.randn(fan_out, fan_in) * std).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> list[list[float]]:
        """
        Kaiming/He normal initialization (designed for ReLU).

        std = sqrt(2 / fan_in)

        ReLU zeroes out roughly half the activations, which halves the
        variance at every layer. Kaiming's extra factor of 2 cancels
        that loss out exactly.

        Returns a (fan_out x fan_in) weight matrix as a plain nested
        Python list.
        """
        torch.manual_seed(0)
        std = (2.0 / fan_in) ** 0.5
        return (torch.randn(fan_out, fan_in) * std).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> list[float]:
        """
        Diagnostic: stack `num_layers` Linear+ReLU layers using the
        requested init scheme, forward one random input through the
        stack, and record the std of the activations after each layer.

        init_type:
            'xavier'  -> std = sqrt(2 / (fan_in + fan_out))
            'kaiming' -> std = sqrt(2 / fan_in)
            'random'  -> std = 1  (plain N(0,1) weights, naive baseline)

        Layer shapes: input_dim -> hidden_dim -> hidden_dim -> ...
        (num_layers layers total, each followed by ReLU).

        Returns a list of num_layers floats: activation std after each
        layer, in order.
        """
        torch.manual_seed(0)

        def layer_std(fan_in: int, fan_out: int) -> float:
            if init_type == 'xavier':
                return   (2.0 / (fan_in + fan_out)) ** 0.5
            elif init_type == 'kaiming':
                fasn = (2.0 / fan_in) ** 0.5
                return fasn
            elif init_type == 'random':
                return 1.0
            else:
                raise ValueError(f"Unknown init_type: {init_type!r}")

        dims = [input_dim] + [hidden_dim] * num_layers  # [in, h, h, ..., h]

        # Build every weight matrix first...
        weights = []
        for i in range(num_layers):
            fan_in, fan_out = dims[i], dims[i + 1]
            weights.append(torch.randn(fan_out, fan_in) * layer_std(fan_in, fan_out))

        # ...then draw the input and forward it through the fixed weights.
        x = torch.randn(1, input_dim)
        stds = []
        for W in weights:
            x = torch.relu(x @ W.T)
            stds.append(round(x.std().item() , 2))

        return stds


