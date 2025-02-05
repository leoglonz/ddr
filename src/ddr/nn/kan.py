import logging

import torch
import torch.nn.functional as F
from kan import KAN

log = logging.getLogger(__name__)


class kan(torch.nn.Module):
    """A Kolmogorov Arnold Neural Network (KAN)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = len(kwargs["input_var_names"])
        self.hidden_size = kwargs["hidden_size"]
        self.output_size = kwargs["output_size"]

        self.input = torch.nn.Linear(self.input_size, self.hidden_size)
        self.layers = torch.nn.ModuleList()
        for _ in range(kwargs["num_hidden_layers"]):
            self.layers.append(
                KAN(
                    [self.hidden_size, self.hidden_size],
                    k=kwargs["k"],
                    grid=kwargs["grid"],
                    seed=kwargs["seed"],
                )
            )
            self.output = torch.nn.Linear(self.hidden_size, self.output_size, bias=False)
            torch.nn.init.kaiming_normal_(self.input.weight)
            torch.nn.init.kaiming_normal_(self.output.weight)
            torch.nn.init.kaiming_normal_(self.output.bias)
            torch.nn.init.zeros_(self.input.bias)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass of the neural network"""
        _x: torch.Tensor = kwargs["inputs"]
        outputs = {}
        _x = self.input(_x)
        for layer in self.layers:
            _x = layer(_x)
        _x = self.output(_x)
        _x = F.sigmoid(_x)
        x_transpose = _x.transpose(0, 1)
        for idx, key in enumerate(self.cfg.learnable_parameters):
            outputs[key] = x_transpose[idx]
        return outputs
