from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .car import ControlInput


@dataclass(frozen=True)
class NetworkArchitecture:
    layers: tuple[int, ...] = (6, 10, 6, 2)

    @property
    def genome_length(self) -> int:
        return sum((inputs + 1) * outputs for inputs, outputs in zip(self.layers, self.layers[1:]))


class NeuralNetwork:
    """Small fully connected tanh network encoded as one flat genome."""

    def __init__(self, genome: Iterable[float], architecture: NetworkArchitecture | None = None) -> None:
        self.architecture = architecture or NetworkArchitecture()
        values = list(genome)
        if len(values) != self.architecture.genome_length:
            raise ValueError(f"Expected {self.architecture.genome_length} genes, got {len(values)}.")
        self.layers: list[tuple[list[list[float]], list[float]]] = []
        cursor = 0
        for inputs, outputs in zip(self.architecture.layers, self.architecture.layers[1:]):
            weights = [values[cursor + row * inputs:cursor + (row + 1) * inputs] for row in range(outputs)]
            cursor += inputs * outputs
            biases = values[cursor:cursor + outputs]
            cursor += outputs
            self.layers.append((weights, biases))

    def forward(self, inputs: Iterable[float]) -> tuple[float, float]:
        activations = list(inputs)
        if len(activations) != self.architecture.layers[0]:
            raise ValueError(f"Expected {self.architecture.layers[0]} inputs, got {len(activations)}.")
        for weights, biases in self.layers:
            activations = [
                math.tanh(sum(weight * value for weight, value in zip(row, activations)) + bias)
                for row, bias in zip(weights, biases)
            ]
        return activations[0], activations[1]

    def control(self, inputs: Iterable[float]) -> ControlInput:
        steering, drive = self.forward(inputs)
        return ControlInput(
            throttle=max(0.0, drive),
            brake=max(0.0, -drive),
            steering=steering,
        )
