import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

# Define constants
VELOCITY_THRESHOLD = 0.5
FLOW_THEORY_CONSTANT = 1.2

# Define configuration settings
class Configuration:
    def __init__(self, 
                 velocity_threshold: float = VELOCITY_THRESHOLD, 
                 flow_theory_constant: float = FLOW_THEORY_CONSTANT):
        """
        Configuration settings for the agent evaluation metrics.

        Args:
        - velocity_threshold (float): The velocity threshold for the velocity-threshold algorithm.
        - flow_theory_constant (float): The constant used in the flow theory algorithm.
        """
        self.velocity_threshold = velocity_threshold
        self.flow_theory_constant = flow_theory_constant

# Define exception classes
class EvaluationError(Exception):
    """Base class for evaluation-related exceptions."""
    pass

class InvalidInputError(EvaluationError):
    """Raised when invalid input is provided."""
    pass

class EvaluationMetric:
    def __init__(self, 
                 name: str, 
                 description: str, 
                 unit: str):
        """
        Base class for evaluation metrics.

        Args:
        - name (str): The name of the metric.
        - description (str): A brief description of the metric.
        - unit (str): The unit of the metric.
        """
        self.name = name
        self.description = description
        self.unit = unit

    def calculate(self, 
                   data: pd.DataFrame) -> float:
        """
        Calculate the metric value.

        Args:
        - data (pd.DataFrame): The input data.

        Returns:
        - float: The calculated metric value.
        """
        raise NotImplementedError

class VelocityThresholdMetric(EvaluationMetric):
    def __init__(self, 
                 configuration: Configuration):
        """
        Velocity-threshold metric.

        Args:
        - configuration (Configuration): The configuration settings.
        """
        super().__init__("Velocity Threshold", "The velocity threshold metric.", "m/s")
        self.configuration = configuration

    def calculate(self, 
                   data: pd.DataFrame) -> float:
        """
        Calculate the velocity-threshold metric value.

        Args:
        - data (pd.DataFrame): The input data.

        Returns:
        - float: The calculated metric value.
        """
        velocity = data["velocity"]
        threshold = self.configuration.velocity_threshold
        return np.mean(velocity > threshold)

class FlowTheoryMetric(EvaluationMetric):
    def __init__(self, 
                 configuration: Configuration):
        """
        Flow theory metric.

        Args:
        - configuration (Configuration): The configuration settings.
        """
        super().__init__("Flow Theory", "The flow theory metric.", "m^3/s")
        self.configuration = configuration

    def calculate(self, 
                   data: pd.DataFrame) -> float:
        """
        Calculate the flow theory metric value.

        Args:
        - data (pd.DataFrame): The input data.

        Returns:
        - float: The calculated metric value.
        """
        flow_rate = data["flow_rate"]
        constant = self.configuration.flow_theory_constant
        return np.mean(flow_rate * constant)

class Evaluator:
    def __init__(self, 
                 configuration: Configuration):
        """
        Evaluator class.

        Args:
        - configuration (Configuration): The configuration settings.
        """
        self.configuration = configuration
        self.metrics = [
            VelocityThresholdMetric(configuration),
            FlowTheoryMetric(configuration)
        ]

    def evaluate(self, 
                 data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the agent using the specified metrics.

        Args:
        - data (pd.DataFrame): The input data.

        Returns:
        - Dict[str, float]: A dictionary containing the metric values.
        """
        results = {}
        for metric in self.metrics:
            try:
                value = metric.calculate(data)
                results[metric.name] = value
            except Exception as e:
                logging.error(f"Error calculating {metric.name}: {str(e)}")
        return results

def main():
    # Create a configuration instance
    configuration = Configuration()

    # Create an evaluator instance
    evaluator = Evaluator(configuration)

    # Create sample data
    data = pd.DataFrame({
        "velocity": np.random.rand(100),
        "flow_rate": np.random.rand(100)
    })

    # Evaluate the agent
    results = evaluator.evaluate(data)

    # Print the results
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()