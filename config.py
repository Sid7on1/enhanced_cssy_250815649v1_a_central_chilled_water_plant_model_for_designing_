import os
import logging
from typing import Dict, List
from datetime import datetime

import yaml
import numpy as np

from agent import Agent
from environment import Environment
from components import Component

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """
    Configuration class for the agent and environment.

    Parameters:
    - config_file (str): Path to the YAML configuration file.
    """
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self._load_config()

        # Set up directories
        self._setup_directories()

        # Initialize agent and environment
        self.agent = self._init_agent()
        self.environment = self._init_environment()

        # Set up performance monitoring
        self.performance_metrics = PerformanceMetrics(self.config['performance_monitoring'])

    def _load_config(self) -> Dict:
        """Load configuration data from YAML file."""
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def _setup_directories(self):
        """Create necessary directories based on configuration."""
        dirs_to_create = [self.config['data_dir'], self.config['model_dir'], self.config['log_dir']]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

    def _init_agent(self) -> Agent:
        """Initialize the agent with the specified configuration."""
        agent_config = self.config['agent']
        components = [self._create_component(c) for c in agent_config['components']]
        return Agent(components, agent_config['learning_rate'], agent_config['discount_factor'])

    def _init_environment(self) -> Environment:
        """Initialize the environment with the specified configuration."""
        env_config = self.config['environment']
        return Environment(env_config['initial_state'], env_config['transition_probabilities'])

    def _create_component(self, component_config: Dict) -> Component:
        """Factory method to create component objects based on configuration."""
        component_type = component_config['type']
        if component_type == 'chiller':
            return ChillerComponent(**component_config)
        elif component_type == 'cooling_tower':
            return CoolingTowerComponent(**component_config)
        else:
            raise ValueError(f"Invalid component type: {component_type}")

class PerformanceMetrics:
    """
    Class to manage performance monitoring and metrics logging.

    Parameters:
    - config (Dict): Configuration for performance monitoring.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = {}

    def update_metric(self, metric_name: str, value):
        """Update the specified metric with a new value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def log_metrics(self):
        """Log the current values of all metrics."""
        for metric_name, values in self.metrics.items():
            metric_stats = self._calculate_statistics(values)
            logging.info(f"Metric '{metric_name}': {metric_stats}")

    def _calculate_statistics(self, values: List) -> Dict:
        """Calculate and return statistics (min, max, mean) for a list of values."""
        return {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values)
        }

# Example usage
if __name__ == '__main__':
    config_file = 'path/to/config.yaml'
    config = Config(config_file)

    # Perform agent-environment interactions and update performance metrics
    # ...

    # Log performance metrics
    config.performance_metrics.log_metrics()