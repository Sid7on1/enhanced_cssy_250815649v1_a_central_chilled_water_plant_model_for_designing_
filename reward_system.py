import logging
import numpy as np
from typing import Dict, List, Tuple
from reward_system.config import Config
from reward_system.exceptions import RewardSystemError
from reward_system.models import RewardModel
from reward_system.utils import calculate_velocity_threshold, calculate_flow_theory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RewardSystem:
    """
    Reward calculation and shaping system.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward system with the given configuration.

        Args:
            config (Config): The configuration for the reward system.
        """
        self.config = config
        self.reward_model = RewardModel(config)

    def calculate_reward(self, state: Dict, action: Dict) -> float:
        """
        Calculate the reward for the given state and action.

        Args:
            state (Dict): The current state of the system.
            action (Dict): The action taken by the agent.

        Returns:
            float: The calculated reward.
        """
        try:
            # Calculate the velocity threshold
            velocity_threshold = calculate_velocity_threshold(state, self.config)

            # Calculate the flow theory
            flow_theory = calculate_flow_theory(state, action, self.config)

            # Calculate the reward using the reward model
            reward = self.reward_model.calculate_reward(state, action, velocity_threshold, flow_theory)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

    def shape_reward(self, reward: float) -> float:
        """
        Shape the reward to fit the agent's learning curve.

        Args:
            reward (float): The reward to be shaped.

        Returns:
            float: The shaped reward.
        """
        try:
            # Apply the reward shaping formula
            shaped_reward = self.config.reward_shaping_formula(reward)

            return shaped_reward

        except RewardSystemError as e:
            logger.error(f"Error shaping reward: {e}")
            return 0.0

class RewardModel:
    """
    Reward model for the reward system.
    """

    def __init__(self, config: Config):
        """
        Initialize the reward model with the given configuration.

        Args:
            config (Config): The configuration for the reward model.
        """
        self.config = config

    def calculate_reward(self, state: Dict, action: Dict, velocity_threshold: float, flow_theory: float) -> float:
        """
        Calculate the reward using the given state, action, velocity threshold, and flow theory.

        Args:
            state (Dict): The current state of the system.
            action (Dict): The action taken by the agent.
            velocity_threshold (float): The calculated velocity threshold.
            flow_theory (float): The calculated flow theory.

        Returns:
            float: The calculated reward.
        """
        try:
            # Calculate the reward using the reward formula
            reward = self.config.reward_formula(state, action, velocity_threshold, flow_theory)

            return reward

        except RewardSystemError as e:
            logger.error(f"Error calculating reward: {e}")
            return 0.0

class Config:
    """
    Configuration for the reward system.
    """

    def __init__(self):
        """
        Initialize the configuration with default values.
        """
        self.reward_shaping_formula = self.default_reward_shaping_formula
        self.reward_formula = self.default_reward_formula

    def default_reward_shaping_formula(self, reward: float) -> float:
        """
        Default reward shaping formula.

        Args:
            reward (float): The reward to be shaped.

        Returns:
            float: The shaped reward.
        """
        return reward * 0.5

    def default_reward_formula(self, state: Dict, action: Dict, velocity_threshold: float, flow_theory: float) -> float:
        """
        Default reward formula.

        Args:
            state (Dict): The current state of the system.
            action (Dict): The action taken by the agent.
            velocity_threshold (float): The calculated velocity threshold.
            flow_theory (float): The calculated flow theory.

        Returns:
            float: The calculated reward.
        """
        return velocity_threshold + flow_theory

class RewardSystemError(Exception):
    """
    Exception for reward system errors.
    """

    def __init__(self, message: str):
        """
        Initialize the exception with the given message.

        Args:
            message (str): The error message.
        """
        self.message = message
        super().__init__(self.message)

def calculate_velocity_threshold(state: Dict, config: Config) -> float:
    """
    Calculate the velocity threshold using the given state and configuration.

    Args:
        state (Dict): The current state of the system.
        config (Config): The configuration for the reward system.

    Returns:
        float: The calculated velocity threshold.
    """
    try:
        # Calculate the velocity threshold using the formula
        velocity_threshold = config.velocity_threshold_formula(state)

        return velocity_threshold

    except RewardSystemError as e:
        logger.error(f"Error calculating velocity threshold: {e}")
        return 0.0

def calculate_flow_theory(state: Dict, action: Dict, config: Config) -> float:
    """
    Calculate the flow theory using the given state, action, and configuration.

    Args:
        state (Dict): The current state of the system.
        action (Dict): The action taken by the agent.
        config (Config): The configuration for the reward system.

    Returns:
        float: The calculated flow theory.
    """
    try:
        # Calculate the flow theory using the formula
        flow_theory = config.flow_theory_formula(state, action)

        return flow_theory

    except RewardSystemError as e:
        logger.error(f"Error calculating flow theory: {e}")
        return 0.0