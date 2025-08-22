import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from enhanced_cs import constants
from enhanced_cs.utils import load_data, save_model, load_model
from enhanced_cs.models import CCWPModel
from enhanced_cs.metrics import calculate_metrics

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class TrainingPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.model = CCWPModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config["lr"])
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6)
        self.criterion = nn.MSELoss()

    def train(self, data_loader: DataLoader):
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def save_model(self, epoch: int):
        save_model(self.model, epoch)

    def load_model(self, epoch: int):
        self.model = load_model(epoch)

    def train_model(self):
        data_loader = DataLoader(load_data(self.config["data_path"]), batch_size=self.config["batch_size"], shuffle=True)
        best_loss = float("inf")
        for epoch in range(self.config["num_epochs"]):
            start_time = time.time()
            loss = self.train(data_loader)
            self.scheduler.step(loss)
            end_time = time.time()
            logging.info(f"Epoch {epoch+1}, Loss: {loss:.4f}, Time: {end_time - start_time:.2f} seconds")
            if loss < best_loss:
                best_loss = loss
                self.save_model(epoch)
            logging.info(f"Best Loss: {best_loss:.4f}")
            metrics = calculate_metrics(self.model, data_loader)
            logging.info(f"Metrics: {metrics}")

def main():
    config = {
        "data_path": "data.csv",
        "batch_size": 32,
        "num_epochs": 100,
        "lr": 1e-3
    }
    pipeline = TrainingPipeline(config)
    pipeline.train_model()

if __name__ == "__main__":
    main()