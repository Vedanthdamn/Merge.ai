"""
Federated Learning Server using Flower Framework

This module implements the aggregation server in a federated learning system.

Key Responsibilities:
1. Coordinate training rounds across multiple hospitals (clients)
2. Aggregate model updates from clients using FedAvg algorithm
3. Distribute updated global model back to clients
4. Never access or store raw patient data

FedAvg (Federated Averaging) Algorithm:
1. Initialize global model
2. Send global model to all participating hospitals
3. Each hospital trains locally and returns updated weights
4. Server computes weighted average of all updates
5. Update global model with aggregated weights
6. Repeat for multiple rounds

Privacy Guarantees:
- Server never sees patient data
- Server only receives model parameters
- Hospitals maintain full data control
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.model import BaselineHealthcareModel


class HealthcareFederatedServer:
    """
    Federated Learning server for healthcare model aggregation.
    
    The server orchestrates distributed training without accessing
    hospital data, preserving privacy while enabling collaboration.
    """
    
    def __init__(self, input_dim: int, num_rounds: int = 10,
                 min_clients: int = 2, min_available_clients: int = 2):
        """
        Initialize the federated learning server.
        
        Args:
            input_dim (int): Number of input features
            num_rounds (int): Number of federated learning rounds
            min_clients (int): Minimum number of clients for training
            min_available_clients (int): Minimum clients that must be available
        """
        self.input_dim = input_dim
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_available_clients = min_available_clients
        
        # Initialize global model
        # This model is shared across hospitals but trained on distributed data
        self.global_model = BaselineHealthcareModel(
            input_dim=input_dim,
            hidden_layers=[64, 32],
            dropout_rate=0.3
        )
        self.global_model.build_model()
        
        print(f"[Server] Initialized with input_dim={input_dim}")
        print(f"[Server] Will run {num_rounds} federated learning rounds")
    
    def get_initial_parameters(self) -> List[np.ndarray]:
        """
        Get initial global model parameters.
        
        These parameters are sent to all clients at the start of training.
        
        Returns:
            List of numpy arrays representing model weights
        """
        return self.global_model.model.get_weights()
    
    def create_strategy(self) -> FedAvg:
        """
        Create the federated learning strategy (FedAvg).
        
        FedAvg (Federated Averaging):
        - Aggregates model updates from multiple clients
        - Computes weighted average based on dataset size
        - Hospitals with more patients have slightly more influence
        
        This is privacy-preserving because:
        - Only model parameters are aggregated
        - Individual hospital contributions are combined
        - No patient data is transmitted or stored
        
        Returns:
            FedAvg strategy instance
        """
        # Define configuration for client training
        def fit_config(server_round: int) -> Dict:
            """
            Configuration sent to clients for training.
            
            Can be customized per round (e.g., learning rate schedule).
            """
            config = {
                "epochs": 5,  # Local training epochs per round
                "batch_size": 32,
                "server_round": server_round,
            }
            return config
        
        # Define configuration for client evaluation
        def evaluate_config(server_round: int) -> Dict:
            """Configuration sent to clients for evaluation."""
            return {"server_round": server_round}
        
        # Create FedAvg strategy
        strategy = FedAvg(
            fraction_fit=1.0,  # Use all available clients for training
            fraction_evaluate=1.0,  # Use all available clients for evaluation
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_available_clients,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(self.get_initial_parameters()),
        )
        
        return strategy
    
    def start_server(self, server_address: str = "0.0.0.0:8080"):
        """
        Start the federated learning server.
        
        The server will:
        1. Wait for clients (hospitals) to connect
        2. Coordinate training rounds
        3. Aggregate model updates using FedAvg
        4. Distribute updated global model
        
        Args:
            server_address (str): Address to bind server to
        """
        print(f"\n{'='*60}")
        print("Starting Federated Learning Server")
        print(f"{'='*60}")
        print(f"Server address: {server_address}")
        print(f"Number of rounds: {self.num_rounds}")
        print(f"Minimum clients required: {self.min_clients}")
        print(f"{'='*60}\n")
        
        # Create aggregation strategy
        strategy = self.create_strategy()
        
        # Start Flower server
        # This will coordinate federated learning across hospitals
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )
        
        print(f"\n{'='*60}")
        print("Federated Learning Complete!")
        print(f"{'='*60}\n")


def main():
    """
    Main function to start the federated learning server.
    
    Usage:
        python server.py --rounds 10 --input-dim 10
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--input-dim", type=int, default=10, help="Input feature dimension")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", 
                        help="Server address")
    
    args = parser.parse_args()
    
    # Initialize and start server
    server = HealthcareFederatedServer(
        input_dim=args.input_dim,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        min_available_clients=args.min_clients
    )
    
    server.start_server(server_address=args.server_address)


if __name__ == "__main__":
    main()
