"""
Federated Learning Client using Flower Framework

This module implements a hospital/client in a federated learning system.

Key Privacy Features:
1. Local Training: Model trains on hospital's local data only
2. Model Updates Only: Only model weights (not raw data) are sent to server
3. Data Locality: Patient data never leaves the hospital

Federated Learning Process:
1. Server sends global model to client (hospital)
2. Client trains model on local patient data
3. Client sends updated model weights back to server
4. Server aggregates updates from all clients
5. Repeat for multiple rounds

This preserves privacy because:
- No patient records are transmitted
- Server never sees raw data
- Each hospital maintains full control over its data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import flwr as fl
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.model import BaselineHealthcareModel


class HealthcareFlowerClient(fl.client.NumPyClient):
    """
    Federated Learning client representing a hospital.
    
    Each client:
    - Maintains its own local patient data
    - Trains model locally without sharing data
    - Sends only model parameters to server
    """
    
    def __init__(self, client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Initialize a hospital client.
        
        Args:
            client_id (int): Unique identifier for this hospital
            X_train: Local training data (patient features)
            y_train: Local training labels (outcomes)
            X_val: Local validation data (optional)
            y_val: Local validation labels (optional)
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Initialize local model
        self.model = BaselineHealthcareModel(
            input_dim=X_train.shape[1],
            hidden_layers=[64, 32],
            dropout_rate=0.3
        )
        self.model.build_model()
        
        print(f"[Client {client_id}] Initialized with {len(X_train)} training samples")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Get model parameters (weights) to send to server.
        
        This is what gets transmitted in federated learning:
        - Neural network weights and biases
        - NO patient data
        - NO individual predictions
        
        Args:
            config: Configuration dictionary from server
            
        Returns:
            List of numpy arrays representing model parameters
        """
        return self.model.model.get_weights()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local hospital data.
        
        Privacy-Preserving Process:
        1. Receive global model parameters from server
        2. Update local model with global parameters
        3. Train on LOCAL patient data only
        4. Return updated parameters (not data) to server
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        # Set model parameters to global model
        self.model.model.set_weights(parameters)
        
        # Train on local data
        # Data never leaves this hospital!
        epochs = config.get("epochs", 5)
        batch_size = config.get("batch_size", 32)
        
        print(f"[Client {self.client_id}] Training for {epochs} epochs...")
        
        # Preprocess local data
        X_train_processed = self.model.preprocess_data(self.X_train, fit=True)
        
        # Local training
        history = self.model.model.fit(
            X_train_processed,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Get updated parameters
        updated_parameters = self.model.model.get_weights()
        
        # Compute metrics
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        print(f"[Client {self.client_id}] Training complete. "
              f"Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        
        # Return: updated weights, number of examples, metrics
        # Note: Only weights are sent to server, never patient data
        return updated_parameters, len(self.X_train), {"loss": final_loss, "accuracy": final_accuracy}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate global model on local test data.
        
        This allows the server to understand model performance
        across different hospitals without accessing their data.
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        # Set model parameters
        self.model.model.set_weights(parameters)
        
        # Evaluate on local validation data if available
        if self.X_val is not None and self.y_val is not None:
            X_val_processed = self.model.preprocess_data(self.X_val, fit=False)
            results = self.model.model.evaluate(X_val_processed, self.y_val, verbose=0)
            
            loss = results[0]
            accuracy = results[1]
            auc = results[2]
            
            print(f"[Client {self.client_id}] Evaluation: "
                  f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            return loss, len(self.X_val), {"accuracy": accuracy, "auc": auc}
        else:
            # No validation data available
            return 0.0, 0, {}


def create_client(client_id: int, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None):
    """
    Factory function to create a Flower client.
    
    Args:
        client_id: Unique client identifier
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        
    Returns:
        HealthcareFlowerClient instance
    """
    return HealthcareFlowerClient(client_id, X_train, y_train, X_val, y_val)


def start_client(client_id: int, server_address: str,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray = None, y_val: np.ndarray = None):
    """
    Start a federated learning client (hospital).
    
    Args:
        client_id: Unique identifier for this client
        server_address: Address of FL server (e.g., "localhost:8080")
        X_train: Local training data
        y_train: Local training labels
        X_val: Local validation data (optional)
        y_val: Local validation labels (optional)
    """
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client (Hospital {client_id})")
    print(f"{'='*60}")
    print(f"Server address: {server_address}")
    print(f"Local training samples: {len(X_train)}")
    if X_val is not None:
        print(f"Local validation samples: {len(X_val)}")
    print(f"{'='*60}\n")
    
    # Create client instance
    client = HealthcareFlowerClient(client_id, X_train, y_train, X_val, y_val)
    
    # Connect to server and start federated learning
    # This will participate in training rounds coordinated by the server
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


def main():
    """
    Example usage: Start a federated learning client.
    
    In practice, this would be run at each hospital with their local data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client (Hospital)")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--data-path", type=str, help="Path to local hospital data")
    
    args = parser.parse_args()
    
    # In practice, each hospital would load its own local data
    # For demonstration, create mock data
    print(f"Initializing client {args.client_id}...")
    print("In practice, would load local hospital data here.")
    print("For demonstration, using mock data.\n")
    
    np.random.seed(args.client_id)  # Different data per client
    n_samples = np.random.randint(200, 400)
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    # Start client
    start_client(
        client_id=args.client_id,
        server_address=args.server,
        X_train=X_train,
        y_train=y_train
    )


if __name__ == "__main__":
    main()
