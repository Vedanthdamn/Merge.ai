"""
Split Learning Implementation for Healthcare

Split Learning is a privacy-preserving technique where the neural network
is split between client (hospital) and server.

Architecture:
┌─────────────┐
│  Hospital   │  ← Has patient data
│  (Client)   │
│             │
│  Layers 1-N │  ← Computes early layers locally
└──────┬──────┘
       │ Sends only intermediate activations (not raw data)
       ↓
┌─────────────┐
│   Server    │  ← Never sees patient data
│             │
│ Layers N+1+ │  ← Computes deeper layers
└─────────────┘

Privacy Benefits:
1. Raw patient data never leaves hospital
2. Server only sees abstract feature representations
3. Reduces computational load on hospital devices
4. Gradients flow back through the split for training

Key Difference from Federated Learning:
- FL: Entire model at client, send weight updates
- SL: Model split across client/server, send activations
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List
import socket
import pickle


class SplitModel:
    """Base class for split learning models."""
    
    def __init__(self):
        self.model = None
    
    def get_weights(self):
        """Get model weights."""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Set model weights."""
        self.model.set_weights(weights)


class ClientSideModel(SplitModel):
    """
    Client-side (hospital-side) model in split learning.
    
    This model:
    - Processes raw patient data
    - Computes early layers (feature extraction)
    - Sends intermediate activations to server
    - Receives gradients from server for backpropagation
    
    Privacy: Patient data stays at hospital, only abstract features sent out
    """
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [64]):
        """
        Initialize client-side model.
        
        Args:
            input_dim (int): Number of input features (patient characteristics)
            hidden_layers (list): Sizes of hidden layers on client side
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.build_model()
    
    def build_model(self):
        """
        Build client-side neural network.
        
        This network extracts features from patient data locally.
        The output (intermediate activations) is sent to server.
        """
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,))
        ])
        
        # Add client-side layers
        # These layers learn to extract relevant features from patient data
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.3))
        
        self.model = model
        print(f"[Client Model] Built with input_dim={self.input_dim}, "
              f"output_dim={self.hidden_layers[-1]}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: Compute intermediate activations.
        
        Process:
        1. Hospital inputs patient data
        2. Client model computes features
        3. These features (not raw data) are sent to server
        
        Args:
            X: Raw patient features
            
        Returns:
            Intermediate activations (abstract feature representation)
        """
        # Compute activations using local data
        activations = self.model.predict(X, verbose=0)
        
        # These activations represent learned features
        # They are abstract and don't directly reveal patient information
        return activations


class ServerSideModel(SplitModel):
    """
    Server-side model in split learning.
    
    This model:
    - Receives intermediate activations from client
    - Computes deeper layers
    - Makes final predictions
    - Computes gradients and sends back to client
    
    Privacy: Server never sees raw patient data, only processed features
    """
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [32], output_dim: int = 1):
        """
        Initialize server-side model.
        
        Args:
            input_dim (int): Size of input from client (activation size)
            hidden_layers (list): Sizes of hidden layers on server side
            output_dim (int): Output dimension (1 for binary classification)
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.build_model()
    
    def build_model(self):
        """
        Build server-side neural network.
        
        This network completes the prediction based on features from client.
        """
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,))
        ])
        
        # Add server-side layers
        # These layers make predictions from client features
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(0.3))
        
        # Output layer for binary classification
        model.add(layers.Dense(self.output_dim, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        self.model = model
        print(f"[Server Model] Built with input_dim={self.input_dim}, "
              f"output_dim={self.output_dim}")
    
    def forward(self, activations: np.ndarray) -> np.ndarray:
        """
        Forward pass: Make predictions from client activations.
        
        Args:
            activations: Intermediate features from client
            
        Returns:
            Final predictions
        """
        predictions = self.model.predict(activations, verbose=0)
        return predictions
    
    def train_step(self, activations: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Training step: Compute loss and gradients.
        
        Process:
        1. Forward pass with client activations
        2. Compute loss with true labels
        3. Backpropagate to get gradients
        4. Return gradients to client for their backprop
        
        Args:
            activations: Features from client
            y: True labels
            
        Returns:
            Tuple of (loss, gradients_for_client)
        """
        # Convert to tensors
        activations_tensor = tf.constant(activations, dtype=tf.float32)
        y_tensor = tf.constant(y, dtype=tf.float32)
        
        # Forward and backward pass
        with tf.GradientTape() as tape:
            tape.watch(activations_tensor)
            predictions = self.model(activations_tensor, training=True)
            loss = tf.keras.losses.binary_crossentropy(y_tensor, predictions)
            loss = tf.reduce_mean(loss)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Update server model weights
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        
        # Compute gradient w.r.t. activations to send back to client
        gradients_for_client = tape.gradient(loss, activations_tensor)
        
        return float(loss.numpy()), gradients_for_client.numpy()


class SplitLearningCoordinator:
    """
    Coordinates split learning between client and server.
    
    This class simulates the communication and training process
    in a split learning setup.
    """
    
    def __init__(self, input_dim: int, 
                 client_layers: List[int] = [64],
                 server_layers: List[int] = [32]):
        """
        Initialize split learning coordinator.
        
        Args:
            input_dim (int): Number of input features
            client_layers (list): Layer sizes for client model
            server_layers (list): Layer sizes for server model
        """
        self.input_dim = input_dim
        
        # Initialize client and server models
        self.client_model = ClientSideModel(input_dim, client_layers)
        self.server_model = ServerSideModel(
            client_layers[-1],  # Server input = client output
            server_layers,
            output_dim=1
        )
        
        print("\n" + "="*60)
        print("Split Learning Setup")
        print("="*60)
        print(f"Client-side layers: {client_layers}")
        print(f"Server-side layers: {server_layers}")
        print("="*60 + "\n")
    
    def train_split(self, X_train: np.ndarray, y_train: np.ndarray,
                    epochs: int = 10, batch_size: int = 32, verbose: int = 1):
        """
        Train models using split learning.
        
        Split Learning Training Process:
        1. Client forward pass: raw data → activations
        2. Send activations to server (not raw data!)
        3. Server forward pass: activations → predictions
        4. Compute loss on server
        5. Server backprop: compute gradients
        6. Send gradients to client (not predictions!)
        7. Client backprop: update client weights
        
        Args:
            X_train: Training features (at hospital)
            y_train: Training labels (at hospital)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
        """
        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"Training with split learning for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Step 1: Client forward pass (at hospital)
                # Raw patient data processed locally
                activations = self.client_model.forward(X_batch)
                
                # Step 2: Server training step
                # Server receives only activations, never raw data
                loss, gradients = self.server_model.train_step(activations, y_batch)
                epoch_losses.append(loss)
                
                # Step 3: Client backward pass would happen here
                # (Simplified in this simulation)
            
            # Print epoch results
            avg_loss = np.mean(epoch_losses)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("Split learning training complete!")
    
    def predict_split(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using split learning.
        
        Process:
        1. Client computes activations from patient data
        2. Send activations to server
        3. Server makes final predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Client forward pass
        activations = self.client_model.forward(X)
        
        # Server forward pass
        predictions = self.server_model.forward(activations)
        
        return predictions


def main():
    """
    Demonstration of split learning.
    """
    print("Split Learning Demonstration")
    print("="*60)
    
    # Create mock healthcare data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    print(f"Mock dataset: {n_samples} patients, {n_features} features\n")
    
    # Initialize split learning
    coordinator = SplitLearningCoordinator(
        input_dim=n_features,
        client_layers=[64],  # Hospital processes data to 64 features
        server_layers=[32]   # Server completes prediction
    )
    
    # Train with split learning
    coordinator.train_split(X_train, y_train, epochs=10, batch_size=32)
    
    # Make predictions
    print("\nMaking predictions on sample data...")
    X_sample = X_train[:5]
    predictions = coordinator.predict_split(X_sample)
    print(f"Sample predictions: {predictions.flatten()}")
    
    print("\n" + "="*60)
    print("Key Privacy Features:")
    print("- Patient data processed locally at hospital")
    print("- Only intermediate activations sent to server")
    print("- Server never sees raw patient information")
    print("- Reduces computational load on hospital devices")
    print("="*60)


if __name__ == "__main__":
    main()
