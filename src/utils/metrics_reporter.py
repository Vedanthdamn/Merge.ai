"""
Metrics Reporter Module - Convenience Wrapper

This module provides a unified interface for all metrics computation and reporting.
It re-exports functionality from src.evaluation.metrics and src.evaluation.metrics_tracker
for easy access.

Usage:
    from src.utils.metrics_reporter import (
        evaluate_global, evaluate_per_client, compute_fairness,
        save_metrics_json, save_rounds_csv, 
        plot_accuracy_loss_vs_rounds, plot_accuracy_loss_vs_epsilon
    )
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, log_loss
)

# Import from the actual implementation modules
# Use try-except for flexible import resolution
try:
    from evaluation.metrics import MetricsComputer
    from evaluation.metrics_tracker import MetricsTracker
except ImportError:
    # If running as a module from project root
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    from src.evaluation.metrics import MetricsComputer
    from src.evaluation.metrics_tracker import MetricsTracker


# ============================================================
# High-level convenience functions
# ============================================================

def evaluate_global(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate global model performance on test data.
    
    Args:
        model: Trained model with predict() and evaluate() methods
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dict with accuracy, precision, recall, F1, confusion matrix, loss
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Compute metrics
    computer = MetricsComputer()
    metrics = computer.compute_model_performance(y_test, y_pred, y_pred_proba)
    
    return metrics


def evaluate_per_client(model, clients: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """
    Evaluate model performance on each client's data.
    
    Args:
        model: Trained model
        clients: List of (X_client, y_client) tuples
        
    Returns:
        Dict with per-client metrics
    """
    client_results = []
    
    for client_id, (X_client, y_client) in enumerate(clients):
        # Evaluate on this client's data
        eval_result = model.evaluate(X_client, y_client)
        
        client_results.append({
            'client_id': client_id,
            'accuracy': eval_result['accuracy'],
            'loss': eval_result.get('loss', None),
            'n_samples': len(X_client)
        })
    
    # Compute aggregate client metrics
    computer = MetricsComputer()
    client_metrics = computer.compute_client_level_metrics(client_results)
    
    return client_metrics


def compute_fairness(per_client_results: List[Dict],
                    client_sample_sizes: List[int]) -> Dict:
    """
    Compute fairness metrics across clients.
    
    Args:
        per_client_results: List of dicts with client metrics
        client_sample_sizes: Number of samples per client
        
    Returns:
        Dict with fairness metrics
    """
    computer = MetricsComputer()
    fairness_metrics = computer.compute_fairness_metrics(
        per_client_results, 
        client_sample_sizes
    )
    
    return fairness_metrics


def save_metrics_json(path: str, metrics_dict: Dict):
    """
    Save metrics dictionary to JSON file.
    
    Args:
        path: Output file path
        metrics_dict: Dictionary of metrics to save
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"✓ Metrics saved to: {path}")


def save_rounds_csv(path: str, rounds_history: List[Dict]):
    """
    Save per-round history to CSV file.
    
    Args:
        path: Output file path
        rounds_history: List of dicts with per-round metrics
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(rounds_history)
    df.to_csv(path, index=False)
    
    print(f"✓ Rounds history saved to: {path}")


def plot_accuracy_loss_vs_rounds(history: List[Dict], out_dir: str):
    """
    Plot accuracy and loss progression vs training rounds.
    
    Args:
        history: List of dicts with per-round metrics
        out_dir: Output directory for plots
    """
    tracker = MetricsTracker(output_dir=out_dir)
    
    # Add history to tracker
    for round_data in history:
        tracker.rounds_history.append(round_data)
    
    # Generate plots
    tracker.plot_accuracy_vs_rounds()
    tracker.plot_loss_vs_rounds()


def plot_accuracy_loss_vs_epsilon(dp_results: List[Dict], out_dir: str):
    """
    Plot accuracy and loss vs privacy budget (epsilon).
    
    Args:
        dp_results: List of dicts with metrics at different epsilon values
        out_dir: Output directory for plots
    """
    tracker = MetricsTracker(output_dir=out_dir)
    
    # Add privacy tradeoff data
    tracker.privacy_tradeoff_data = dp_results
    
    # Generate plots
    tracker.plot_accuracy_vs_epsilon()
    tracker.plot_loss_vs_epsilon()


# ============================================================
# Additional utility functions
# ============================================================

def compute_cross_entropy_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_true: True labels (binary)
        y_pred_proba: Predicted probabilities
        
    Returns:
        Cross-entropy loss value
    """
    # Ensure probabilities are in correct format
    if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
        proba = y_pred_proba.flatten()
    else:
        proba = y_pred_proba[:, 1]
    
    # Clip to avoid log(0)
    proba = np.clip(proba, 1e-7, 1 - 1e-7)
    
    # Compute binary cross-entropy
    loss = -np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba))
    
    return float(loss)


def compute_communication_metrics(n_clients: int, 
                                 n_rounds: int,
                                 model_size: int,
                                 smpc_enabled: bool = True) -> Dict:
    """
    Compute communication cost estimates for federated learning.
    
    Args:
        n_clients: Number of clients
        n_rounds: Number of training rounds
        model_size: Size of model in parameters
        smpc_enabled: Whether SMPC is enabled
        
    Returns:
        Dict with communication metrics
    """
    # Messages per round: each client uploads and downloads
    messages_per_round = n_clients * 2
    total_messages = messages_per_round * n_rounds
    
    # Estimate bytes (assuming ~4 bytes per parameter)
    bytes_per_model = model_size * 4
    
    # Upload: client sends model update to server
    upload_bytes_per_client = bytes_per_model
    total_upload_bytes = upload_bytes_per_client * n_clients * n_rounds
    
    # Download: server sends global model to each client
    download_bytes_per_client = bytes_per_model
    total_download_bytes = download_bytes_per_client * n_clients * n_rounds
    
    # Total transmission
    total_bytes = total_upload_bytes + total_download_bytes
    
    # SMPC overhead (typically 2-3x)
    if smpc_enabled:
        smpc_multiplier = 2.5
        total_bytes *= smpc_multiplier
    
    return {
        'total_messages': total_messages,
        'messages_per_round': messages_per_round,
        'total_bytes_transmitted': total_bytes,
        'upload_bytes_per_client': upload_bytes_per_client,
        'download_bytes_per_client': download_bytes_per_client,
        'bytes_per_round': total_bytes / n_rounds if n_rounds > 0 else 0,
        'smpc_multiplier': smpc_multiplier if smpc_enabled else 1.0
    }


def main():
    """Test the metrics reporter module."""
    print("Metrics Reporter Module")
    print("="*60)
    
    print("\nThis module provides convenience functions for:")
    print("  - evaluate_global(model, X_test, y_test)")
    print("  - evaluate_per_client(model, clients)")
    print("  - compute_fairness(per_client_results, sample_sizes)")
    print("  - save_metrics_json(path, metrics_dict)")
    print("  - save_rounds_csv(path, rounds_history)")
    print("  - plot_accuracy_loss_vs_rounds(history, out_dir)")
    print("  - plot_accuracy_loss_vs_epsilon(dp_results, out_dir)")
    print("  - compute_cross_entropy_loss(y_true, y_pred_proba)")
    print("  - compute_communication_metrics(n_clients, n_rounds, model_size)")
    
    print("\n" + "="*60)
    print("All functions available for import!")
    print("="*60)
    
    # Example usage
    print("\nExample: Computing communication metrics")
    print("-"*60)
    comm_metrics = compute_communication_metrics(
        n_clients=3,
        n_rounds=10,
        model_size=10000,
        smpc_enabled=True
    )
    print(json.dumps(comm_metrics, indent=2))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
