"""
Evaluation Module for Privacy-Preserving Federated Learning

This module implements comprehensive metrics for evaluating:
1. Model Performance Metrics
2. Federated Learning Specific Metrics
3. Differential Privacy Metrics
4. SMPC Metrics (placeholders)
5. Fairness Metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss
)
from typing import Dict, List, Optional, Tuple
import json


class MetricsComputer:
    """
    Comprehensive metrics computation for federated learning.
    
    Computes all required metrics including model performance,
    federated learning specific metrics, privacy metrics, and fairness metrics.
    """
    
    def __init__(self):
        """Initialize metrics computer."""
        self.metrics_history = []
        
    def compute_model_performance(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Compute standard model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (binary)
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dict with accuracy, precision, recall, F1, confusion matrix, loss
        """
        # Binary predictions
        if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
            y_pred_binary = (y_pred_proba[:, 1] > 0.5).astype(int) if y_pred_proba.shape[1] > 1 else (y_pred_proba[:, 0] > 0.5).astype(int)
        elif y_pred_proba is not None:
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_binary = y_pred
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Compute loss if probabilities available
        loss = None
        auc = None
        if y_pred_proba is not None:
            try:
                # Ensure probabilities are in correct format for binary classification
                if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:
                    proba_for_loss = y_pred_proba.flatten()
                else:
                    proba_for_loss = y_pred_proba[:, 1]
                
                # Clip probabilities to avoid log(0)
                proba_for_loss = np.clip(proba_for_loss, 1e-7, 1 - 1e-7)
                
                # Compute binary cross-entropy loss
                loss = -np.mean(y_true * np.log(proba_for_loss) + (1 - y_true) * np.log(1 - proba_for_loss))
                
                # Compute AUC if possible
                auc = roc_auc_score(y_true, proba_for_loss)
            except Exception as e:
                print(f"Warning: Could not compute loss/AUC: {e}")
                loss = None
                auc = None
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'loss': float(loss) if loss is not None else None,
            'auc': float(auc) if auc is not None else None
        }
        
        return metrics
    
    def compute_client_level_metrics(self,
                                     client_results: List[Dict]) -> Dict:
        """
        Compute per-client metrics for federated learning.
        
        Args:
            client_results: List of dicts with client metrics
                Each dict should have: {'accuracy', 'loss', 'n_samples'}
            
        Returns:
            Dict with client-level statistics
        """
        if not client_results:
            return {}
        
        accuracies = [r['accuracy'] for r in client_results]
        losses = [r['loss'] for r in client_results if 'loss' in r and r['loss'] is not None]
        samples = [r['n_samples'] for r in client_results if 'n_samples' in r]
        
        # Compute weighted averages
        if samples:
            total_samples = sum(samples)
            weighted_accuracy = sum(acc * n / total_samples for acc, n in zip(accuracies, samples))
        else:
            weighted_accuracy = np.mean(accuracies)
        
        metrics = {
            'client_accuracies': accuracies,
            'mean_accuracy': float(np.mean(accuracies)),
            'weighted_accuracy': float(weighted_accuracy),
            'accuracy_std': float(np.std(accuracies)),
            'accuracy_min': float(np.min(accuracies)),
            'accuracy_max': float(np.max(accuracies)),
        }
        
        if losses:
            metrics.update({
                'client_losses': losses,
                'mean_loss': float(np.mean(losses)),
                'loss_std': float(np.std(losses)),
            })
        
        return metrics
    
    def compute_federated_learning_metrics(self,
                                          rounds_history: List[Dict],
                                          convergence_threshold: float = 0.01,
                                          convergence_window: int = 3) -> Dict:
        """
        Compute federated learning specific metrics.
        
        Args:
            rounds_history: List of dicts with per-round metrics
            convergence_threshold: Threshold for determining convergence
            convergence_window: Number of rounds to check for convergence
            
        Returns:
            Dict with FL-specific metrics
        """
        if not rounds_history:
            return {}
        
        # Extract accuracy and loss trajectories
        accuracies = [r.get('accuracy', 0) for r in rounds_history]
        losses = [r.get('loss', 0) for r in rounds_history if r.get('loss') is not None]
        
        # Determine rounds to convergence
        rounds_to_converge = len(rounds_history)
        for i in range(convergence_window, len(accuracies)):
            recent_accuracies = accuracies[i-convergence_window:i]
            if max(recent_accuracies) - min(recent_accuracies) < convergence_threshold:
                rounds_to_converge = i
                break
        
        # Communication cost estimation
        total_rounds = len(rounds_history)
        messages_per_round = rounds_history[0].get('n_clients', 3) * 2  # upload + download per client
        total_messages = total_rounds * messages_per_round
        
        # Estimate bytes transmitted (rough estimate based on model size)
        # Assuming ~1MB per model update (typical for small NN)
        bytes_per_update = 1024 * 1024  # 1 MB
        total_bytes = total_messages * bytes_per_update
        
        # Client participation
        active_clients_per_round = [r.get('n_clients', 0) for r in rounds_history]
        
        # Convergence rate (improvement per round)
        if len(accuracies) > 1:
            accuracy_improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
            convergence_rate = np.mean(accuracy_improvements)
        else:
            convergence_rate = 0.0
        
        metrics = {
            'total_rounds': total_rounds,
            'rounds_to_convergence': rounds_to_converge,
            'communication_cost': {
                'total_messages': total_messages,
                'messages_per_round': messages_per_round,
                'total_bytes_transmitted': total_bytes,
                'bytes_per_round': total_bytes / total_rounds if total_rounds > 0 else 0
            },
            'client_participation': {
                'mean_active_clients': float(np.mean(active_clients_per_round)) if active_clients_per_round else 0,
                'active_clients_per_round': active_clients_per_round
            },
            'convergence_rate': float(convergence_rate),
            'final_accuracy': float(accuracies[-1]) if accuracies else 0.0,
            'accuracy_improvement': float(accuracies[-1] - accuracies[0]) if len(accuracies) > 1 else 0.0
        }
        
        return metrics
    
    def compute_client_drift_metrics(self,
                                     client_weights: List[np.ndarray],
                                     global_weights: np.ndarray) -> Dict:
        """
        Compute client drift and data heterogeneity impact.
        
        Args:
            client_weights: List of client model weights
            global_weights: Global model weights
            
        Returns:
            Dict with drift metrics
        """
        if not client_weights or global_weights is None:
            return {}
        
        # Flatten weights for comparison
        global_flat = np.concatenate([w.flatten() for w in global_weights])
        client_flats = [np.concatenate([w.flatten() for w in cw]) for cw in client_weights]
        
        # Compute weight divergence (L2 distance from global)
        divergences = [np.linalg.norm(cf - global_flat) for cf in client_flats]
        
        # Gradient divergence approximation (weight delta from global)
        gradient_divergences = [float(div) / len(global_flat) for div in divergences]
        
        metrics = {
            'mean_weight_divergence': float(np.mean(divergences)),
            'max_weight_divergence': float(np.max(divergences)),
            'min_weight_divergence': float(np.min(divergences)),
            'gradient_divergence_normalized': float(np.mean(gradient_divergences)),
            'client_divergences': [float(d) for d in divergences]
        }
        
        return metrics
    
    def compute_privacy_metrics(self,
                                epsilon: float,
                                delta: float,
                                n_rounds: int,
                                accuracy: float,
                                loss: Optional[float] = None) -> Dict:
        """
        Compute differential privacy metrics.
        
        Args:
            epsilon: Privacy budget
            delta: Privacy delta parameter
            n_rounds: Number of training rounds
            accuracy: Final model accuracy
            loss: Final model loss
            
        Returns:
            Dict with privacy metrics
        """
        # Privacy budget tracking
        # In composition, privacy budget accumulates
        total_epsilon = epsilon * n_rounds  # Simplified composition
        
        metrics = {
            'privacy_budget': {
                'epsilon_per_round': float(epsilon),
                'delta': float(delta),
                'total_epsilon': float(total_epsilon),
                'rounds': n_rounds
            },
            'accuracy_with_privacy': float(accuracy),
        }
        
        if loss is not None:
            metrics['loss_with_privacy'] = float(loss)
        
        return metrics
    
    def compute_smpc_metrics(self,
                            n_clients: int,
                            n_rounds: int,
                            model_size: int) -> Dict:
        """
        Compute SMPC metrics (placeholders for now).
        
        In a real implementation, these would measure actual SMPC overhead.
        
        Args:
            n_clients: Number of participating clients
            n_rounds: Number of rounds
            model_size: Size of model in parameters
            
        Returns:
            Dict with SMPC metrics (placeholders)
        """
        # Placeholder computation overhead (mock)
        # In reality, this would measure actual computation time
        computation_overhead_per_round = n_clients * 0.1  # Mock: 0.1s per client
        total_computation_overhead = computation_overhead_per_round * n_rounds
        
        # Placeholder communication overhead
        # SMPC typically adds ~2-3x communication overhead
        smpc_communication_multiplier = 2.5
        base_communication = model_size * n_clients * n_rounds
        smpc_communication_overhead = base_communication * (smpc_communication_multiplier - 1)
        
        # Placeholder aggregation latency
        aggregation_latency_per_round = n_clients * 0.05  # Mock: 0.05s per client
        total_aggregation_latency = aggregation_latency_per_round * n_rounds
        
        metrics = {
            'computation_overhead': {
                'per_round_seconds': float(computation_overhead_per_round),
                'total_seconds': float(total_computation_overhead),
                'note': 'Placeholder - would measure actual SMPC computation time'
            },
            'communication_overhead': {
                'multiplier': float(smpc_communication_multiplier),
                'additional_bytes': float(smpc_communication_overhead),
                'note': 'Placeholder - would measure actual SMPC communication overhead'
            },
            'aggregation_latency': {
                'per_round_seconds': float(aggregation_latency_per_round),
                'total_seconds': float(total_aggregation_latency),
                'note': 'Placeholder - would measure actual aggregation time'
            }
        }
        
        return metrics
    
    def compute_fairness_metrics(self,
                                 client_results: List[Dict],
                                 client_sample_sizes: List[int]) -> Dict:
        """
        Compute fairness metrics for federated learning.
        
        Args:
            client_results: List of per-client results with accuracy/loss
            client_sample_sizes: Number of samples per client
            
        Returns:
            Dict with fairness metrics
        """
        if not client_results:
            return {}
        
        accuracies = [r['accuracy'] for r in client_results]
        
        # Client accuracy variance (measure of fairness)
        accuracy_variance = np.var(accuracies)
        
        # Per-client contribution score (approximation)
        # Higher sample size = higher contribution to global model
        total_samples = sum(client_sample_sizes)
        contribution_scores = [n / total_samples for n in client_sample_sizes]
        
        # Weighted fairness: variance weighted by contribution
        weighted_accuracy_variance = np.average(
            [(acc - np.mean(accuracies))**2 for acc in accuracies],
            weights=contribution_scores
        )
        
        metrics = {
            'client_accuracy_variance': float(accuracy_variance),
            'weighted_accuracy_variance': float(weighted_accuracy_variance),
            'accuracy_range': float(max(accuracies) - min(accuracies)),
            'contribution_scores': [float(cs) for cs in contribution_scores],
            'fairness_score': float(1.0 / (1.0 + accuracy_variance))  # Higher is more fair
        }
        
        return metrics


def main():
    """Test metrics computation."""
    print("Testing Metrics Computer")
    print("="*60)
    
    computer = MetricsComputer()
    
    # Test 1: Model performance metrics
    print("\n1. Model Performance Metrics")
    print("-"*60)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.3, 0.7, 0.9, 0.1, 0.6])
    
    metrics = computer.compute_model_performance(y_true, y_pred, y_pred_proba)
    print(json.dumps(metrics, indent=2))
    
    # Test 2: Client-level metrics
    print("\n2. Client-Level Metrics")
    print("-"*60)
    client_results = [
        {'accuracy': 0.85, 'loss': 0.4, 'n_samples': 100},
        {'accuracy': 0.82, 'loss': 0.5, 'n_samples': 150},
        {'accuracy': 0.88, 'loss': 0.3, 'n_samples': 80}
    ]
    metrics = computer.compute_client_level_metrics(client_results)
    print(json.dumps(metrics, indent=2))
    
    # Test 3: Privacy metrics
    print("\n3. Privacy Metrics")
    print("-"*60)
    metrics = computer.compute_privacy_metrics(
        epsilon=1.0, delta=1e-5, n_rounds=10, accuracy=0.85, loss=0.4
    )
    print(json.dumps(metrics, indent=2))
    
    print("\n" + "="*60)
    print("Metrics computation test complete!")


if __name__ == "__main__":
    main()
