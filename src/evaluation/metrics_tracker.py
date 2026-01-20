"""
Metrics Tracker and Visualization

This module tracks metrics across training rounds and generates:
- JSON summary of all metrics
- CSV history of rounds
- Plots (accuracy/loss vs rounds, accuracy/loss vs epsilon)
"""

import numpy as np
import json
import csv
import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsTracker:
    """
    Tracks and visualizes metrics throughout federated learning.
    
    Saves:
    - outputs/metrics.json: Complete metrics summary
    - outputs/rounds_history.csv: Per-round metrics
    - outputs/plots/*.png: Visualization plots
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize metrics tracker.
        
        Args:
            output_dir: Base directory for outputs
        """
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Tracking data
        self.rounds_history = []
        self.privacy_tradeoff_data = []
        self.final_metrics = {}
        
        print(f"Metrics tracker initialized. Output directory: {output_dir}")
    
    def track_round(self, round_num: int, metrics: Dict):
        """
        Track metrics for a single round.
        
        Args:
            round_num: Round number
            metrics: Dict with round metrics
        """
        round_data = {
            'round': round_num,
            **metrics
        }
        self.rounds_history.append(round_data)
    
    def track_privacy_tradeoff(self, epsilon: float, metrics: Dict):
        """
        Track metrics for privacy-accuracy tradeoff analysis.
        
        Args:
            epsilon: Privacy budget epsilon
            metrics: Dict with metrics at this epsilon
        """
        tradeoff_data = {
            'epsilon': epsilon,
            **metrics
        }
        self.privacy_tradeoff_data.append(tradeoff_data)
    
    def set_final_metrics(self, metrics: Dict):
        """
        Set final comprehensive metrics.
        
        Args:
            metrics: Complete metrics dict
        """
        self.final_metrics = metrics
    
    def save_metrics_json(self, filename: str = 'metrics.json'):
        """
        Save all metrics to JSON file.
        
        Args:
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        output = {
            'final_metrics': self.final_metrics,
            'rounds_history': self.rounds_history,
            'privacy_tradeoff': self.privacy_tradeoff_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Metrics saved to: {filepath}")
    
    def save_rounds_history_csv(self, filename: str = 'rounds_history.csv'):
        """
        Save per-round history to CSV.
        
        Args:
            filename: Output filename
        """
        if not self.rounds_history:
            print("Warning: No rounds history to save")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Get all unique keys
        keys = set()
        for round_data in self.rounds_history:
            keys.update(round_data.keys())
        keys = sorted(keys)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for round_data in self.rounds_history:
                # Convert lists/dicts to strings for CSV
                row = {}
                for k, v in round_data.items():
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v)
                    else:
                        row[k] = v
                writer.writerow(row)
        
        print(f"✓ Rounds history saved to: {filepath}")
    
    def plot_accuracy_vs_rounds(self, filename: str = 'accuracy_vs_rounds.png'):
        """
        Plot accuracy progression across rounds.
        
        Args:
            filename: Output filename
        """
        if not self.rounds_history:
            print("Warning: No rounds history to plot")
            return
        
        rounds = [r['round'] for r in self.rounds_history]
        accuracies = [r.get('accuracy', 0) for r in self.rounds_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Global Accuracy', fontsize=12)
        plt.title('Model Accuracy vs Federated Learning Rounds', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {filepath}")
    
    def plot_loss_vs_rounds(self, filename: str = 'loss_vs_rounds.png'):
        """
        Plot loss progression across rounds.
        
        Args:
            filename: Output filename
        """
        if not self.rounds_history:
            print("Warning: No rounds history to plot")
            return
        
        rounds = [r['round'] for r in self.rounds_history]
        losses = [r.get('loss', 0) for r in self.rounds_history if r.get('loss') is not None]
        
        if not losses:
            print("Warning: No loss data to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds[:len(losses)], losses, marker='o', linewidth=2, markersize=8, color='red')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Global Loss', fontsize=12)
        plt.title('Model Loss vs Federated Learning Rounds', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {filepath}")
    
    def plot_accuracy_vs_epsilon(self, filename: str = 'accuracy_vs_epsilon.png'):
        """
        Plot accuracy vs privacy budget (epsilon).
        
        Args:
            filename: Output filename
        """
        if not self.privacy_tradeoff_data:
            print("Note: No privacy tradeoff data to plot")
            return
        
        epsilons = [d['epsilon'] for d in self.privacy_tradeoff_data]
        accuracies = [d.get('accuracy', 0) for d in self.privacy_tradeoff_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8, color='green')
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('Global Accuracy', fontsize=12)
        plt.title('Privacy-Accuracy Tradeoff', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {filepath}")
    
    def plot_loss_vs_epsilon(self, filename: str = 'loss_vs_epsilon.png'):
        """
        Plot loss vs privacy budget (epsilon).
        
        Args:
            filename: Output filename
        """
        if not self.privacy_tradeoff_data:
            print("Note: No privacy tradeoff data to plot")
            return
        
        epsilons = [d['epsilon'] for d in self.privacy_tradeoff_data]
        losses = [d.get('loss', 0) for d in self.privacy_tradeoff_data if d.get('loss') is not None]
        
        if not losses or len(losses) != len(epsilons):
            print("Note: Insufficient loss data for epsilon plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, losses, marker='o', linewidth=2, markersize=8, color='purple')
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('Global Loss', fontsize=12)
        plt.title('Privacy-Loss Tradeoff', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {filepath}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, filename: str = 'confusion_matrix.png'):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix (2D array)
            filename: Output filename
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {filepath}")
    
    def plot_client_accuracy_variance(self, 
                                      client_accuracies: List[float],
                                      filename: str = 'client_accuracy_variance.png'):
        """
        Plot client accuracy distribution.
        
        Args:
            client_accuracies: List of per-client accuracies
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        
        # Bar plot
        clients = [f'Client {i+1}' for i in range(len(client_accuracies))]
        plt.bar(clients, client_accuracies, color='steelblue', alpha=0.7)
        
        # Add mean line
        mean_acc = np.mean(client_accuracies)
        plt.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
        
        plt.xlabel('Client', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Per-Client Accuracy Distribution (Fairness)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plot saved to: {filepath}")
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print(f"\n{'='*60}")
        print("Generating visualization plots...")
        print(f"{'='*60}\n")
        
        self.plot_accuracy_vs_rounds()
        self.plot_loss_vs_rounds()
        
        if self.privacy_tradeoff_data:
            self.plot_accuracy_vs_epsilon()
            self.plot_loss_vs_epsilon()
        
        # Plot confusion matrix if available in final metrics
        if 'model_performance' in self.final_metrics:
            if 'confusion_matrix' in self.final_metrics['model_performance']:
                cm = np.array(self.final_metrics['model_performance']['confusion_matrix'])
                self.plot_confusion_matrix(cm)
        
        # Plot client accuracy variance if available
        if 'fairness_metrics' in self.final_metrics:
            if 'client_accuracy_variance' in self.final_metrics['fairness_metrics']:
                # Need to get client accuracies from somewhere
                # Check if we stored them in final metrics
                pass  # Will be populated by the main script
        
        print(f"\n{'='*60}")
        print("All plots generated successfully!")
        print(f"{'='*60}\n")
    
    def save_all(self):
        """Save all metrics and generate all plots."""
        print(f"\n{'='*60}")
        print("Saving Metrics and Generating Reports")
        print(f"{'='*60}\n")
        
        self.save_metrics_json()
        self.save_rounds_history_csv()
        self.generate_all_plots()
        
        print(f"\n{'='*60}")
        print("All outputs saved successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """Test the metrics tracker."""
    print("Testing Metrics Tracker")
    print("="*60)
    
    tracker = MetricsTracker(output_dir='outputs_test')
    
    # Simulate tracking rounds
    print("\nSimulating federated learning rounds...")
    for round_num in range(1, 11):
        metrics = {
            'accuracy': 0.5 + round_num * 0.03,
            'loss': 0.8 - round_num * 0.05,
            'n_clients': 3
        }
        tracker.track_round(round_num, metrics)
    
    # Simulate privacy tradeoff
    print("\nSimulating privacy-accuracy tradeoff...")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        metrics = {
            'accuracy': 0.65 + 0.05 * np.log(epsilon + 1),
            'loss': 0.5 - 0.03 * np.log(epsilon + 1)
        }
        tracker.track_privacy_tradeoff(epsilon, metrics)
    
    # Set final metrics
    tracker.set_final_metrics({
        'model_performance': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85,
            'confusion_matrix': [[45, 5], [7, 43]]
        },
        'federated_learning': {
            'total_rounds': 10,
            'rounds_to_convergence': 8
        }
    })
    
    # Save everything
    tracker.save_all()
    
    print("\n" + "="*60)
    print("Metrics tracker test complete!")
    print("Check 'outputs_test' directory for results")


if __name__ == "__main__":
    main()
