"""
Complete Privacy-Preserving Federated Learning Demo with Benchmark Evaluation

This script demonstrates the complete federated learning workflow with:
- Benchmark dataset evaluation (diabetes, breast_cancer)
- SRM hospital dataset support (plug-and-play)
- Comprehensive metrics computation and tracking
- Privacy-preserving techniques (DP, SMPC)
- Complete output generation (JSON, CSV, plots)

Usage:
    python demo_integration.py --dataset benchmark
    python demo_integration.py --dataset srm
    python demo_integration.py --config config.yaml
"""

import numpy as np
import argparse
import yaml
import sys
import os
from typing import Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from src.utils.dataset_loader import DatasetLoader
from src.utils.data_partitioner import HospitalDataPartitioner
from src.utils.srm_dataset_adapter import SRMDatasetAdapter
from src.baseline.model import BaselineHealthcareModel
from src.privacy.differential_privacy import DifferentialPrivacy
from src.privacy.smpc import SecureAggregator
from src.blockchain.audit_log import FederatedLearningBlockchain
from src.evaluation.metrics import MetricsComputer
from src.evaluation.metrics_tracker import MetricsTracker


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_data(config: Dict, dataset_type: str = None) -> Tuple:
    """
    Load dataset based on configuration.
    
    Args:
        config: Configuration dict
        dataset_type: Override dataset type ('benchmark' or 'srm')
        
    Returns:
        Tuple of (X, y, dataset_info)
    """
    print(f"\n{'='*80}")
    print("STEP 1: LOADING DATASET")
    print(f"{'='*80}\n")
    
    loader = DatasetLoader(random_seed=config.get('training', {}).get('random_seed', 42))
    
    # Determine dataset type
    if dataset_type:
        dataset_config_type = dataset_type
    else:
        dataset_config_type = config.get('dataset', {}).get('type', 'benchmark')
    
    dataset_info = {'type': dataset_config_type}
    
    if dataset_config_type == 'benchmark':
        # Load benchmark dataset
        benchmark_name = config.get('dataset', {}).get('benchmark_name', 'diabetes')
        print(f"Loading benchmark dataset: {benchmark_name}")
        X, y = loader.load_dataset(benchmark_name)
        dataset_info['name'] = benchmark_name
        
    elif dataset_config_type == 'srm':
        # Load SRM hospital dataset
        srm_config = config.get('dataset', {}).get('srm', {})
        srm_path = srm_config.get('path', 'data/srm_hospital_data.csv')
        schema = srm_config.get('schema', None)
        
        print(f"Loading SRM hospital dataset: {srm_path}")
        
        # Use SRM adapter
        adapter = SRMDatasetAdapter(schema_config=schema)
        
        # Validate schema first
        validation = adapter.validate_schema(srm_path)
        if not validation['valid']:
            print(f"ERROR: {validation['error']}")
            print("Please check your schema configuration in config.yaml")
            sys.exit(1)
        
        X, y, metadata = adapter.load_srm_dataset(srm_path)
        dataset_info['name'] = 'SRM Global Hospital'
        dataset_info['metadata'] = metadata
        
    elif dataset_config_type == 'csv':
        # Load custom CSV
        csv_path = config.get('dataset', {}).get('custom_path', 'data/healthcare_data.csv')
        schema = config.get('dataset', {}).get('schema', None)
        print(f"Loading custom CSV dataset: {csv_path}")
        X, y = loader.load_dataset('csv', csv_path, schema)
        dataset_info['name'] = 'Custom CSV'
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_config_type}")
    
    print(f"\nDataset loaded successfully!")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Positive class: {np.sum(y == 1)} ({np.mean(y)*100:.1f}%)")
    print(f"  Negative class: {np.sum(y == 0)} ({(1-np.mean(y))*100:.1f}%)")
    
    return X, y, dataset_info


def simulate_federated_learning(X: np.ndarray, 
                                y: np.ndarray,
                                config: Dict,
                                dataset_info: Dict):
    """
    Simulate complete federated learning with all privacy mechanisms.
    
    Args:
        X: Feature data
        y: Target labels
        config: Configuration dict
        dataset_info: Dataset metadata
    """
    
    # Extract configuration
    fl_config = config.get('federated_learning', {})
    privacy_config = config.get('privacy', {})
    training_config = config.get('training', {})
    output_config = config.get('output', {})
    metrics_config = config.get('metrics', {})
    
    n_clients = fl_config.get('n_clients', 3)
    n_rounds = fl_config.get('n_rounds', 10)
    local_epochs = fl_config.get('local_epochs', 5)
    batch_size = fl_config.get('batch_size', 32)
    partition_strategy = fl_config.get('partition_strategy', 'non_iid')
    
    # Privacy settings
    dp_enabled = privacy_config.get('differential_privacy', {}).get('enabled', True)
    epsilon = privacy_config.get('differential_privacy', {}).get('epsilon', 1.0)
    delta = privacy_config.get('differential_privacy', {}).get('delta', 1e-5)
    clip_norm = privacy_config.get('differential_privacy', {}).get('clip_norm', 1.0)
    
    smpc_enabled = privacy_config.get('smpc', {}).get('enabled', True)
    
    # Initialize metrics tracker
    output_dir = output_config.get('base_dir', 'outputs')
    tracker = MetricsTracker(output_dir=output_dir)
    metrics_computer = MetricsComputer()
    
    # ============================================================
    # Step 2: Partition data across hospitals (Non-IID)
    # ============================================================
    print(f"\n{'='*80}")
    print("STEP 2: PARTITIONING DATA ACROSS HOSPITALS")
    print(f"{'='*80}\n")
    
    partitioner = HospitalDataPartitioner(
        n_hospitals=n_clients,
        partition_strategy=partition_strategy
    )
    
    hospital_data = partitioner.partition_data(X, y)
    partitioner.print_statistics()
    
    # Get client sample sizes for fairness metrics
    client_sample_sizes = [len(hospital_data[i][0]) for i in range(n_clients)]
    
    # ============================================================
    # Step 3: Initialize Privacy Mechanisms
    # ============================================================
    print(f"\n{'='*80}")
    print("STEP 3: INITIALIZING PRIVACY MECHANISMS")
    print(f"{'='*80}\n")
    
    # Differential Privacy
    dp = None
    if dp_enabled:
        dp = DifferentialPrivacy(epsilon=epsilon, delta=delta, clip_norm=clip_norm)
        print(f"✓ Differential Privacy enabled (ε={epsilon}, δ={delta})")
    
    # Secure Aggregation
    secure_agg = None
    if smpc_enabled:
        secure_agg = SecureAggregator(n_clients=n_clients)
        print(f"✓ Secure Aggregation enabled")
    
    # Blockchain Audit Log
    blockchain = FederatedLearningBlockchain()
    print(f"✓ Blockchain audit log initialized")
    
    # ============================================================
    # Step 4: Initialize Global Model
    # ============================================================
    print(f"\n{'='*80}")
    print("STEP 4: INITIALIZING GLOBAL MODEL")
    print(f"{'='*80}\n")
    
    n_features = X.shape[1]
    hidden_layers = config.get('model', {}).get('hidden_layers', [64, 32])
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.3)
    
    global_model = BaselineHealthcareModel(
        input_dim=n_features,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate
    )
    global_model.build_model()
    
    # Fit preprocessor on all data (just for imputer/scaler)
    # This simulates what would happen in practice where preprocessing
    # statistics are shared among clients
    global_model.preprocess_data(X, fit=True)
    
    print("Global model architecture:")
    global_model.model.summary()
    
    # ============================================================
    # Step 5: Simulate Federated Learning Rounds
    # ============================================================
    print(f"\n{'='*80}")
    print("STEP 5: FEDERATED LEARNING TRAINING")
    print(f"{'='*80}\n")
    
    # Split each client's data into train/val
    client_train_data = []
    client_val_data = []
    
    for client_id in range(n_clients):
        X_client, y_client = hospital_data[client_id]
        
        # Simple train/val split
        val_size = int(0.2 * len(X_client))
        if val_size > 0:
            X_train = X_client[:-val_size]
            y_train = y_client[:-val_size]
            X_val = X_client[-val_size:]
            y_val = y_client[-val_size:]
        else:
            X_train, y_train = X_client, y_client
            X_val, y_val = None, None
        
        client_train_data.append((X_train, y_train))
        client_val_data.append((X_val, y_val))
    
    # Track global weights for drift computation
    global_weights_history = []
    
    for round_num in range(1, n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{n_rounds}")
        print(f"{'='*60}\n")
        
        # Store global weights
        global_weights = global_model.model.get_weights()
        global_weights_history.append(global_weights)
        
        # Simulate each client training locally
        client_updates = []
        client_results = []
        
        for client_id in range(n_clients):
            X_train, y_train = client_train_data[client_id]
            X_val, y_val = client_val_data[client_id]
            
            print(f"[Client {client_id}] Training locally...")
            
            # Create local model with global weights
            local_model = BaselineHealthcareModel(
                input_dim=n_features,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            )
            local_model.build_model()
            local_model.model.set_weights(global_weights)
            
            # Train locally
            local_model.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            # Get updated weights
            updated_weights = local_model.model.get_weights()
            
            # Apply Differential Privacy to weight update
            if dp:
                # Compute weight delta
                weight_delta = [uw - gw for uw, gw in zip(updated_weights, global_weights)]
                # Add noise to delta
                noisy_delta = dp.add_noise_to_gradients(weight_delta)
                # Reconstruct weights
                private_weights = [gw + nd for gw, nd in zip(global_weights, noisy_delta)]
                client_updates.append(private_weights)
            else:
                client_updates.append(updated_weights)
            
            # Evaluate local model
            if X_val is not None:
                eval_results = local_model.evaluate(X_val, y_val)
                client_results.append({
                    'client_id': client_id,
                    'accuracy': eval_results['accuracy'],
                    'loss': eval_results['loss'],
                    'n_samples': len(X_train)
                })
                print(f"  Accuracy: {eval_results['accuracy']:.4f}, Loss: {eval_results['loss']:.4f}")
        
        # Secure Aggregation
        print(f"\n[Server] Aggregating client updates...")
        if secure_agg:
            aggregated_weights = secure_agg.secure_aggregate(client_updates)
        else:
            # Simple averaging
            aggregated_weights = []
            for i in range(len(client_updates[0])):
                avg_weight = np.mean([client_update[i] for client_update in client_updates], axis=0)
                aggregated_weights.append(avg_weight)
        
        # Update global model
        global_model.model.set_weights(aggregated_weights)
        print(f"  ✓ Global model updated")
        
        # Evaluate global model on all data
        # Use a test set (last 20% of data)
        test_size = int(0.2 * len(X))
        X_test = X[-test_size:]
        y_test = y[-test_size:]
        
        global_eval = global_model.evaluate(X_test, y_test)
        y_pred = global_model.predict(X_test)
        
        # Compute comprehensive metrics
        model_metrics = metrics_computer.compute_model_performance(
            y_test, (y_pred > 0.5).astype(int).flatten(), y_pred
        )
        
        client_metrics = metrics_computer.compute_client_level_metrics(client_results)
        
        # Track round metrics
        round_metrics = {
            'round': round_num,
            'accuracy': model_metrics['accuracy'],
            'loss': model_metrics.get('loss', None),
            'precision': model_metrics['precision'],
            'recall': model_metrics['recall'],
            'f1_score': model_metrics['f1_score'],
            'n_clients': n_clients,
            'client_mean_accuracy': client_metrics.get('mean_accuracy', 0),
            'client_accuracy_std': client_metrics.get('accuracy_std', 0)
        }
        tracker.track_round(round_num, round_metrics)
        
        print(f"\n[Global Model] Round {round_num} Results:")
        print(f"  Global Accuracy: {model_metrics['accuracy']:.4f}")
        print(f"  Global Loss: {model_metrics.get('loss', 0):.4f}")
        print(f"  Precision: {model_metrics['precision']:.4f}")
        print(f"  Recall: {model_metrics['recall']:.4f}")
        print(f"  F1 Score: {model_metrics['f1_score']:.4f}")
        
        # Log to blockchain
        model_hash = f"model_v{round_num}_{hash(str(aggregated_weights[0].sum()))}"
        blockchain.log_training_round(
            round_number=round_num,
            participating_hospitals=list(range(n_clients)),
            model_hash=model_hash,
            metrics={
                'accuracy': float(model_metrics['accuracy']),
                'loss': float(model_metrics.get('loss', 0))
            },
            aggregation_method="FedAvg with DP and SMPC" if (dp and secure_agg) else "FedAvg"
        )
    
    # ============================================================
    # Step 6: Compute Final Comprehensive Metrics
    # ============================================================
    print(f"\n{'='*80}")
    print("STEP 6: COMPUTING COMPREHENSIVE METRICS")
    print(f"{'='*80}\n")
    
    # Final model evaluation
    final_eval = global_model.evaluate(X_test, y_test)
    y_pred_final = global_model.predict(X_test)
    
    final_model_metrics = metrics_computer.compute_model_performance(
        y_test, (y_pred_final > 0.5).astype(int).flatten(), y_pred_final
    )
    
    # Federated learning metrics
    fl_metrics = metrics_computer.compute_federated_learning_metrics(
        tracker.rounds_history,
        convergence_threshold=metrics_config.get('convergence_threshold', 0.01),
        convergence_window=metrics_config.get('convergence_window', 3)
    )
    
    # Client drift metrics (using last round)
    if len(global_weights_history) > 1:
        drift_metrics = metrics_computer.compute_client_drift_metrics(
            client_updates,
            global_weights_history[-1]
        )
    else:
        drift_metrics = {}
    
    # Privacy metrics
    privacy_metrics = metrics_computer.compute_privacy_metrics(
        epsilon=epsilon,
        delta=delta,
        n_rounds=n_rounds,
        accuracy=final_model_metrics['accuracy'],
        loss=final_model_metrics.get('loss')
    )
    
    # SMPC metrics (placeholders)
    smpc_metrics = metrics_computer.compute_smpc_metrics(
        n_clients=n_clients,
        n_rounds=n_rounds,
        model_size=sum([w.size for w in global_weights])
    )
    
    # Fairness metrics
    fairness_metrics = metrics_computer.compute_fairness_metrics(
        client_results,
        client_sample_sizes
    )
    
    # Compile all metrics
    all_metrics = {
        'dataset_info': dataset_info,
        'configuration': {
            'n_clients': n_clients,
            'n_rounds': n_rounds,
            'local_epochs': local_epochs,
            'partition_strategy': partition_strategy,
            'privacy': {
                'differential_privacy_enabled': dp_enabled,
                'epsilon': epsilon if dp_enabled else None,
                'smpc_enabled': smpc_enabled
            }
        },
        'model_performance': final_model_metrics,
        'federated_learning_metrics': fl_metrics,
        'client_drift_metrics': drift_metrics,
        'privacy_metrics': privacy_metrics,
        'smpc_metrics': smpc_metrics,
        'fairness_metrics': fairness_metrics
    }
    
    tracker.set_final_metrics(all_metrics)
    
    # Print summary
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    print(f"\nModel Performance:")
    print(f"  Global Accuracy: {final_model_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_model_metrics['precision']:.4f}")
    print(f"  Recall: {final_model_metrics['recall']:.4f}")
    print(f"  F1 Score: {final_model_metrics['f1_score']:.4f}")
    
    print(f"\nFederated Learning:")
    print(f"  Total Rounds: {fl_metrics['total_rounds']}")
    print(f"  Rounds to Convergence: {fl_metrics['rounds_to_convergence']}")
    print(f"  Communication Cost: {fl_metrics['communication_cost']['total_messages']} messages")
    print(f"  Final Accuracy: {fl_metrics['final_accuracy']:.4f}")
    print(f"  Accuracy Improvement: {fl_metrics['accuracy_improvement']:.4f}")
    
    print(f"\nPrivacy:")
    print(f"  Differential Privacy: {'Enabled' if dp_enabled else 'Disabled'}")
    if dp_enabled:
        print(f"    Epsilon (per round): {epsilon}")
        print(f"    Total Epsilon: {privacy_metrics['privacy_budget']['total_epsilon']:.2f}")
    print(f"  SMPC: {'Enabled' if smpc_enabled else 'Disabled'}")
    
    print(f"\nFairness:")
    print(f"  Client Accuracy Variance: {fairness_metrics['client_accuracy_variance']:.6f}")
    print(f"  Fairness Score: {fairness_metrics['fairness_score']:.4f}")
    
    # ============================================================
    # Step 7: Save Outputs
    # ============================================================
    print(f"\n{'='*80}")
    print("STEP 7: SAVING OUTPUTS")
    print(f"{'='*80}\n")
    
    # Save all metrics and plots
    tracker.save_all()
    
    # Save blockchain audit
    blockchain.export_to_json("federated_learning_audit.json")
    print("✓ Blockchain audit saved to: federated_learning_audit.json")
    
    # Plot client accuracy distribution
    if client_results:
        client_accuracies = [r['accuracy'] for r in client_results]
        tracker.plot_client_accuracy_variance(client_accuracies)
    
    print(f"\n{'='*80}")
    print("FEDERATED LEARNING COMPLETE!")
    print(f"{'='*80}\n")
    
    print("✓ All privacy-preserving mechanisms applied")
    print("✓ Comprehensive metrics computed and saved")
    print("✓ Visualizations generated")
    print(f"✓ Results saved to: {output_dir}/")
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Federated Learning with Benchmark Evaluation"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['benchmark', 'srm', 'csv'],
        default=None,
        help="Dataset type to use (overrides config.yaml)"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help="Path to configuration file"
    )
    parser.add_argument(
        '--clients',
        type=int,
        default=None,
        help="Number of clients/hospitals (overrides config.yaml)"
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=None,
        help="Number of federated learning rounds (overrides config.yaml)"
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['iid', 'non_iid', 'class_imbalance'],
        default=None,
        help="Data partitioning strategy (overrides config.yaml)"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help="Path to CSV dataset (for SRM dataset)"
    )
    parser.add_argument(
        '--schema',
        type=str,
        default=None,
        help="Path to schema JSON file (for SRM dataset)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments if provided
    if args.clients is not None:
        config.setdefault('federated_learning', {})['n_clients'] = args.clients
    
    if args.rounds is not None:
        config.setdefault('federated_learning', {})['n_rounds'] = args.rounds
    
    if args.strategy is not None:
        config.setdefault('federated_learning', {})['partition_strategy'] = args.strategy
    
    if args.csv is not None:
        config.setdefault('dataset', {}).setdefault('srm', {})['path'] = args.csv
    
    if args.schema is not None:
        # Load schema from JSON file
        import json
        with open(args.schema, 'r') as f:
            schema = json.load(f)
        config.setdefault('dataset', {}).setdefault('srm', {})['schema'] = schema
    
    # Welcome message
    print("\n" + "="*80)
    print("PRIVACY-PRESERVING FEDERATED LEARNING")
    print("Benchmark Dataset Evaluation & Comprehensive Metrics")
    print("="*80)
    
    # Load data
    X, y, dataset_info = load_data(config, dataset_type=args.dataset)
    
    # Run federated learning
    simulate_federated_learning(X, y, config, dataset_info)
    
    # Final message
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE!")
    print("="*80)
    print("\nThis prototype demonstrates privacy-preserving federated learning.")
    print("Production systems require additional security hardening.")
    print("\nCheck the 'outputs/' directory for:")
    print("  - metrics.json: Complete metrics summary")
    print("  - rounds_history.csv: Per-round training history")
    print("  - plots/: Visualization plots")
    print()


if __name__ == "__main__":
    main()
