"""
Dataset Loader Abstraction for Federated Learning

This module provides a clean abstraction for loading various datasets:
1. Benchmark datasets (scikit-learn built-in datasets)
2. Custom CSV/JSON datasets with schema mapping
3. SRM Hospital dataset with configurable schema

The loader handles:
- Data loading and preprocessing
- Schema mapping for different data sources
- Train/test splitting
- Compatibility with federated learning pipelines
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import os


class DatasetLoader:
    """
    Unified dataset loader supporting multiple data sources.
    
    Supports:
    - Benchmark datasets (diabetes, breast_cancer)
    - Custom CSV/JSON with schema configuration
    - SRM hospital dataset adapter
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the dataset loader.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def load_benchmark_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a benchmark dataset from scikit-learn.
        
        Args:
            dataset_name: Name of the dataset ('diabetes', 'breast_cancer')
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        print(f"Loading benchmark dataset: {dataset_name}")
        
        if dataset_name == 'diabetes':
            # Diabetes dataset: regression -> convert to binary classification
            data = load_diabetes()
            X = data.data
            y = data.target
            # Convert to binary: above/below median
            y = (y > np.median(y)).astype(int)
            print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"  Binary classes: {np.unique(y, return_counts=True)}")
            
        elif dataset_name == 'breast_cancer':
            # Breast cancer dataset: already binary classification
            data = load_breast_cancer()
            X = data.data
            y = data.target
            print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"  Classes: {np.unique(y, return_counts=True)}")
            
        else:
            raise ValueError(
                f"Unknown benchmark dataset: {dataset_name}. "
                f"Supported: 'diabetes', 'breast_cancer'"
            )
        
        return X, y
    
    def load_csv_dataset(self, 
                        filepath: str, 
                        schema_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a custom CSV dataset with optional schema mapping.
        
        Args:
            filepath: Path to CSV file
            schema_config: Schema configuration dict with:
                - feature_columns: List of feature column names
                - target_column: Name of target column
                - patient_id_column: Optional ID column to exclude
                
        Returns:
            Tuple of (X, y)
        """
        print(f"Loading CSV dataset: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"  Loaded CSV with shape: {df.shape}")
        
        # Apply schema configuration
        if schema_config:
            feature_cols = schema_config.get('feature_columns', None)
            target_col = schema_config.get('target_column', 'outcome')
            id_col = schema_config.get('patient_id_column', None)
            
            # Remove ID column if specified
            if id_col and id_col in df.columns:
                df = df.drop(columns=[id_col])
                print(f"  Dropped ID column: {id_col}")
            
            # Select feature columns if specified
            if feature_cols:
                # Ensure target column is not in features
                if target_col in feature_cols:
                    feature_cols = [c for c in feature_cols if c != target_col]
                X = df[feature_cols].values
            else:
                # Use all columns except target
                X = df.drop(columns=[target_col]).values
            
            y = df[target_col].values
            
        else:
            # Default: last column is target, rest are features
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution: {np.unique(y, return_counts=True)}")
        
        return X, y
    
    def load_json_dataset(self, 
                         filepath: str, 
                         schema_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a custom JSON dataset with optional schema mapping.
        
        Args:
            filepath: Path to JSON file
            schema_config: Schema configuration (same as CSV)
            
        Returns:
            Tuple of (X, y)
        """
        print(f"Loading JSON dataset: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        df = pd.read_json(filepath)
        print(f"  Loaded JSON with shape: {df.shape}")
        
        # Same processing as CSV
        if schema_config:
            feature_cols = schema_config.get('feature_columns', None)
            target_col = schema_config.get('target_column', 'outcome')
            id_col = schema_config.get('patient_id_column', None)
            
            if id_col and id_col in df.columns:
                df = df.drop(columns=[id_col])
            
            if feature_cols:
                if target_col in feature_cols:
                    feature_cols = [c for c in feature_cols if c != target_col]
                X = df[feature_cols].values
            else:
                X = df.drop(columns=[target_col]).values
            
            y = df[target_col].values
        else:
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution: {np.unique(y, return_counts=True)}")
        
        return X, y
    
    def load_dataset(self, 
                    dataset_name: str, 
                    dataset_path: Optional[str] = None,
                    schema_config: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unified dataset loading function.
        
        This is the main entry point for loading any dataset.
        
        Args:
            dataset_name: Name of dataset. Options:
                - 'diabetes', 'breast_cancer' (benchmark datasets)
                - 'csv' (custom CSV file)
                - 'json' (custom JSON file)
                - 'srm' (SRM hospital dataset)
            dataset_path: Path to dataset file (required for csv/json/srm)
            schema_config: Schema configuration for custom datasets
            
        Returns:
            Tuple of (X, y) where X is features, y is binary target
            
        Examples:
            # Load benchmark dataset
            X, y = loader.load_dataset('diabetes')
            
            # Load custom CSV with schema
            X, y = loader.load_dataset('csv', 'data/hospital.csv', 
                                      {'target_column': 'disease'})
            
            # Load SRM dataset
            X, y = loader.load_dataset('srm', 'data/srm_hospital_data.csv',
                                      srm_schema_config)
        """
        print(f"\n{'='*60}")
        print(f"Dataset Loader")
        print(f"{'='*60}")
        
        # Benchmark datasets
        if dataset_name in ['diabetes', 'breast_cancer']:
            X, y = self.load_benchmark_dataset(dataset_name)
        
        # Custom CSV
        elif dataset_name == 'csv':
            if not dataset_path:
                raise ValueError("dataset_path required for CSV datasets")
            X, y = self.load_csv_dataset(dataset_path, schema_config)
        
        # Custom JSON
        elif dataset_name == 'json':
            if not dataset_path:
                raise ValueError("dataset_path required for JSON datasets")
            X, y = self.load_json_dataset(dataset_path, schema_config)
        
        # SRM hospital dataset (CSV with specific schema)
        elif dataset_name == 'srm':
            if not dataset_path:
                raise ValueError("dataset_path required for SRM dataset")
            X, y = self.load_csv_dataset(dataset_path, schema_config)
        
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: 'diabetes', 'breast_cancer', 'csv', 'json', 'srm'"
            )
        
        print(f"{'='*60}\n")
        
        return X, y
    
    def prepare_federated_data(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              test_size: float = 0.2,
                              val_size: float = 0.1) -> Dict:
        """
        Prepare data for federated learning.
        
        Splits data into train/val/test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Fraction for test set
            val_size: Fraction of training set for validation
            
        Returns:
            Dict with train/val/test splits
        """
        print(f"Preparing federated learning data splits...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        
        # Second split: train vs val
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, 
                random_state=self.random_seed, stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        print(f"  Train: {X_train.shape[0]} samples")
        if X_val is not None:
            print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }


# ============================================================
# Standalone utility functions for easy imports
# ============================================================

def load_benchmark_dataset(n_samples: int = 1000, 
                          n_features: int = 10, 
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or create a benchmark dataset with specified parameters.
    
    This function creates synthetic data matching the demo_integration.py mock dataset.
    For real benchmark datasets, use DatasetLoader class.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X is features and y is binary labels
    """
    np.random.seed(seed)
    
    # Generate synthetic healthcare-like data
    X = np.random.randn(n_samples, n_features)
    
    # Generate binary labels with ~50% positive rate
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    return X, y


def split_train_test(X: np.ndarray, 
                    y: np.ndarray, 
                    test_size: float = 0.2, 
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    Args:
        X: Feature data
        y: Target labels
        test_size: Fraction of data for testing (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split as sklearn_split
    
    X_train, X_test, y_train, y_test = sklearn_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def make_clients(X_train: np.ndarray,
                y_train: np.ndarray,
                num_clients: int = 3,
                strategy: str = "non_iid",
                seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """
    Partition training data across multiple clients.
    
    Reuses the HospitalDataPartitioner logic for consistency.
    
    Args:
        X_train: Training features
        y_train: Training labels
        num_clients: Number of clients to create
        strategy: Partitioning strategy ('iid', 'non_iid', 'class_imbalance')
        seed: Random seed for reproducibility
        
    Returns:
        List of client data dictionaries: [{"X": X_client, "y": y_client}, ...]
    """
    # Import locally to avoid circular import issues
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_partitioner import HospitalDataPartitioner
    
    np.random.seed(seed)
    
    # Use the existing partitioner
    partitioner = HospitalDataPartitioner(
        n_hospitals=num_clients,
        partition_strategy=strategy
    )
    
    hospital_data = partitioner.partition_data(X_train, y_train)
    
    # Convert to list of dicts format
    clients = []
    for client_id in range(num_clients):
        X_client, y_client = hospital_data[client_id]
        clients.append({"X": X_client, "y": y_client})
    
    return clients


def main():
    """Test the dataset loader with standalone functions."""
    print("Testing Dataset Loader")
    print("="*60)
    
    # Test 1: Standalone functions
    print("\n1. Testing Standalone Functions")
    print("-"*60)
    
    # Load benchmark dataset
    print("\nLoading benchmark dataset...")
    X, y = load_benchmark_dataset(n_samples=1000, n_features=10, seed=42)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Label distribution: {np.unique(y, return_counts=True)}")
    
    # Split train/test
    print("\nSplitting train/test...")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, seed=42)
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    
    # Make clients
    print("\nCreating clients (non-IID)...")
    clients = make_clients(X_train, y_train, num_clients=3, strategy="non_iid", seed=42)
    print(f"Created {len(clients)} clients:")
    for i, client in enumerate(clients):
        print(f"  Client {i}: X={client['X'].shape}, y={client['y'].shape}, "
              f"pos_rate={np.mean(client['y']):.2%}")
    
    # Test 2: Class-based loader
    print("\n2. Testing Class-Based Loader")
    print("-"*60)
    
    loader = DatasetLoader(random_seed=42)
    
    for dataset_name in ['diabetes', 'breast_cancer']:
        print(f"\nLoading {dataset_name}...")
        X, y = loader.load_dataset(dataset_name)
        print(f"Shape: X={X.shape}, y={y.shape}")
    
    print("\n" + "="*60)
    print("Dataset loader test complete!")


if __name__ == "__main__":
    main()
