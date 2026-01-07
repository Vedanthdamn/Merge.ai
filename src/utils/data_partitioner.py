"""
Hospital-Based Data Partitioning for Distributed Learning

This module simulates how healthcare data would be distributed across
multiple hospitals in a real federated learning scenario.

Key Concepts:
- Non-IID (Non-Independent and Identically Distributed) data:
  Each hospital serves different patient populations with unique characteristics
- Data locality: Patient data never leaves the hospital
- Simulation: This code demonstrates partitioning without requiring real multi-hospital data

Example non-IID scenarios:
1. Hospital A: Urban, treats more cardiac patients
2. Hospital B: Rural, treats more diabetes patients
3. Hospital C: Specialty cancer center
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class HospitalDataPartitioner:
    """
    Simulates partitioning of healthcare data across multiple hospitals.
    
    In real federated learning:
    - Each hospital has its own local data
    - Data distributions differ (non-IID)
    - Data never leaves hospital premises
    
    This simulator demonstrates these properties using a single dataset.
    """
    
    def __init__(self, n_hospitals=3, partition_strategy='iid'):
        """
        Initialize the data partitioner.
        
        Args:
            n_hospitals (int): Number of hospitals to simulate
            partition_strategy (str): How to partition data
                - 'iid': Independent and Identically Distributed (equal random splits)
                - 'non_iid': Simulates different patient demographics per hospital
                - 'class_imbalance': Each hospital has different outcome distributions
        """
        self.n_hospitals = n_hospitals
        self.partition_strategy = partition_strategy
        self.hospital_data = {}
        self.hospital_metadata = {}
    
    def partition_iid(self, X, y) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data equally and randomly across hospitals (IID).
        
        This represents an idealized scenario where each hospital
        has similar patient populations. Rarely true in practice.
        
        Args:
            X: Feature data
            y: Target labels
            
        Returns:
            dict: {hospital_id: (X_hospital, y_hospital)}
        """
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        # Split indices into roughly equal parts
        splits = np.array_split(indices, self.n_hospitals)
        
        hospital_data = {}
        for hospital_id, hospital_indices in enumerate(splits):
            X_hospital = X[hospital_indices]
            y_hospital = y[hospital_indices]
            hospital_data[hospital_id] = (X_hospital, y_hospital)
            
            print(f"Hospital {hospital_id}: {len(hospital_indices)} patients, "
                  f"positive rate: {np.mean(y_hospital):.2%}")
        
        return hospital_data
    
    def partition_non_iid(self, X, y, 
                          skew_factor=0.5) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data with non-IID distribution (simulates real-world).
        
        Non-IID characteristics:
        1. Different sample sizes (some hospitals are larger)
        2. Different feature distributions (patient demographics vary)
        3. Different label distributions (disease prevalence varies)
        
        Args:
            X: Feature data
            y: Target labels
            skew_factor (float): Degree of non-IID-ness (0=IID, 1=highly skewed)
            
        Returns:
            dict: {hospital_id: (X_hospital, y_hospital)}
            
        Example: Hospital in elderly community sees more age-related conditions
        """
        n_samples = len(X)
        
        # Create non-uniform sample sizes
        # Some hospitals are larger than others
        base_sizes = np.ones(self.n_hospitals) / self.n_hospitals
        size_variation = np.random.dirichlet(np.ones(self.n_hospitals) * (1 / skew_factor))
        hospital_sizes = (base_sizes + skew_factor * size_variation) / (1 + skew_factor)
        hospital_sizes = (hospital_sizes / hospital_sizes.sum() * n_samples).astype(int)
        
        # Ensure we use all samples
        hospital_sizes[-1] = n_samples - hospital_sizes[:-1].sum()
        
        hospital_data = {}
        current_idx = 0
        
        for hospital_id in range(self.n_hospitals):
            size = hospital_sizes[hospital_id]
            
            # For non-IID, prefer certain labels at certain hospitals
            # Simulates different disease prevalence across regions
            if np.random.rand() < skew_factor:
                # This hospital has preference for one class
                preferred_class = hospital_id % 2
                class_indices = np.where(y == preferred_class)[0]
                other_indices = np.where(y != preferred_class)[0]
                
                # Sample more from preferred class
                n_preferred = int(size * (0.5 + skew_factor * 0.3))
                n_other = size - n_preferred
                
                preferred_samples = np.random.choice(
                    class_indices, 
                    min(n_preferred, len(class_indices)), 
                    replace=False
                )
                other_samples = np.random.choice(
                    other_indices, 
                    min(n_other, len(other_indices)), 
                    replace=False
                )
                
                hospital_indices = np.concatenate([preferred_samples, other_samples])
            else:
                # Random sampling for this hospital
                end_idx = min(current_idx + size, n_samples)
                hospital_indices = np.arange(current_idx, end_idx)
                current_idx = end_idx
            
            X_hospital = X[hospital_indices]
            y_hospital = y[hospital_indices]
            hospital_data[hospital_id] = (X_hospital, y_hospital)
            
            print(f"Hospital {hospital_id}: {len(hospital_indices)} patients, "
                  f"positive rate: {np.mean(y_hospital):.2%}")
        
        return hospital_data
    
    def partition_by_class_imbalance(self, X, y) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition with extreme class imbalance across hospitals.
        
        Simulates scenarios like:
        - Specialty hospitals (e.g., cancer center has mostly positive cases)
        - General hospitals (balanced case distribution)
        
        Args:
            X: Feature data
            y: Target labels
            
        Returns:
            dict: {hospital_id: (X_hospital, y_hospital)}
        """
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
        
        hospital_data = {}
        
        for hospital_id in range(self.n_hospitals):
            # Each hospital gets different ratio of positive/negative cases
            if hospital_id == 0:
                # Hospital 0: Specialty center (mostly positive cases)
                ratio = 0.8
                n_positive = int(len(positive_indices) * 0.5)
                n_negative = int(n_positive * (1 - ratio) / ratio)
            elif hospital_id == 1:
                # Hospital 1: General hospital (balanced)
                ratio = 0.5
                n_positive = int(len(positive_indices) * 0.3)
                n_negative = n_positive
            else:
                # Hospital 2+: Preventive care (mostly negative cases)
                ratio = 0.2
                n_positive = int(len(positive_indices) * 0.2 / (self.n_hospitals - 2))
                n_negative = int(n_positive * (1 - ratio) / ratio)
            
            # Sample indices
            pos_sample = np.random.choice(
                positive_indices, 
                min(n_positive, len(positive_indices)), 
                replace=False
            )
            neg_sample = np.random.choice(
                negative_indices, 
                min(n_negative, len(negative_indices)), 
                replace=False
            )
            
            hospital_indices = np.concatenate([pos_sample, neg_sample])
            X_hospital = X[hospital_indices]
            y_hospital = y[hospital_indices]
            
            hospital_data[hospital_id] = (X_hospital, y_hospital)
            
            print(f"Hospital {hospital_id}: {len(hospital_indices)} patients, "
                  f"positive rate: {np.mean(y_hospital):.2%}")
        
        return hospital_data
    
    def partition_data(self, X, y) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Partition data according to selected strategy.
        
        Args:
            X: Feature data
            y: Target labels
            
        Returns:
            dict: {hospital_id: (X_hospital, y_hospital)}
        """
        print(f"\n{'='*60}")
        print(f"Partitioning data for {self.n_hospitals} hospitals")
        print(f"Strategy: {self.partition_strategy}")
        print(f"{'='*60}\n")
        
        if self.partition_strategy == 'iid':
            self.hospital_data = self.partition_iid(X, y)
        elif self.partition_strategy == 'non_iid':
            self.hospital_data = self.partition_non_iid(X, y)
        elif self.partition_strategy == 'class_imbalance':
            self.hospital_data = self.partition_by_class_imbalance(X, y)
        else:
            raise ValueError(f"Unknown partition strategy: {self.partition_strategy}")
        
        # Store metadata
        for hospital_id, (X_hosp, y_hosp) in self.hospital_data.items():
            self.hospital_metadata[hospital_id] = {
                'n_samples': len(X_hosp),
                'n_features': X_hosp.shape[1],
                'positive_rate': float(np.mean(y_hosp)),
                'negative_rate': float(1 - np.mean(y_hosp))
            }
        
        return self.hospital_data
    
    def get_hospital_data(self, hospital_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get data for a specific hospital."""
        if hospital_id not in self.hospital_data:
            raise ValueError(f"Hospital {hospital_id} not found")
        return self.hospital_data[hospital_id]
    
    def get_all_hospital_ids(self) -> List[int]:
        """Get list of all hospital IDs."""
        return list(self.hospital_data.keys())
    
    def print_statistics(self):
        """Print statistics about the partitioned data."""
        print(f"\n{'='*60}")
        print("Hospital Data Distribution Summary")
        print(f"{'='*60}")
        
        for hospital_id, metadata in self.hospital_metadata.items():
            print(f"\nHospital {hospital_id}:")
            print(f"  Samples: {metadata['n_samples']}")
            print(f"  Features: {metadata['n_features']}")
            print(f"  Positive cases: {metadata['positive_rate']:.2%}")
            print(f"  Negative cases: {metadata['negative_rate']:.2%}")


def main():
    """
    Demonstration of hospital data partitioning.
    """
    print("Hospital-Based Data Partitioning Demonstration")
    print("=" * 60)
    
    # Create mock healthcare data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    print(f"Mock dataset: {n_samples} patients, {n_features} features")
    
    # Demonstrate different partitioning strategies
    strategies = ['iid', 'non_iid', 'class_imbalance']
    
    for strategy in strategies:
        print(f"\n\n{'#'*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'#'*60}")
        
        partitioner = HospitalDataPartitioner(
            n_hospitals=3,
            partition_strategy=strategy
        )
        
        hospital_data = partitioner.partition_data(X, y)
        partitioner.print_statistics()


if __name__ == "__main__":
    main()
