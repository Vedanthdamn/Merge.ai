"""
SRM Global Hospital Dataset Adapter Template

This template provides a reusable adapter for SRM Global Hospital's dataset.
When SRM provides their CSV file, you only need to:
1. Update the dataset path in config.yaml
2. Adjust column mappings in the schema_config if needed

The rest of the pipeline will work without any code changes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import os
import json
import argparse


class SRMDatasetAdapter:
    """
    Adapter for SRM Global Hospital dataset.
    
    This template allows plug-and-play integration of SRM's dataset.
    Simply configure the schema mapping and the entire pipeline works.
    
    Usage:
        # In config.yaml, set:
        dataset:
          type: 'srm'
          srm:
            path: 'data/srm_hospital_data.csv'
            schema:
              feature_columns: ['age', 'gender', 'bp_systolic', ...]
              target_column: 'diagnosis'
              patient_id_column: 'patient_id'
        
        # Run normally:
        python demo_integration.py --dataset srm
    """
    
    # Default SRM schema (update this when you receive actual data)
    DEFAULT_SCHEMA = {
        'feature_columns': [
            'age',
            'sex',
            'systolic_bp',
            'diastolic_bp',
            'cholesterol',
            'fasting_glucose',
            'bmi',
            'heart_rate',
            'smoking',
            'family_history'
        ],
        'target_column': 'outcome',
        'patient_id_column': 'patient_id'  # Optional: will be dropped if present
    }
    
    def __init__(self, schema_config: Optional[Dict] = None):
        """
        Initialize SRM dataset adapter.
        
        Args:
            schema_config: Optional schema configuration.
                          If not provided, uses DEFAULT_SCHEMA.
        """
        self.schema_config = schema_config or self.DEFAULT_SCHEMA
        
    def load_srm_dataset(self, filepath: str) -> tuple:
        """
        Load SRM hospital dataset with schema mapping.
        
        Args:
            filepath: Path to SRM CSV file
            
        Returns:
            Tuple of (X, y, metadata)
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If required columns are missing
        """
        print(f"\n{'='*60}")
        print("SRM Global Hospital Dataset Adapter")
        print(f"{'='*60}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"SRM dataset not found at: {filepath}\n"
                f"Please place SRM's CSV file at this location."
            )
        
        print(f"Loading SRM dataset from: {filepath}")
        
        # Load CSV
        df = pd.read_csv(filepath)
        print(f"  Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Extract schema configuration
        feature_cols = self.schema_config.get('feature_columns')
        target_col = self.schema_config.get('target_column', 'outcome')
        id_col = self.schema_config.get('patient_id_column', None)
        
        # Validate required columns exist
        missing_cols = []
        if feature_cols:
            missing_cols.extend([col for col in feature_cols if col not in df.columns])
        if target_col not in df.columns:
            missing_cols.append(target_col)
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns in SRM dataset: {missing_cols}\n"
                f"Available columns: {df.columns.tolist()}\n"
                f"Please update schema_config in config.yaml to match SRM's column names."
            )
        
        # Remove patient ID column if present (privacy protection)
        if id_col and id_col in df.columns:
            df = df.drop(columns=[id_col])
            print(f"  Dropped patient ID column: {id_col} (privacy protection)")
        
        # Extract features and target
        if feature_cols:
            # Use specified feature columns
            X = df[feature_cols].values
            print(f"  Selected {len(feature_cols)} feature columns")
        else:
            # Use all columns except target
            X = df.drop(columns=[target_col]).values
            print(f"  Using all columns except target as features")
        
        y = df[target_col].values
        
        # Metadata about the dataset
        metadata = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_names': feature_cols if feature_cols else df.drop(columns=[target_col]).columns.tolist(),
            'target_name': target_col,
            'class_distribution': {
                'unique_values': np.unique(y).tolist(),
                'counts': {str(val): int(count) for val, count in zip(*np.unique(y, return_counts=True))}
            },
            'missing_values': {
                'total': int(pd.DataFrame(X).isnull().sum().sum()),
                'per_feature': pd.DataFrame(X).isnull().sum().to_dict()
            }
        }
        
        print(f"\n  Dataset Summary:")
        print(f"    Samples: {metadata['n_samples']}")
        print(f"    Features: {metadata['n_features']}")
        print(f"    Target: {target_col}")
        print(f"    Class distribution: {metadata['class_distribution']['counts']}")
        print(f"    Missing values: {metadata['missing_values']['total']}")
        
        print(f"{'='*60}\n")
        
        return X, y, metadata
    
    def validate_schema(self, filepath: str) -> Dict:
        """
        Validate that the SRM dataset matches the configured schema.
        
        Use this function to check compatibility before running training.
        
        Args:
            filepath: Path to SRM CSV file
            
        Returns:
            Dict with validation results
        """
        print("Validating SRM dataset schema...")
        
        if not os.path.exists(filepath):
            return {
                'valid': False,
                'error': f"File not found: {filepath}"
            }
        
        try:
            df = pd.read_csv(filepath)
            
            feature_cols = self.schema_config.get('feature_columns', [])
            target_col = self.schema_config.get('target_column', 'outcome')
            
            # Check for missing columns
            required_cols = feature_cols + [target_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    'valid': False,
                    'error': f"Missing columns: {missing_cols}",
                    'available_columns': df.columns.tolist(),
                    'required_columns': required_cols
                }
            
            # Check target column is binary
            unique_targets = df[target_col].unique()
            if len(unique_targets) != 2:
                return {
                    'valid': False,
                    'error': f"Target column '{target_col}' must be binary, found {len(unique_targets)} unique values: {unique_targets}"
                }
            
            return {
                'valid': True,
                'message': "Schema validation passed",
                'samples': len(df),
                'features': len(feature_cols) if feature_cols else len(df.columns) - 1,
                'available_columns': df.columns.tolist()
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation failed: {str(e)}"
            }


# ============================================================
# Standalone utility functions for easy imports
# ============================================================

def load_srm_csv(csv_path: str, 
                schema_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SRM hospital dataset from CSV with optional schema configuration.
    
    Args:
        csv_path: Path to SRM CSV file
        schema_path: Optional path to JSON schema configuration file
        
    Returns:
        Tuple of (X, y) where X is features and y is binary labels
        
    Raises:
        FileNotFoundError: If CSV or schema file doesn't exist
        ValueError: If schema validation fails
    """
    # Load schema if provided
    if schema_path:
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema_config = json.load(f)
    else:
        schema_config = None
    
    # Use the adapter class
    adapter = SRMDatasetAdapter(schema_config=schema_config)
    
    # Load dataset
    X, y, metadata = adapter.load_srm_dataset(csv_path)
    
    return X, y


def validate_schema(df: pd.DataFrame, 
                   schema: Dict) -> Dict:
    """
    Validate that a DataFrame matches the expected schema.
    
    Args:
        df: Pandas DataFrame to validate
        schema: Schema dictionary with 'feature_columns', 'target_column', etc.
        
    Returns:
        Dict with validation results: {'valid': bool, 'error': str or None}
    """
    feature_cols = schema.get('feature_columns', [])
    target_col = schema.get('target_column', 'outcome')
    
    # Check for missing columns
    required_cols = feature_cols + [target_col] if feature_cols else [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return {
            'valid': False,
            'error': f"Missing columns: {missing_cols}",
            'available_columns': df.columns.tolist()
        }
    
    # Check target column is binary or convertible to binary
    unique_targets = df[target_col].dropna().unique()
    if len(unique_targets) > 2:
        return {
            'valid': False,
            'error': f"Target column '{target_col}' must be binary, found {len(unique_targets)} unique values"
        }
    
    return {
        'valid': True,
        'message': "Schema validation passed",
        'samples': len(df),
        'features': len(feature_cols) if feature_cols else len(df.columns) - 1
    }


def preprocess_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess missing values in a DataFrame.
    
    Simple strategy:
    - Numeric columns: Fill with median
    - Categorical columns: Fill with mode
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values filled
    """
    df = df.copy()
    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                # Numeric: fill with median
                df[col].fillna(df[col].median(), inplace=True)
            else:
                # Categorical: fill with mode
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col].fillna(mode_value[0], inplace=True)
    
    return df


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to binary format (0/1).
    
    Handles:
    - Already binary (0/1): No change
    - Binary strings/categories: Convert to 0/1
    - Multi-class: Not supported, raises error
    
    Args:
        y: Label array
        
    Returns:
        Binary encoded labels (0/1)
        
    Raises:
        ValueError: If labels are not binary
    """
    unique_vals = np.unique(y)
    
    if len(unique_vals) != 2:
        raise ValueError(
            f"Labels must be binary. Found {len(unique_vals)} unique values: {unique_vals}"
        )
    
    # Check if already 0/1
    if set(unique_vals) == {0, 1}:
        return y.astype(int)
    
    # Map to 0/1
    # Lower/first value -> 0, higher/second value -> 1
    sorted_vals = sorted(unique_vals)
    mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
    
    encoded = np.array([mapping[val] for val in y])
    
    print(f"Encoded labels: {sorted_vals[0]} -> 0, {sorted_vals[1]} -> 1")
    
    return encoded


# Example schema configurations for different scenarios
EXAMPLE_SCHEMAS = {
    'default': {
        'feature_columns': ['age', 'sex', 'systolic_bp', 'diastolic_bp', 
                          'cholesterol', 'fasting_glucose', 'bmi', 
                          'heart_rate', 'smoking', 'family_history'],
        'target_column': 'outcome',
        'patient_id_column': 'patient_id'
    },
    
    'cardiac': {
        'feature_columns': ['age', 'gender', 'chest_pain_type', 'resting_bp',
                          'cholesterol', 'fasting_blood_sugar', 'ecg_results',
                          'max_heart_rate', 'exercise_angina', 'st_depression'],
        'target_column': 'heart_disease',
        'patient_id_column': 'patient_id'
    },
    
    'diabetes': {
        'feature_columns': ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                          'insulin', 'bmi', 'diabetes_pedigree', 'age'],
        'target_column': 'diabetes_diagnosis',
        'patient_id_column': 'patient_id'
    }
}


def main():
    """
    Test the SRM dataset adapter with CLI support.
    """
    parser = argparse.ArgumentParser(
        description="SRM Hospital Dataset Adapter - Load and validate SRM datasets"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help="Path to SRM CSV file"
    )
    parser.add_argument(
        '--schema',
        type=str,
        default=None,
        help="Path to JSON schema configuration file"
    )
    
    args = parser.parse_args()
    
    print("SRM Dataset Adapter")
    print("="*60)
    
    # If CSV path provided, load and display dataset
    if args.csv:
        print(f"\nLoading dataset: {args.csv}")
        if args.schema:
            print(f"Using schema: {args.schema}")
        
        try:
            X, y = load_srm_csv(args.csv, args.schema)
            
            print("\n" + "="*60)
            print("DATASET SUMMARY")
            print("="*60)
            print(f"\nShape: {X.shape}")
            print(f"  Samples: {X.shape[0]}")
            print(f"  Features: {X.shape[1]}")
            
            print(f"\nFeature columns: {X.shape[1]} features")
            
            print(f"\nClass distribution:")
            unique, counts = np.unique(y, return_counts=True)
            for val, count in zip(unique, counts):
                print(f"  Class {val}: {count} ({count/len(y)*100:.1f}%)")
            
            print("\n" + "="*60)
            print("Dataset loaded successfully!")
            print("="*60)
            
        except Exception as e:
            print(f"\nError loading dataset: {e}")
            return
    
    else:
        # Show help and examples
        print("\nUsage Examples:")
        print("-"*60)
        print("\n1. Load SRM dataset with default schema:")
        print("   python src/utils/srm_dataset_adapter.py --csv data/srm.csv")
        
        print("\n2. Load with custom schema:")
        print("   python src/utils/srm_dataset_adapter.py --csv data/srm.csv --schema config/srm_schema.json")
        
        print("\n3. Available example schemas:")
        print("-"*60)
        for name, schema in EXAMPLE_SCHEMAS.items():
            print(f"\n{name.upper()} schema:")
            print(f"  Features: {len(schema['feature_columns'])} columns")
            print(f"  Target: {schema['target_column']}")
        
        print("\n" + "="*60)
        print("SRM Adapter Ready!")
        print("="*60)
        print("\nTo use with actual SRM data:")
        print("1. Place SRM CSV file in data/ directory")
        print("2. Create schema JSON if needed in config/ directory")
        print("3. Run with --csv and --schema arguments")
        print()


if __name__ == "__main__":
    main()
