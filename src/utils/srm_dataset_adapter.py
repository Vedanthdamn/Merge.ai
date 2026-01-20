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
from typing import Dict, Optional
import os


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
    Test the SRM dataset adapter.
    """
    print("SRM Dataset Adapter - Template Test")
    print("="*60)
    
    # Example 1: Using default schema
    print("\n1. Testing with default schema")
    print("-"*60)
    adapter = SRMDatasetAdapter()
    
    # Check if test data exists
    test_path = "data/healthcare_data.csv"
    if os.path.exists(test_path):
        print(f"\nValidating schema against: {test_path}")
        validation = adapter.validate_schema(test_path)
        print(f"Validation result: {validation}")
        
        if validation['valid']:
            print("\nLoading dataset...")
            X, y, metadata = adapter.load_srm_dataset(test_path)
            print(f"Successfully loaded: X.shape={X.shape}, y.shape={y.shape}")
    else:
        print(f"Note: Test file not found at {test_path}")
        print("When SRM provides data, place it at the configured path.")
    
    # Example 2: Custom schema
    print("\n\n2. Example: Custom schema configuration")
    print("-"*60)
    custom_schema = {
        'feature_columns': ['age', 'sex', 'bmi', 'cholesterol'],
        'target_column': 'disease',
        'patient_id_column': 'id'
    }
    print("Custom schema:")
    for key, value in custom_schema.items():
        print(f"  {key}: {value}")
    
    # Example 3: Show available example schemas
    print("\n\n3. Available example schemas")
    print("-"*60)
    for name, schema in EXAMPLE_SCHEMAS.items():
        print(f"\n{name.upper()} schema:")
        print(f"  Features: {len(schema['feature_columns'])} columns")
        print(f"  Target: {schema['target_column']}")
    
    print("\n" + "="*60)
    print("SRM Adapter Template Ready!")
    print("="*60)
    print("\nTo use with SRM data:")
    print("1. Update config.yaml with SRM dataset path")
    print("2. Adjust schema mapping if column names differ")
    print("3. Run: python demo_integration.py --dataset srm")
    print()


if __name__ == "__main__":
    main()
