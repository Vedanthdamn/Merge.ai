# Benchmark Dataset Evaluation & Comprehensive Metrics Guide

This guide explains the new benchmark evaluation and metrics features added to the Merge.ai federated learning framework.

## 🎯 Overview

The system now supports:
- ✅ Benchmark dataset evaluation (diabetes, breast_cancer)
- ✅ SRM hospital dataset plug-in template
- ✅ Comprehensive metrics computation
- ✅ Automated report generation and visualization

## 🚀 Quick Start

### Run with Benchmark Dataset

```bash
# Using diabetes dataset (default)
python demo_integration.py --dataset benchmark

# The system will:
# 1. Load diabetes benchmark dataset
# 2. Partition data across 3 hospitals (non-IID)
# 3. Run 10 rounds of federated learning
# 4. Compute comprehensive metrics
# 5. Generate plots and reports
```

### Run with SRM Hospital Dataset

```bash
# Using SRM dataset (configure in config.yaml first)
python demo_integration.py --dataset srm
```

## 📁 Output Structure

After running, check the `outputs/` directory:

```
outputs/
├── metrics.json              # Complete metrics summary
├── rounds_history.csv        # Per-round training history
└── plots/
    ├── accuracy_vs_rounds.png       # Accuracy progression
    ├── loss_vs_rounds.png          # Loss progression
    ├── confusion_matrix.png        # Final confusion matrix
    └── client_accuracy_variance.png # Fairness visualization
```

## 📊 Metrics Computed

### 1. Model Performance Metrics
- Global accuracy
- Client-level accuracy (per hospital)
- Precision, Recall, F1 score
- Confusion Matrix
- Loss (Cross-Entropy)
- AUC-ROC

### 2. Federated Learning Metrics
- **Communication:**
  - Total messages exchanged
  - Bytes transmitted
  - Upload/download per client
- **Convergence:**
  - Rounds needed to converge
  - Convergence rate (accuracy gain per round)
- **Client Participation:**
  - Active clients per round
  - Participation ratio
- **Data Heterogeneity:**
  - Client drift metrics
  - Weight divergence
  - Gradient divergence approximation

### 3. Privacy Metrics
- **Differential Privacy:**
  - Privacy budget (ε, δ)
  - Per-round epsilon
  - Total epsilon consumed
- **Accuracy vs Privacy Tradeoff:**
  - Plots generated for different epsilon values
  
### 4. SMPC Metrics (Placeholders)
- Computation overhead estimate
- Communication overhead multiplier
- Aggregation latency estimate

### 5. Fairness Metrics
- Client accuracy variance
- Per-client contribution scores
- Fairness score (0-1, higher is more fair)

## ⚙️ Configuration

Edit `config.yaml` to customize:

### Dataset Configuration

```yaml
dataset:
  type: 'benchmark'  # or 'srm' or 'csv'
  benchmark_name: 'diabetes'  # or 'breast_cancer'
  
  # For SRM dataset:
  srm:
    path: 'data/srm_hospital_data.csv'
    schema:
      feature_columns: ['age', 'gender', 'bp', ...]
      target_column: 'diagnosis'
```

### Model Configuration

```yaml
model:
  hidden_layers: [64, 32]
  dropout_rate: 0.3
  learning_rate: 0.001
```

### Federated Learning Configuration

```yaml
federated_learning:
  n_clients: 3
  n_rounds: 10
  local_epochs: 5
  batch_size: 32
  partition_strategy: 'non_iid'  # 'iid', 'non_iid', 'class_imbalance'
```

### Privacy Configuration

```yaml
privacy:
  differential_privacy:
    enabled: true
    epsilon: 1.0
    delta: 1e-5
    clip_norm: 1.0
  smpc:
    enabled: true
```

## 🏥 Using SRM Hospital Dataset

When SRM provides their dataset:

### Step 1: Place the CSV file

```bash
cp srm_data.csv data/srm_hospital_data.csv
```

### Step 2: Update config.yaml schema mapping

```yaml
dataset:
  type: 'srm'
  srm:
    path: 'data/srm_hospital_data.csv'
    schema:
      # Map SRM column names to expected format
      feature_columns: 
        - 'patient_age'      # SRM's age column
        - 'patient_gender'   # SRM's gender column
        - 'blood_pressure'   # SRM's BP column
        # ... add all feature columns
      target_column: 'disease_outcome'  # SRM's target column
      patient_id_column: 'patient_id'   # Will be dropped for privacy
```

### Step 3: Run

```bash
python demo_integration.py --dataset srm
```

That's it! The pipeline will work without any code changes.

## 🔍 Understanding the Metrics

### metrics.json Structure

```json
{
  "final_metrics": {
    "dataset_info": {...},
    "configuration": {...},
    "model_performance": {
      "accuracy": 0.85,
      "precision": 0.83,
      "recall": 0.87,
      "f1_score": 0.85,
      "confusion_matrix": [[45, 5], [7, 43]],
      "loss": 0.45,
      "auc": 0.89
    },
    "federated_learning_metrics": {
      "total_rounds": 10,
      "rounds_to_convergence": 8,
      "communication_cost": {
        "total_messages": 60,
        "total_bytes_transmitted": 62914560
      },
      "convergence_rate": 0.035,
      "final_accuracy": 0.85
    },
    "privacy_metrics": {
      "privacy_budget": {
        "epsilon_per_round": 1.0,
        "total_epsilon": 10.0,
        "delta": 1e-5
      }
    },
    "fairness_metrics": {
      "client_accuracy_variance": 0.0004,
      "fairness_score": 0.9996
    }
  },
  "rounds_history": [...]
}
```

### rounds_history.csv

Tracks per-round metrics:

| round | accuracy | loss | precision | recall | f1_score | client_mean_accuracy | ... |
|-------|----------|------|-----------|--------|----------|---------------------|-----|
| 1     | 0.625    | 5.79 | 0.677     | 0.477  | 0.560    | 0.744               | ... |
| 2     | 0.568    | 6.07 | 0.539     | 0.932  | 0.683    | 0.724               | ... |
| ...   | ...      | ...  | ...       | ...    | ...      | ...                 | ... |

## 📈 Plots Generated

### 1. accuracy_vs_rounds.png
Shows how global model accuracy improves across federated learning rounds.

### 2. loss_vs_rounds.png
Shows how global model loss decreases across rounds.

### 3. confusion_matrix.png
Heatmap showing true vs predicted classifications.

### 4. client_accuracy_variance.png
Bar chart showing per-client accuracy distribution (fairness analysis).

### 5. accuracy_vs_epsilon.png (if DP tradeoff analysis enabled)
Shows privacy-accuracy tradeoff curve.

## 🧪 Testing Individual Modules

### Test Dataset Loader

```bash
python src/utils/dataset_loader.py
```

### Test SRM Adapter

```bash
python src/utils/srm_dataset_adapter.py
```

### Test Metrics Computer

```bash
python src/evaluation/metrics.py
```

### Test Metrics Tracker

```bash
python src/evaluation/metrics_tracker.py
```

## 🔧 Advanced Usage

### Use Different Benchmark Dataset

```bash
# Edit config.yaml:
dataset:
  benchmark_name: 'breast_cancer'

# Run:
python demo_integration.py --dataset benchmark
```

### Adjust Privacy Budget

```bash
# Edit config.yaml:
privacy:
  differential_privacy:
    epsilon: 5.0  # Higher = less privacy, more accuracy

# Run:
python demo_integration.py --dataset benchmark
```

### Change Data Partitioning Strategy

```bash
# Edit config.yaml:
federated_learning:
  partition_strategy: 'iid'  # or 'non_iid' or 'class_imbalance'

# Run:
python demo_integration.py --dataset benchmark
```

## 🎓 Example Workflow

### Complete Benchmark Evaluation

```bash
# 1. Clone repository
git clone https://github.com/Vedanthdamn/Merge.ai.git
cd Merge.ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run benchmark evaluation
python demo_integration.py --dataset benchmark

# 4. Check results
ls outputs/
cat outputs/metrics.json
```

### SRM Hospital Integration

```bash
# 1. Receive dataset from SRM
# 2. Place in data/ folder
cp ~/Downloads/srm_data.csv data/srm_hospital_data.csv

# 3. Update schema in config.yaml
nano config.yaml

# 4. Run evaluation
python demo_integration.py --dataset srm

# 5. Share results with SRM
tar -czf srm_results.tar.gz outputs/ federated_learning_audit.json
```

## 📝 Notes

- **CPU-only:** Runs on CPU without GPU requirements
- **Privacy-preserving:** All data partitioning respects privacy constraints
- **Extensible:** Easy to add new datasets or metrics
- **Reproducible:** Fixed random seeds for consistent results

## 🐛 Troubleshooting

### "Dataset not found"
- Check the path in config.yaml
- Ensure CSV file exists in data/ directory

### "Column not found"
- Verify schema mapping in config.yaml
- Check CSV column names match configuration

### "Out of memory"
- Reduce batch_size in config.yaml
- Reduce n_clients or local_epochs

## 📚 References

- Main README: `README.md`
- Installation guide: `INSTALL.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`

## 🤝 Support

For issues or questions, please open a GitHub issue or contact the development team.
