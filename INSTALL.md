# Installation Guide

## Quick Start (Minimal Installation)

If you want to test the core components without heavy ML dependencies:

```bash
pip install numpy pandas
```

This allows you to run:
- Data partitioning (`src/utils/data_partitioner.py`)
- Blockchain audit logging (`src/blockchain/audit_log.py`)
- Privacy modules (`src/privacy/differential_privacy.py`, `src/privacy/smpc.py`)

## Full Installation

For complete functionality including model training and federated learning:

```bash
pip install -r requirements.txt
```

**Note**: This installs TensorFlow and Flower, which are large packages (several GB).

## Optional Components

### Frontend (Streamlit)
```bash
pip install streamlit matplotlib plotly
streamlit run frontend/app.py
```

### Additional Privacy Libraries
```bash
pip install opacus pycryptodome
```

## Troubleshooting

### Disk Space Issues
TensorFlow and its dependencies require significant disk space. If installation fails:

1. Install minimal dependencies first:
   ```bash
   pip install numpy pandas
   ```

2. Test core components (partitioning, blockchain, privacy)

3. Install ML libraries separately when needed:
   ```bash
   pip install tensorflow scikit-learn
   pip install flwr
   ```

### Platform-Specific Issues

**Windows**: Use `python -m pip` instead of `pip`

**Mac (M1/M2)**: TensorFlow installation may require additional steps. See [TensorFlow Mac Guide](https://www.tensorflow.org/install/pip#macos)

**Linux**: Ensure you have Python development headers:
```bash
sudo apt-get install python3-dev
```

## Verifying Installation

Test individual components:

```bash
# Test data partitioning (requires only numpy)
python src/utils/data_partitioner.py

# Test blockchain (no ML dependencies)
python src/blockchain/audit_log.py

# Test privacy modules (requires only numpy)
python src/privacy/differential_privacy.py
python src/privacy/smpc.py

# Test baseline model (requires TensorFlow)
python src/baseline/model.py

# Test split learning (requires TensorFlow)
python src/split_learning/split_model.py
```

## Docker Alternative (Coming Soon)

For easier deployment, we plan to provide Docker containers with all dependencies pre-installed.
