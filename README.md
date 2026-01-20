# Merge.ai: Privacy-Preserving Distributed Healthcare ML Framework

A comprehensive research prototype implementing privacy-preserving distributed machine learning techniques for healthcare data.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Flower](https://img.shields.io/badge/Flower-1.6-green.svg)](https://flower.dev/)

---

## 🎯 Project Overview

### What Problem Are We Solving?

Healthcare data is highly sensitive and subject to strict privacy regulations (HIPAA, GDPR). Traditional machine learning requires centralizing data in one location, which:
- **Violates patient privacy** 
- **Creates security vulnerabilities**
- **Limits data sharing** between institutions
- **Prevents collaborative research**

This project demonstrates how distributed learning techniques enable multiple hospitals to collaboratively train machine learning models **without sharing raw patient data**.

### Why Distributed Learning is Necessary

**Traditional Centralized ML:**
```
Hospital A data ──┐
Hospital B data ──┼──> Central Server (PRIVACY RISK!)
Hospital C data ──┘
```

**Privacy-Preserving Distributed ML:**
```
Hospital A ──┐
Hospital B ──┼──> Only model updates shared ──> Aggregation Server
Hospital C ──┘       (no raw patient data)
```

### Key Benefits

✅ **Privacy Protection**: Patient data never leaves hospitals  
✅ **Collaborative Learning**: Hospitals benefit from collective knowledge  
✅ **Regulatory Compliance**: Meets HIPAA/GDPR requirements  
✅ **Security**: Reduced attack surface, no central data repository  
✅ **Transparency**: Blockchain audit trail for accountability  

---

## 🚀 Quick Start

### 🆕 Run Benchmark Dataset Evaluation (NEW!)

Evaluate federated learning with comprehensive metrics:

```bash
# Clone and setup
git clone https://github.com/Vedanthdamn/Merge.ai.git
cd Merge.ai

# Install dependencies
pip install -r requirements.txt

# Run with benchmark dataset (diabetes or breast_cancer)
python demo_integration.py --dataset benchmark

# Or run with SRM hospital dataset
python demo_integration.py --dataset srm
```

**New Features:**
- ✅ Benchmark datasets (diabetes, breast_cancer) ready out-of-the-box
- ✅ SRM hospital dataset plug-in template (CSV with schema mapping)
- ✅ Comprehensive metrics: model performance, FL metrics, privacy metrics, fairness
- ✅ Automated report generation: `outputs/metrics.json`, `outputs/rounds_history.csv`
- ✅ Visualization plots: accuracy/loss curves, confusion matrix, fairness charts

**See [BENCHMARK_EVALUATION_GUIDE.md](BENCHMARK_EVALUATION_GUIDE.md) for detailed documentation.**

### Run Complete Integration Demo

Test all privacy-preserving components together:

```bash
# Run complete demonstration
python demo_integration.py --dataset benchmark
```

This demonstrates:
- Hospital data partitioning (Non-IID)
- Differential privacy on gradients
- Secure multi-party computation
- Blockchain audit logging
- Complete privacy-preserving workflow
- **Comprehensive metrics computation and reporting (NEW!)**

**Output**: See how all components work together without exposing patient data!

### Test Individual Components

```bash
# Data partitioning
python src/utils/data_partitioner.py

# Dataset loader (NEW!)
python src/utils/dataset_loader.py

# SRM adapter (NEW!)
python src/utils/srm_dataset_adapter.py

# Metrics computer (NEW!)
python src/evaluation/metrics.py

# Blockchain audit logging
python src/blockchain/audit_log.py

# Differential privacy
python src/privacy/differential_privacy.py

# Secure aggregation
python src/privacy/smpc.py
```

---

## 🌐 Inference API and Frontend

### Running the Inference Server (FastAPI)

The inference server provides a REST API for making predictions with the trained model:

```bash
# 1. Train the model first (if not already done)
python src/baseline/model.py

# 2. Start the FastAPI backend server
cd backend
python main.py
# Server will run on http://localhost:8000
```

**API Endpoints:**
- `GET /` - API information and health status
- `GET /health` - Health check endpoint
- `POST /predict` - Make predictions on patient data

**Example API Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62.5,
    "sex": 0,
    "systolic_bp": 136.2,
    "diastolic_bp": 93.3,
    "cholesterol": 268.8,
    "fasting_glucose": 70.0,
    "bmi": 29.2,
    "heart_rate": 98.4,
    "smoking": 0,
    "family_history": 1
  }'
```

**API Response:**
```json
{
  "probability": 0.6954,
  "risk_class": "High Risk",
  "confidence": 0.3908,
  "disclaimer": "Research Prototype – Not for Clinical Use. Always consult qualified healthcare professionals."
}
```

### Running the React Frontend

The frontend provides a modern, user-friendly interface for making predictions:

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies (first time only)
npm install

# 3. Start the development server
npm run dev
# Frontend will run on http://localhost:5173
```

**Frontend Features:**
- Modern, clean healthcare UI with Tailwind CSS
- Patient data input form with validation
- Real-time predictions from the backend API
- Visual display of risk classification and probability
- Prominent disclaimers about research prototype status

**Building for Production:**
```bash
cd frontend
npm run build
# Static files will be in frontend/dist/
```

### How Inference Works

1. **Model Loading**: On startup, the FastAPI server loads:
   - Trained neural network model (`models/baseline_model.h5`)
   - Fitted preprocessing components (`models/imputer.pkl`, `models/scaler.pkl`)

2. **Prediction Pipeline**:
   - Frontend sends patient data as JSON
   - Backend validates and preprocesses input using saved imputer and scaler
   - Model makes prediction
   - Results (probability, risk class, confidence) returned to frontend

3. **Privacy & Security**:
   - No patient data is stored
   - CORS enabled for frontend communication
   - All disclaimers prominently displayed

**Note**: The inference API is separate from the research components (federated learning, differential privacy, etc.) and is designed only for model inference.

---

## 🏗️ System Architecture

This system implements multiple privacy-preserving techniques that can be used individually or in combination:

### 1. Baseline Machine Learning Model

**Purpose**: Binary classification of clinical outcomes from tabular patient data.

**Architecture**: Multi-Layer Perceptron (MLP) neural network
- Input layer: Patient features (age, lab values, vitals, etc.)
- Hidden layers: Feature extraction and pattern learning
- Output layer: Binary prediction (disease presence/absence)

**Key Features**:
- Dataset-agnostic design (works with any healthcare CSV)
- Preprocessing pipeline (missing value imputation, normalization)
- Standard supervised learning workflow

**Privacy Consideration**: In isolation, the baseline model requires centralized data. It serves as the foundation for distributed approaches below.

---

### 2. Federated Learning (FL)

**What is Federated Learning?**

Federated Learning allows multiple hospitals to collaboratively train a shared model without exchanging patient data.

**How It Works**:

```
1. Server sends initial model to all hospitals
2. Each hospital trains model on LOCAL data only
3. Hospitals send ONLY model updates (weights) back to server
4. Server aggregates updates using FedAvg algorithm
5. Updated global model sent back to hospitals
6. Repeat for multiple rounds
```

**Privacy Guarantee**: 
- Raw patient data **never** leaves hospital premises
- Server only sees aggregated model parameters
- No individual hospital's data can be reconstructed

**Implementation**: Uses the [Flower](https://flower.dev/) framework for federated learning.

**Why Federated Learning?**
- Hospitals maintain full control over their data
- Model benefits from diverse patient populations
- Complies with data locality regulations

---

### 3. Split Learning (SL)

**What is Split Learning?**

Split Learning divides the neural network between client (hospital) and server, reducing computational burden on hospitals.

**Architecture**:

```
┌─────────────────┐
│  Hospital Side  │
│   (Client)      │
│                 │
│  Input Layer    │ ← Raw patient data (stays here)
│  Hidden Layer 1 │
│  Hidden Layer 2 │
└────────┬────────┘
         │ Intermediate activations
         │ (abstract features, NOT raw data)
         ↓
┌─────────────────┐
│  Server Side    │
│                 │
│  Hidden Layer 3 │
│  Hidden Layer 4 │
│  Output Layer   │
└─────────────────┘
```

**Privacy Guarantee**:
- Hospital processes raw data locally
- Only abstract feature representations sent to server
- Server never sees actual patient information
- Gradients flow back for training

**Why Split Learning?**
- Reduces computational load on hospital devices
- Still preserves data privacy (no raw data transmitted)
- Enables training even with limited hospital computing resources

---

### 4. Differential Privacy (DP)

**What is Differential Privacy?**

Differential Privacy adds carefully calibrated noise to prevent re-identification of individual patients.

**Core Principle**:
> "Adding or removing a single patient from the dataset should not significantly change the model's behavior."

**How It Works**:

1. **Gradient Clipping**: Limit influence of any single patient
   ```
   If gradient from one patient is too large → clip it
   ```

2. **Noise Addition**: Add Gaussian noise to gradients
   ```
   Noisy Gradient = Original Gradient + Noise(0, σ)
   ```

3. **Privacy Budget (ε)**: Controls privacy-utility tradeoff
   - ε < 1: Strong privacy (more noise)
   - ε = 1-10: Moderate privacy
   - ε > 10: Weak privacy (less noise)

**Privacy Guarantee**:
An adversary cannot determine if a specific patient was in the training data.

**Why Differential Privacy?**
- Formal mathematical privacy guarantee
- Prevents model from memorizing individual patients
- Protects against membership inference attacks
- Balances privacy and model accuracy

---

### 5. Secure Multi-Party Computation (SMPC)

**What is SMPC?**

SMPC allows multiple hospitals to compute an aggregate function (model average) without any party seeing others' individual inputs.

**How It Works** (Simplified):

```
Hospital A has update: 5
Hospital B has update: 3
Hospital C has update: 2

WITHOUT SMPC:
Server sees: [5, 3, 2] ← Privacy leak!

WITH SMPC (Secret Sharing):
Hospital A creates shares: [2, 1, 2] (sum = 5)
Hospital B creates shares: [1, 1, 1] (sum = 3)
Hospital C creates shares: [0, 1, 1] (sum = 2)

Server aggregates shares:
[2+1+0, 1+1+1, 2+1+1] = [3, 3, 4] = 10 ✓

Server only sees: 10 (the sum)
Individual values remain private!
```

**Privacy Guarantee**:
- Server cannot see any individual hospital's contribution
- Only the aggregate is revealed
- Collusion-resistant (up to a threshold of compromised parties)

**Implementation Note**: This is a **simplified simulation** for demonstration. Production SMPC requires proper cryptographic protocols (e.g., Shamir's Secret Sharing, Homomorphic Encryption).

**Why SMPC?**
- Strongest privacy guarantee (even server doesn't see individual updates)
- Enables secure aggregation in federated learning
- Protects against malicious aggregation servers

---

### 6. Blockchain-Based Audit Logging

**What is the Blockchain Component?**

An immutable ledger that records all training activities, creating a transparent audit trail.

**What Gets Logged**:
- Training round numbers
- Participating hospital IDs
- Model version hashes
- Aggregation methods used
- Performance metrics
- Privacy audit results

**What Does NOT Get Logged**:
- ❌ Raw patient data
- ❌ Individual patient records
- ❌ Hospital-specific model updates
- ❌ Any sensitive information

**How It Works**:

```
Block 0 (Genesis)
    ↓
Block 1: Round 1, Hospitals [A,B,C], Model v1
    ↓
Block 2: Round 2, Hospitals [A,B,C], Model v2
    ↓
Block 3: Evaluation Results, Model v2
    ↓
Block 4: Privacy Audit Passed
    ↓
Block 5: Model Deployed to Production
```

**Key Properties**:
- **Immutability**: Past records cannot be altered
- **Transparency**: All stakeholders can verify history
- **Accountability**: Know who participated in each round
- **Reproducibility**: Trace model lineage

**Why Blockchain?**
- Creates trust in multi-hospital collaborations
- Enables regulatory audits
- Provides tamper-evident record
- No single entity controls the history

**Important**: Blockchain is used ONLY for audit logging, NOT for training or inference (which would be computationally expensive).

---

### 7. Frontend Demo (Optional)

**Purpose**: Simple interface for demonstration purposes only.

**Features**:
- Manual input of patient features
- Display prediction results
- Show model performance metrics

**Important Disclaimers**:
- ⚠️ **NOT for clinical use**
- ⚠️ **Demo purposes only**
- ⚠️ **Does not store data**
- ⚠️ **Does not participate in training**
- ⚠️ **Does not access hospital databases**

**Implementation**: Lightweight Streamlit web application

---

## 🚫 Why We Don't Use Benchmark Datasets

### Datasets We Intentionally AVOID:

❌ **MNIST** (handwritten digits)  
❌ **CIFAR-10/100** (natural images)  
❌ **ImageNet** (image classification)  
❌ **Kaggle competition datasets**  
❌ **Any toy/benchmark datasets**  

### Why?

1. **Realism**: Benchmark datasets don't reflect real healthcare data challenges
   - Healthcare data is tabular, heterogeneous, and sensitive
   - Privacy concerns are not applicable to public datasets

2. **Privacy Sensitivity**: 
   - Real healthcare systems work with actual patient data
   - Privacy techniques must be validated in realistic contexts

3. **Academic Integrity**:
   - Using MNIST for healthcare privacy is misleading
   - Our goal is realistic system design, not inflated accuracy numbers

4. **Dataset-Agnostic Design**:
   - This system is built to work with ANY healthcare CSV
   - Focus on architecture and privacy, not specific dataset tuning

### What We Do Instead:

✅ **Mock data generation** for testing system components  
✅ **Placeholder paths** for user-provided real data  
✅ **Generic preprocessing** that works with any tabular data  
✅ **Clear documentation** on how to use your own healthcare dataset  

---

## 📁 Repository Structure

```
Merge.ai/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Ignore data, models, etc.
│
├── data/                             # PLACEHOLDER for user data
│   ├── .gitkeep                      # (Add your healthcare CSV here)
│   └── healthcare_data.csv           # (User provides - not in repo)
│
├── models/                           # Saved trained models
│   └── baseline_model.h5             # (Generated during training)
│
├── src/                              # Source code
│   ├── baseline/                     # Baseline ML model
│   │   ├── __init__.py
│   │   └── model.py                  # MLP for binary classification
│   │
│   ├── federated_learning/          # Federated Learning (Flower)
│   │   ├── __init__.py
│   │   ├── server.py                 # FL aggregation server
│   │   └── client.py                 # FL hospital client
│   │
│   ├── split_learning/               # Split Learning
│   │   ├── __init__.py
│   │   └── split_model.py            # Client/server split architecture
│   │
│   ├── privacy/                      # Privacy techniques
│   │   ├── __init__.py
│   │   ├── differential_privacy.py   # DP mechanisms
│   │   └── smpc.py                   # Secure aggregation (simulated)
│   │
│   ├── blockchain/                   # Audit logging
│   │   ├── __init__.py
│   │   └── audit_log.py              # Blockchain implementation
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       └── data_partitioner.py       # Hospital data simulation
│
└── frontend/                         # Demo web interface
    └── app.py                        # Streamlit application
```

---

## 🚀 How to Run This Project

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Vedanthdamn/Merge.ai.git
cd Merge.ai
```

### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow (deep learning)
- Flower (federated learning)
- Streamlit (web interface)
- Scikit-learn (preprocessing)
- NumPy, Pandas (data handling)
- Other required libraries

**Note**: Installation may take several minutes depending on your internet speed.

### Step 4: Prepare Your Dataset (Optional)

#### Option A: Use Your Own Healthcare Data

1. Prepare a CSV file with:
   - Feature columns (patient characteristics: age, lab values, vitals, etc.)
   - Target column named `outcome` with binary values (0 or 1)

2. Place the CSV in the `data/` folder:
   ```bash
   cp /path/to/your/healthcare_data.csv data/healthcare_data.csv
   ```

3. Ensure your CSV has:
   - No missing column names
   - Numeric features (or encode categorical variables first)
   - Binary target column (0/1)

#### Option B: Use Mock Data (For Testing)

The system will automatically generate mock data if no real dataset is found. This is sufficient for testing system components.

### Step 5: Run Baseline Model

Train the baseline machine learning model:

```bash
python src/baseline/model.py
```

**Expected Output**:
```
============================================================
Baseline Healthcare ML Model - Binary Classification
============================================================

Looking for dataset at: data/healthcare_data.csv

[INFO] No dataset found. This is expected for this prototype.
Creating mock data for demonstration...
Mock data created: 800 training samples, 200 test samples

------------------------------------------------------------
Initializing baseline model...
Model architecture:
Model: "sequential"
...
------------------------------------------------------------
Training model...
Epoch 1/10
...
------------------------------------------------------------
Evaluating model on test set...
Test Loss: 0.XXXX
Test Accuracy: 0.XXXX
Test AUC: 0.XXXX

Model saved to models/baseline_model.h5
============================================================
```

**What This Does**:
- Loads healthcare data (or generates mock data)
- Preprocesses features (imputation, normalization)
- Trains MLP neural network
- Evaluates on test set
- Saves trained model

### Step 6: Test Data Partitioning

Simulate hospital-based data distribution:

```bash
python src/utils/data_partitioner.py
```

**Expected Output**:
```
Hospital-Based Data Partitioning Demonstration
============================================================
Mock dataset: 1000 patients, 10 features

############################################################
Strategy: IID
############################################################

============================================================
Partitioning data for 3 hospitals
Strategy: iid
============================================================

Hospital 0: 333 patients, positive rate: XX.XX%
Hospital 1: 333 patients, positive rate: XX.XX%
Hospital 2: 334 patients, positive rate: XX.XX%
...
```

**What This Does**:
- Demonstrates IID vs Non-IID data distributions
- Shows how real hospitals have different patient populations
- Prepares for federated learning

### Step 7: Run Federated Learning

#### Terminal 1: Start FL Server

```bash
python src/federated_learning/server.py --rounds 5 --input-dim 10 --min-clients 2
```

**Parameters**:
- `--rounds`: Number of federated learning rounds (default: 10)
- `--input-dim`: Number of input features (default: 10)
- `--min-clients`: Minimum clients required (default: 2)
- `--server-address`: Server address (default: "0.0.0.0:8080")

**Expected Output**:
```
============================================================
Starting Federated Learning Server
============================================================
Server address: 0.0.0.0:8080
Number of rounds: 5
Minimum clients required: 2
============================================================

INFO:     Waiting for clients to connect...
```

**Server will wait for clients to connect.**

#### Terminal 2: Start FL Client 1

Open a **new terminal**, activate the virtual environment, and run:

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

python src/federated_learning/client.py --client-id 0 --server localhost:8080
```

**Expected Output**:
```
============================================================
Starting Federated Learning Client (Hospital 0)
============================================================
Server address: localhost:8080
Local training samples: XXX
============================================================

[Client 0] Initialized with XXX training samples
[Client 0] Training for 5 epochs...
...
```

#### Terminal 3: Start FL Client 2

Open **another new terminal**, activate the virtual environment, and run:

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

python src/federated_learning/client.py --client-id 1 --server localhost:8080
```

**What Happens**:
1. Server distributes initial model to clients
2. Each client trains on local data
3. Clients send model updates (NOT data) to server
4. Server aggregates updates using FedAvg
5. Process repeats for specified number of rounds

**Privacy**: Patient data never leaves individual clients!

### Step 8: Test Split Learning

```bash
python src/split_learning/split_model.py
```

**Expected Output**:
```
Split Learning Demonstration
============================================================
Mock dataset: 1000 patients, 10 features

============================================================
Split Learning Setup
============================================================
Client-side layers: [64]
Server-side layers: [32]
============================================================

Training with split learning for 10 epochs...
Epoch 1/10, Loss: X.XXXX
...

Making predictions on sample data...
Sample predictions: [0.XX, 0.XX, 0.XX, 0.XX, 0.XX]

============================================================
Key Privacy Features:
- Patient data processed locally at hospital
- Only intermediate activations sent to server
- Server never sees raw patient information
- Reduces computational load on hospital devices
============================================================
```

### Step 9: Test Privacy Mechanisms

#### Differential Privacy:

```bash
python src/privacy/differential_privacy.py
```

**Expected Output**:
```
============================================================
Differential Privacy Demonstration
============================================================

Original gradient norms:
  Layer 0: X.XXXX
  Layer 1: X.XXXX

------------------------------------------------------------
Strong Privacy (epsilon = 1.0)
------------------------------------------------------------
[Differential Privacy] Initialized
  Epsilon (privacy budget): 1.0
  Delta (failure probability): 1e-05
  Clip norm: 1.0
  Noise scale: X.XXXX

Noisy gradient norms:
  Layer 0: X.XXXX
  Layer 1: X.XXXX

╔═══════════════════════════════════════════════════════════╗
║           DIFFERENTIAL PRIVACY REPORT                     ║
╚═══════════════════════════════════════════════════════════╝
...
```

#### Secure Multi-Party Computation:

```bash
python src/privacy/smpc.py
```

**Expected Output**:
```
============================================================
Secure Multi-Party Computation (SMPC) Demonstration
============================================================

Simulating 3 hospitals
Model update shape: (10, 5)

Original hospital updates (norms):
  Hospital 0: X.XXXX
  Hospital 1: X.XXXX
  Hospital 2: X.XXXX

[Secure Aggregator] Initialized for 3 clients
------------------------------------------------------------
Performing secure aggregation...
------------------------------------------------------------
[Secure Aggregation] Processing 3 client updates...
  Client 0: Created 3 shares
  Client 1: Created 3 shares
  Client 2: Created 3 shares
...

============================================================
Key Privacy Properties:
============================================================
✓ Server never sees individual hospital updates
✓ Only aggregated result is revealed
✓ Collusion resistance (with proper protocols)
✓ Computation is correct despite privacy protection
============================================================
```

### Step 10: Test Blockchain Audit Log

```bash
python src/blockchain/audit_log.py
```

**Expected Output**:
```
============================================================
Blockchain-Based Audit Logging Demonstration
============================================================

[Blockchain] Initialized with genesis block

Simulating federated learning training...
[Blockchain] Block 1 added: training_round
[Blockchain] Block 2 added: training_round
[Blockchain] Block 3 added: model_evaluation
[Blockchain] Block 4 added: model_deployment
[Blockchain] Block 5 added: privacy_audit

============================================================
BLOCKCHAIN AUDIT LOG
============================================================

────────────────────────────────────────────────────────────
Block #0
Timestamp: 2024-XX-XXTXX:XX:XX.XXXXXX
Hash: XXXXXXXXXXXXXXXX...
Previous Hash: 0000000000000000...
Data: {
  "event": "blockchain_initialized",
  ...
}
...

Verifying blockchain integrity...
[Blockchain] Chain verified: All blocks are valid
...
```

### Step 11: Run Frontend Demo (Optional)

Start the Streamlit web interface:

```bash
streamlit run frontend/app.py
```

**Expected Output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.X.X:8501
```

**Open your browser** to `http://localhost:8501`

**Features**:
- Input patient features manually
- Get binary classification predictions
- View model information
- See privacy guarantees

**Remember**: This is a demo interface only, NOT for clinical use!

---

## 📊 Expected Results

### What to Expect:

Since this is a **research prototype** focused on architecture and privacy:

✅ **System Works**: All components run without errors  
✅ **Privacy Preserved**: No raw data transmitted between parties  
✅ **Audit Trail**: Blockchain logs all activities  
✅ **Modular Design**: Components can be used independently or together  

❓ **Accuracy**: Will depend on your dataset
- Mock data: Random accuracy (~50%)
- Real data: Depends on data quality and size

### This Project Demonstrates:

1. ✅ Privacy-preserving distributed learning is feasible
2. ✅ Multiple techniques can work together
3. ✅ System is modular and extensible
4. ✅ Audit trails enable transparency
5. ✅ No benchmark datasets needed

### This Project Does NOT:

❌ Guarantee production-ready cryptographic security  
❌ Provide medical advice or clinical decision support  
❌ Replace proper security audits  
❌ Handle all edge cases in real deployments  

---

## 📝 Notes and Limitations

### Intended Use:

- ✅ Educational purposes
- ✅ Research prototyping
- ✅ Architecture demonstration
- ✅ Academic papers and theses

### NOT Intended For:

- ❌ Production clinical deployment
- ❌ Real patient care decisions
- ❌ Unaudited security-critical applications

### Known Limitations:

1. **Simplified SMPC**: Production systems need proper cryptographic protocols (Shamir's Secret Sharing, MPC frameworks)

2. **Basic Blockchain**: Real systems would use established blockchains (Hyperledger, Ethereum) with proper consensus mechanisms

3. **DP Implementation**: Formal privacy accounting requires specialized libraries (TensorFlow Privacy, Opacus)

4. **No Byzantine Fault Tolerance**: Assumes honest participants; production needs robustness against malicious actors

5. **Communication**: Real federated learning requires secure channels (TLS, VPNs)

6. **Dataset Size**: Performance will vary with dataset size and quality

### Future Enhancements:

- Integration with proper cryptographic libraries
- Byzantine-robust aggregation
- Adaptive privacy budgets
- Cross-silo and cross-device scenarios
- Integration with FHIR healthcare standards
- Proper key management and authentication

---

## 🤝 Contributing

This is a research prototype. Contributions welcome for:

- Improved privacy mechanisms
- Additional distributed learning techniques
- Better documentation
- Bug fixes
- Security enhancements

**Please DO NOT**:
- Commit actual patient data
- Include proprietary healthcare information
- Use benchmark datasets (MNIST, CIFAR, etc.)

---

## 📄 License

This is a research prototype developed for educational purposes.

**Disclaimer**: This software is provided "as-is" without warranty. Not intended for clinical use. Always consult qualified healthcare professionals for medical decisions.

---

## 📧 Contact

For questions, issues, or collaboration:

- Open an issue on GitHub
- Refer to academic literature on federated learning and differential privacy

---

## 📚 References and Further Reading

### Federated Learning:
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- [Flower Framework Documentation](https://flower.dev/)

### Differential Privacy:
- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (2014)
- [TensorFlow Privacy](https://github.com/tensorflow/privacy)

### Split Learning:
- Gupta & Raskar, "Distributed Learning of Deep Neural Network over Multiple Agents" (2018)

### Healthcare ML Privacy:
- Kaissis et al., "Secure, privacy-preserving and federated machine learning in medical imaging" (2020)

### SMPC:
- Yao, "How to Generate and Exchange Secrets" (1986)
- [PySyft Framework](https://github.com/OpenMined/PySyft)

### Blockchain:
- [Hyperledger Fabric](https://www.hyperledger.org/use/fabric)

---

## 🎓 Academic Context

This project is suitable for:

- **Master's theses** in AI, healthcare informatics, security
- **Research papers** on privacy-preserving ML
- **Course projects** in distributed systems or ML
- **Proof-of-concept** for grant proposals

**Key Contributions**:
- Integrated multi-technique privacy framework
- Dataset-agnostic design
- Comprehensive documentation
- Realistic healthcare focus (no toy datasets)

---

## ⚠️ Final Disclaimer

**THIS IS A RESEARCH PROTOTYPE**

- ⚠️ Not validated for clinical use
- ⚠️ Not FDA approved
- ⚠️ Not HIPAA audited
- ⚠️ Requires security review before any real deployment
- ⚠️ Always consult healthcare professionals for medical decisions

**Use at your own risk. Authors assume no liability.**

---

**Built with privacy in mind. Healthcare data deserves protection.** 🏥🔒