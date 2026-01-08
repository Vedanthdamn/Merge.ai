# Quick Start Guide: Inference API and Frontend

This guide will help you quickly set up and run the inference API and frontend for the Privacy-Preserving Healthcare ML Framework.

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ and npm installed
- Git (to clone the repository)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vedanthdamn/Merge.ai.git
cd Merge.ai
```

### 2. Set Up Python Environment

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Train the Model

Before running the inference server, you need to train the model:

```bash
python src/baseline/model.py
```

This will:
- Train a baseline model on the healthcare dataset
- Save the trained model to `models/baseline_model.h5`
- Save preprocessing components (`imputer.pkl` and `scaler.pkl`)

Expected output: Model trained with ~60-70% accuracy on test set.

### 4. Start the Backend Server

In a new terminal window:

```bash
cd Merge.ai
source venv/bin/activate  # Activate virtual environment
python backend/main.py
```

The API server will start on `http://localhost:8000`

You should see:
```
✓ Model loaded successfully
✓ Imputer loaded successfully
✓ Scaler loaded successfully
Server ready for inference!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. Start the Frontend

In another terminal window:

```bash
cd Merge.ai/frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm run dev
```

The frontend will start on `http://localhost:5173`

You should see:
```
VITE v7.3.1  ready in XXX ms
➜  Local:   http://localhost:5173/
```

### 6. Use the Application

1. Open your browser and navigate to `http://localhost:5173`
2. Fill in the patient data form with sample values:
   - Age: 62.5
   - Sex: Female
   - Systolic BP: 136.2
   - Diastolic BP: 93.3
   - Cholesterol: 268.8
   - Fasting Glucose: 70.0
   - BMI: 29.2
   - Heart Rate: 98.4
   - Smoking: No
   - Family History: Yes
3. Click "Predict Risk"
4. View the prediction results showing probability and risk classification

## Testing the API Directly

You can also test the API using curl:

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

Expected response:
```json
{
  "probability": 0.6954,
  "risk_class": "High Risk",
  "confidence": 0.3908,
  "disclaimer": "Research Prototype – Not for Clinical Use. Always consult qualified healthcare professionals."
}
```

## Building for Production

To build the frontend for production:

```bash
cd frontend
npm run build
```

The static files will be generated in `frontend/dist/` and can be served by any static file server.

## Troubleshooting

### Backend Issues

**Problem**: Model not found error
```
RuntimeError: Model not found at models/baseline_model.h5
```

**Solution**: Run `python src/baseline/model.py` to train and save the model first.

---

**Problem**: Import errors (sklearn, tensorflow, etc.)
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**: Install dependencies: `pip install -r requirements.txt`

### Frontend Issues

**Problem**: Port 5173 already in use
```
Error: listen EADDRINUSE: address already in use :::5173
```

**Solution**: Kill the process using port 5173 or use a different port:
```bash
npm run dev -- --port 3000
```

---

**Problem**: Cannot connect to backend
```
Failed to get prediction. Make sure the backend server is running.
```

**Solution**: Ensure the backend is running on `http://localhost:8000` and check CORS settings.

### Network Issues

**Problem**: Frontend can't reach backend
```
Failed to fetch
```

**Solution**: 
1. Verify backend is running: `curl http://localhost:8000/health`
2. Check if you need to update the API_URL in `frontend/src/App.jsx` if using a different host/port

## Important Notes

⚠️ **This is a research prototype** - Not for clinical use!

- Always display disclaimers when showing predictions
- Do not use for actual medical decision-making
- Consult qualified healthcare professionals for medical decisions

## Next Steps

- Explore the federated learning capabilities: `python demo_integration.py`
- Read the full README for details on privacy-preserving techniques
- Review the architecture documentation

## Support

For issues or questions:
- Check the main README.md
- Review the documentation in the repository
- Open an issue on GitHub
