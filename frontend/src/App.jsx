import { useState } from 'react'

const API_URL = 'http://localhost:8000'

function App() {
  const [formData, setFormData] = useState({
    age: '',
    sex: '0',
    systolic_bp: '',
    diastolic_bp: '',
    cholesterol: '',
    fasting_glucose: '',
    bmi: '',
    heart_rate: '',
    smoking: '0',
    family_history: '0',
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setPrediction(null)

    try {
      // Convert form data to proper types
      const data = {
        age: parseFloat(formData.age),
        sex: parseInt(formData.sex),
        systolic_bp: parseFloat(formData.systolic_bp),
        diastolic_bp: parseFloat(formData.diastolic_bp),
        cholesterol: parseFloat(formData.cholesterol),
        fasting_glucose: parseFloat(formData.fasting_glucose),
        bmi: parseFloat(formData.bmi),
        heart_rate: parseFloat(formData.heart_rate),
        smoking: parseInt(formData.smoking),
        family_history: parseInt(formData.family_history),
      }

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error('Prediction request failed')
      }

      const result = await response.json()
      setPrediction(result)
    } catch (err) {
      setError(err.message || 'Failed to get prediction. Make sure the backend server is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🏥 Privacy-Preserving Healthcare ML
          </h1>
          <p className="text-lg text-gray-600">
            Risk Prediction System
          </p>
          <div className="mt-4 inline-block bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-3 rounded">
            <p className="font-semibold">⚠️ Research Prototype – Not for Clinical Use</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Patient Data Form */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-6">
              Patient Data Input
            </h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Age */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Age (years)
                </label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  required
                  step="0.1"
                  min="0"
                  max="120"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., 62.5"
                />
              </div>

              {/* Sex */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Sex
                </label>
                <select
                  name="sex"
                  value={formData.sex}
                  onChange={handleChange}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="0">Female</option>
                  <option value="1">Male</option>
                </select>
              </div>

              {/* Blood Pressure */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Systolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    name="systolic_bp"
                    value={formData.systolic_bp}
                    onChange={handleChange}
                    required
                    step="0.1"
                    min="0"
                    max="300"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 136.2"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Diastolic BP (mmHg)
                  </label>
                  <input
                    type="number"
                    name="diastolic_bp"
                    value={formData.diastolic_bp}
                    onChange={handleChange}
                    required
                    step="0.1"
                    min="0"
                    max="200"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., 93.3"
                  />
                </div>
              </div>

              {/* Cholesterol */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Cholesterol (mg/dL)
                </label>
                <input
                  type="number"
                  name="cholesterol"
                  value={formData.cholesterol}
                  onChange={handleChange}
                  required
                  step="0.1"
                  min="0"
                  max="500"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., 268.8"
                />
              </div>

              {/* Fasting Glucose */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Fasting Glucose (mg/dL)
                </label>
                <input
                  type="number"
                  name="fasting_glucose"
                  value={formData.fasting_glucose}
                  onChange={handleChange}
                  required
                  step="0.1"
                  min="0"
                  max="500"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., 70.0"
                />
              </div>

              {/* BMI */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  BMI (Body Mass Index)
                </label>
                <input
                  type="number"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleChange}
                  required
                  step="0.1"
                  min="10"
                  max="80"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., 29.2"
                />
              </div>

              {/* Heart Rate */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Heart Rate (bpm)
                </label>
                <input
                  type="number"
                  name="heart_rate"
                  value={formData.heart_rate}
                  onChange={handleChange}
                  required
                  step="0.1"
                  min="30"
                  max="200"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., 98.4"
                />
              </div>

              {/* Smoking */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Smoking
                </label>
                <select
                  name="smoking"
                  value={formData.smoking}
                  onChange={handleChange}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>

              {/* Family History */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Family History
                </label>
                <select
                  name="family_history"
                  value={formData.family_history}
                  onChange={handleChange}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="0">No</option>
                  <option value="1">Yes</option>
                </select>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? 'Predicting...' : 'Predict Risk'}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Results Card */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6">
                Prediction Results
              </h2>

              {!prediction && !error && !loading && (
                <div className="text-center py-12 text-gray-500">
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="mt-4">Submit patient data to see prediction results</p>
                </div>
              )}

              {loading && (
                <div className="text-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                  <p className="mt-4 text-gray-600">Analyzing patient data...</p>
                </div>
              )}

              {error && (
                <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-red-800">Error</h3>
                      <p className="text-sm text-red-700 mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}

              {prediction && (
                <div className="space-y-6">
                  {/* Risk Classification */}
                  <div className={`p-6 rounded-lg text-center ${
                    prediction.risk_class === 'High Risk' 
                      ? 'bg-red-50 border-2 border-red-300' 
                      : 'bg-green-50 border-2 border-green-300'
                  }`}>
                    <p className="text-sm font-medium text-gray-600 mb-2">Risk Classification</p>
                    <p className={`text-3xl font-bold ${
                      prediction.risk_class === 'High Risk' 
                        ? 'text-red-600' 
                        : 'text-green-600'
                    }`}>
                      {prediction.risk_class === 'High Risk' ? '🔴' : '🟢'} {prediction.risk_class}
                    </p>
                  </div>

                  {/* Probability */}
                  <div>
                    <div className="flex justify-between text-sm font-medium text-gray-600 mb-2">
                      <span>Probability</span>
                      <span>{(prediction.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div 
                        className={`h-4 rounded-full ${
                          prediction.probability >= 0.5 ? 'bg-red-500' : 'bg-green-500'
                        }`}
                        style={{ width: `${prediction.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div>
                    <div className="flex justify-between text-sm font-medium text-gray-600 mb-2">
                      <span>Confidence</span>
                      <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-4">
                      <div 
                        className="h-4 rounded-full bg-blue-500"
                        style={{ width: `${prediction.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Disclaimer Card */}
            <div className="bg-amber-50 border-l-4 border-amber-500 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-amber-800 mb-3">
                ⚠️ Important Disclaimer
              </h3>
              <div className="text-sm text-amber-700 space-y-2">
                <p><strong>Research Prototype – Not for Clinical Use</strong></p>
                <p>This prediction is from a research prototype and should NOT be used for:</p>
                <ul className="list-disc list-inside ml-2 space-y-1">
                  <li>Clinical diagnosis</li>
                  <li>Treatment decisions</li>
                  <li>Medical advice</li>
                </ul>
                <p className="mt-3 font-medium">
                  Always consult qualified healthcare professionals for medical decisions.
                </p>
              </div>
            </div>

            {/* System Info Card */}
            <div className="bg-blue-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-blue-800 mb-3">
                🔒 Privacy Features
              </h3>
              <div className="text-sm text-blue-700 space-y-2">
                <div className="flex items-start">
                  <span className="mr-2">✓</span>
                  <span>Federated Learning</span>
                </div>
                <div className="flex items-start">
                  <span className="mr-2">✓</span>
                  <span>Differential Privacy</span>
                </div>
                <div className="flex items-start">
                  <span className="mr-2">✓</span>
                  <span>Secure Multi-Party Computation</span>
                </div>
                <div className="flex items-start">
                  <span className="mr-2">✓</span>
                  <span>Blockchain Audit Trail</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-600 text-sm">
          <p>Privacy-Preserving Distributed Healthcare ML Framework</p>
          <p className="mt-1">Research Prototype - Not for Clinical Use</p>
        </div>
      </div>
    </div>
  )
}

export default App
