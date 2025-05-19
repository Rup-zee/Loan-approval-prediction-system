from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model artifacts
model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def get_decision_reasons(input_data, prediction, prediction_proba):
    """Generate human-readable reasons for the decision"""
    reasons = []
    
    # Create a dictionary of feature values
    feature_values = {
        'dependents': input_data[0][0],
        'education': "Graduate" if input_data[0][1] == 1 else "Not Graduate",
        'self_employed': "Yes" if input_data[0][2] == 1 else "No",
        'income': input_data[0][3],
        'loan_amount': input_data[0][4],
        'loan_term': input_data[0][5],
        'cibil_score': input_data[0][6],
        'assets': input_data[0][7],
        'loan_to_income': input_data[0][4] / max(input_data[0][3], 1),
        'asset_coverage': input_data[0][7] / max(input_data[0][4], 1),
        'debt_burden': input_data[0][4] / (max(input_data[0][3], 1) * max(input_data[0][5], 1)/12)
    }
    
    # Decision logic
    if feature_values['cibil_score'] < 650:
        reasons.append(f"Low CIBIL score ({feature_values['cibil_score']}) - minimum required is 650")
    else:
        reasons.append(f"Good CIBIL score ({feature_values['cibil_score']})")
    
    if feature_values['loan_to_income'] > 0.5:
        reasons.append(f"High loan-to-income ratio ({feature_values['loan_to_income']:.2f})")
    else:
        reasons.append(f"Acceptable loan-to-income ratio ({feature_values['loan_to_income']:.2f})")
    
    if feature_values['asset_coverage'] < 0.8:
        reasons.append(f"Low asset coverage ({feature_values['asset_coverage']:.2f}x loan amount)")
    else:
        reasons.append(f"Good asset coverage ({feature_values['asset_coverage']:.2f}x loan amount)")
    
    # Confidence level
    confidence = prediction_proba * 100
    if prediction == 1:
        reasons.insert(0, "Application meets approval criteria")
    else:
        reasons.insert(0, "Application doesn't meet approval criteria")
    
    return prediction, reasons, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'dependents': float(request.form['dependents']),
            'education': 1 if request.form['education'] == 'Graduate' else 0,
            'self_employed': 1 if request.form['self_employed'] == 'Yes' else 0,
            'income': float(request.form['income']),
            'loan_amount': float(request.form['loan_amount']),
            'loan_term': float(request.form['loan_term']),
            'cibil_score': float(request.form['cibil_score']),
            'assets': float(request.form['assets'])
        }

        # Prepare input array
        input_data = np.array([[
            form_data['dependents'],
            form_data['education'],
            form_data['self_employed'],
            form_data['income'],
            form_data['loan_amount'],
            form_data['loan_term'],
            form_data['cibil_score'],
            form_data['assets']
        ]])
        
        # Scale and predict
        scaled_input = scaler.transform(input_data)
        prediction_proba = model.predict_proba(scaled_input)[0][1]
        initial_prediction = model.predict(scaled_input)[0]
        
        # Get decision reasons
        final_prediction, reasons, confidence = get_decision_reasons(
            input_data, 
            initial_prediction, 
            prediction_proba
        )
        
        result = {
            'decision': "Approved" if final_prediction == 1 else "Rejected",
            'reasons': reasons,
            'confidence': round(confidence),
            'form_data': form_data
        }

    except Exception as e:
        result = {
            'error': str(e),
            'decision': "Error",
            'reasons': [],
            'confidence': 0
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
