from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('wmodel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    uid = data['uid']
    product_id = data['productId']
    product_type = data['productType']
    air_temp = float(data['airTemp'])
    process_temp = float(data['processTemp'])
    rpm = float(data['rpm'])
    torque = float(data['torque'])
    tool_wear = float(data['toolWear'])
    machine_failure = data['machineFailure'] == 'true'
    
    # Example input features based on the Iris dataset
    features = [[air_temp, process_temp, rpm, torque]]

    # Predict using the loaded model
    prediction = model.predict(features)
    
    # Return the result as JSON
    result = {
        'prediction': int(prediction[0]),  # Assuming the model returns a class label
        'failure_mode': 'Example Failure Mode'  # Replace this with your logic to determine failure mode
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)