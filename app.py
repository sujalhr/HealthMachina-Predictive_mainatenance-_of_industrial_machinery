from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained machine learning model
model = joblib.load('wmodel.pkl')
product_type_encoder =  joblib.load('product_type_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Extract the input features from the form
    uid = data['uid']
    product_id = data['productId']
    product_type = [data['productType']]  # Needs to be passed as a list for encoding
    air_temp = float(data['airTemp'])
    process_temp = float(data['processTemp'])
    rpm = float(data['rpm'])
    torque = float(data['torque'])
    tool_wear = float(data['toolWear'])
    machine_failure = 1 if data['machineFailure'] == 'true' else 0  # convert to 1 or 0
    
    # One-hot encode the product type
    encoded_product_type = product_type_encoder.transform(np.array(product_type).reshape(-1, 1)).toarray()  # Converts to dense array
    
    # Combine all features into a single array
    # Convert scalar values into 2D arrays and flatten
    features = np.hstack([np.array([[air_temp, process_temp, rpm, torque, tool_wear]]), encoded_product_type])

    # Predict using the loaded model
    prediction = model.predict(features)
    
    # Define failure modes
    failure_mode = {0: "Heat Dissipation Failure",
                    1: "No Failure",
                    2: "Overstrain Failure",
                    3: "Power Failure",
                    4: "Random Failures",
                    5: "Tool Wear Failure"}
    
    # Return the result as JSON
    result = {
        'prediction': int(prediction[0]),  # Assuming the model returns a class label
        'failure_mode': failure_mode[int(prediction[0])]  # Replace this with your logic to determine failure mode
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
