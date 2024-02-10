from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

# Load the saved Isolation Forest model
with open('isolation_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a route to render the input form
@app.route('/')
def home():
    return render_template('index.html')

label_encoder=LabelEncoder()
# Define a route to handle form submission and display predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    source_ip = request.form['source_ip']
    user_identity = request.form['user_identity']
    error_code = request.form['error_code']
    
    # Create a DataFrame with the input data
    data = pd.DataFrame({'sourceIPAddress': [source_ip],
                         'userIdentitytype': [user_identity],
                         'errorCode': [error_code]})
    
    # Convert categorical features to numerical using one-hot encoding
    data['sourceIPAddress'] = label_encoder.fit_transform(data['sourceIPAddress'])
    data['userIdentitytype'] = label_encoder.fit_transform(data['userIdentitytype'])
    data['errorCode'] = label_encoder.fit_transform(data['errorCode'])

    # Make predictions using the loaded model
    anomaly_score = model.decision_function(data)
    is_anomaly = model.predict(data)
    
    # Pass the prediction results to the HTML page
    return render_template('result.html', anomaly_score=anomaly_score[0], is_anomaly=is_anomaly[0])

if __name__ == '__main__':
    app.run(debug=True)
