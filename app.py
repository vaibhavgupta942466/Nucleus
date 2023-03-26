from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from model import RFCmodel, DTCmodel
from dateutil import parser
from datetime import datetime

app = Flask(__name__)

# Load the trained models
rfc_model = RFCmodel()
dtc_model = DTCmodel()

# Define the column names and categories
col_names = {
    'demographics': ['age', 'sex'],
    'medical_history': ['anaemia', 'diabetes', 'blood_pressure', 'smoking'],
    'lab_results': ['creatinine_phosphokinase', 'platelets', 'serum_creatinine', 'serum_sodium', 'ejection_fraction'],
}


# Define the route for the form
@app.route('/', methods=['GET', 'POST'])
def landing_page():
    if request.method == 'POST':
        return redirect(url_for('heart_attack_predict'))
    else:
        return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def heart_attack_predict():
    if request.method == 'POST':
        # Get the user inputs and store them in a Pandas DataFrame
        data = {}
        prediction = {}
        print(request.form)
        for key, value in request.form.items():
            if key in col_names['demographics']:
                if key == 'age':
                    age = int(datetime.now().strftime('%Y')) - int(parser.parse(request.form.get('age')).strftime('%Y'))
                    data[key] = age
                else:
                    data[key] = [int(value)]
            elif key in col_names['medical_history']:
                data[key] = [int(value)]
            elif key in col_names['lab_results']:
                data[key] = [float(value)]
            elif key == 'name' or key == 'phone':
                prediction[key] = value
            else:
                data[key] = [value]

        user_inputs = pd.DataFrame(data, index=[0])

        print(user_inputs)
        
        # Make the prediction
        prediction['rfc'] = (rfc_model.predict(user_inputs))[0]
        prediction['dtc'] = (dtc_model.predict(user_inputs))[0]
        # Return the prediction as a response to the user
        print(prediction)
        return render_template('report.html', prediction=prediction)
    
    # If the request method is GET, generate the HTML form
    else:
        return render_template('form.html')
    
@app.route('/about')
def about():
    return render_template('about_us.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
