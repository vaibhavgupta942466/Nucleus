from flask import Flask, render_template, request
import pandas as pd
from model import RFCmodel, DTCmodel

app = Flask(__name__)

# Load the trained models
rfc_model = RFCmodel()
dtc_model = DTCmodel()

# Define the column names and categories
col_names = {
    'demographics': ['age', 'sex'],
    'medical_history': ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking'],
    'lab_results': ['creatinine_phosphokinase', 'platelets', 'serum_creatinine', 'serum_sodium'],
    'heart_function': ['ejection_fraction'],
    'follow_up': ['time']
}

# Create a function to generate the HTML form
def generate_form():
    form = ''
    form += '<input type="hidden" id="model" name="model" value="RFC">'
    form += '<input type="submit" value="Use Random Forest Classifier"><br><br>'
    form += '<input type="hidden" id="model" name="model" value="DTC">'
    form += '<input type="submit" value="Use Decision Tree Classifier"><br><br>'
    for category, cols in col_names.items():
        form += '<h2>{}</h2>'.format(category)
        for col in cols:
            form += '<label for="{}">{}</label><br>'.format(col, col)
            form += '<input type="text" id="{}" name="{}"><br><br>'.format(col, col)
    form += '<input type="submit" value="Submit">'
    return form

# Define the route for the form
@app.route('/', methods=['GET', 'POST'])
def heart_attack_form():
    if request.method == 'POST':
        # Get the user inputs and store them in a Pandas DataFrame
        data = {}
        for category, cols in col_names.items():
            for col in cols:
                data[col] = request.form[col]
        user_inputs = pd.DataFrame(data, index=[0])
        
        # Use the appropriate trained model to predict the outcome
        model_name = request.form['model']
        if model_name == 'DTC':
            prediction = dtc_model.predict(user_inputs)[0]
        else:
            prediction = rfc_model.predict(user_inputs)[0]
        
        # Return the prediction as a response to the user
        return 'The predicted outcome is {}'.format(prediction)
    
    # If the request method is GET, generate the HTML form
    else:
        form = generate_form()
        return '<form method="post">' + form + '</form>'

if __name__ == '__main__':
    app.run(debug=True)
