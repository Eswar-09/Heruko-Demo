import numpy as np 

from flask import Flask, request, jsonify, render_template
import pickle 

app = Flask(__name__, template_folder="templates")
model = pickle.load(open('model.pkl','rb')) # rb - binary read mode

@app.route('/', )   # it associates the root URL ('/') 
def home():
    return render_template("index.html")

#  when a user accesses the root URL of your web application, 
# Flask will execute the home() function, 
# and that function will render and return the 'index.html' template 
# as the response to be displayed in the user's web browser. 

@app.route('/predict',methods=['POST'])  # this route will be triggered when a form is submitted via POST to the '/predict' URL.
def predict():
    
    int_features = [int(x) for x in request.form.values()]   # takes all the values from the form fields, converts them to integers, and stores them in the int_features list.  
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html',prediction_text='Employee Salary should be $ {}'.format(output))

# this code handles a POST request to the '/predict' URL,
# processes the form data, makes a prediction using a machine learning model, 
# and then displays the result on the web page using the 'index.html' template.

if __name__ == '__main__':   # this is the main function thate runs the whole function
    app.run(debug=True)
    
