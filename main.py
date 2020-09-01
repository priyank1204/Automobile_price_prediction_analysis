##importing required libraries
from flask import Flask,flash, render_template, request,redirect
import flask_monitoringdashboard as dashboard
from werkzeug.utils import secure_filename
import csv
from predictionfolder.prediction import predict
from logs.logger import App_Logger
import requests
import pandas as pd
import joblib
from retraining import retraining



##for logging
flask_log = App_Logger()
file_object = open('./flask_logs.txt','a+')
flask_log.log(file_object,"starting user interface")


##setting allowed files criteria
flask_log.log(file_object,"setting allowed files extensions for file input")
ALLOWED_EXTENSIONS = set(['csv','xlsx','data'])

UPLOAD_FOLDER = './Charts'

##function to check whether file is is allowed extensions or not
def allowed_file(filename):
    file_object = open('./flask_logs.txt', 'a+')
    flask_log.log(file_object, "checking if file is in correct extension")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
##inintialization of flask app instance
flask_log.log(file_object,"initializing flask app")
app = Flask(__name__)
dashboard.bind(app)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

##loading best model
flask_log.log(file_object,"Loading the best model for predictions")
model = joblib.load("model.pkl")
instance = predict()


##redirecting to root template
@app.route('/')
def upload_form():
    file_object = open('./flask_logs.txt', 'a+')
    flask_log.log(file_object, "Entering to the first template ")
    file_object.close()
    return render_template('start.htm')


@app.route('/',methods = ['POST'])
def start():
    file_object = open('./flask_logs.txt', 'a+')
    flask_log.log(file_object, "Now th user will choose for input type")
    if request.method == 'POST':
        input_type = request.form['input_type']
        if input_type=='single':
            flask_log.log(file_object, "User has choosen for single inputs")
            file_object.close()
            return render_template('my_index.htm')
        elif input_type == 'file':
            flask_log.log(file_object, "user has choosen for file upload predictions or bulk predictions")
            file_object.close()
            return render_template('upload.html')
        else:
            flask_log.log(file_object, "invalid choice !!! try again")
            file_object.close()
            return redirect('/')    

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file_object = open('./flask_logs.txt', 'a+')
    try:
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flask_log.log(file_object, "File is not in the request methods")
                flash('No file part')
                file_object.close()
                return redirect(request.url)
            flask_log.log(file_object, "Now input file from user")
            file = request.files['file']
            if file.filename == '':
                flash('No file selected for uploading')
                flask_log.log(file_object, "user has not selected any file for upload")
                file_object.close()
                return redirect(request.url)
            if file and allowed_file(file.filename):
                flask_log.log(file_object, "Everthings fine ---- prediction starts for file")
                data= instance.predictor(file)
                flask_log.log(file_object, "Model retrained succesfully")
                file_object.close()
                return render_template('simple.html',  tables=[data.to_html(classes='data')], titles=data.columns.values)
            else:
                flask_log.log(file_object, "input file from the user is not in correct extensions")
                file_object.close()
                flash('Allowed file types are csv,xlsx')
                return redirect(request.url)
    except Exception as e:
        flask_log.log(file_object, "looks like an error occured in predicting results")
        file_object.close()
        raise e


@app.route('/predict',methods = ['POST'])
def predict():
    file_object = open('./flask_logs.txt', 'a+')
    try:
        flask_log.log(file_object, "predictions for single inputs starts")
        drive_wheels_rwd=0
        num_of_cylinder_five=0
        num_of_cylinder_four=0
        num_of_cylinder_six=0
        num_of_cylinder_three=0
        num_of_cylinder_twelve=0
        num_of_cylinder_two=0
        if request.method == 'POST':
            flask_log.log(file_object, "Input each feature manually")
            length = int(request.form['length'])
            width = int(request.form['width'])
            horsepower = int(request.form['horsepower'])
            curb_weight=int(request.form['curb_weight'])
            engine_size=int(request.form['engine_size'])
            city_mpg=int(request.form['city_mpg'])
            highway_mpg =int(request.form['highway_mpg'])
            drive_wheels_fwd=request.form['drive_wheels_fwd']
            flask_log.log(file_object, "all features ar ecorrectly input")
            if(drive_wheels_fwd=='fwd'):
                drive_wheels_fwd=1
                drive_wheels_rwd=0
            elif(drive_wheels_fwd=='rwd'):
                drive_wheels_fwd=0
                drive_wheels_rwd=1
            else:
                drive_wheels_fwd=0
                drive_wheels_rwd=0
            area = length*width
            num_of_cylinder=request.form['num_of_cylinder']
            if(num_of_cylinder=='five'):
                num_of_cylinder_five =1
            elif(num_of_cylinder=='four'):
                num_of_cylinder_four=1
            elif(num_of_cylinder=='six'):
                num_of_cylinder_six=1
            elif(num_of_cylinder=='three'):
                num_of_cylinder_three=1
            elif(num_of_cylinder=='twelve'):
                num_of_cylinder_twelve=1
            elif(num_of_cylinder=='two'):
                num_of_cylinder_two=1
            miles = city_mpg-highway_mpg
            flask_log.log(file_object, "prediction for single input starts using saved model")
            prediction = model.predict([[horsepower,curb_weight,engine_size,miles,area,drive_wheels_fwd,drive_wheels_rwd,num_of_cylinder_five,num_of_cylinder_four,num_of_cylinder_six,num_of_cylinder_three]])
            flask_log.log(file_object, "we have got the results for the predictions")
            output=round(prediction[0],2)
            if output<0:
                flask_log.log(file_object, "Invalid entry for any features")
                file_object.close()
                return render_template('results.htm',prediction_text="Invalid info")
            else:
                flask_log.log(file_object, "viewing th eoutput on the screen")
                file_object.close()
                return render_template('results.htm',prediction_text="The Car price is {}".format(output))
        else:
            return render_template('my_index.htm')

    except Exception as e:
        flask_log.log(file_object, "prediction for single input not succesfull")
        file_object.close()
        raise e


@app.route('/retrain',methods  =['POST'])
def retrain():
    file_object = open('./flask_logs.txt', 'a+')
    try:
        flask_log.log(file_object, "model retraining phase entering")
        if request.method == "POST":
            if 'file' not in request.files:
                return render_template('simple.html')
            file = request.files['file']
            if file.filename == '':
                flash('No file selected for uploading')
                return render_template('start.htm')
            if file and allowed_file(file.filename):
                flask_log.log(file_object, "everythings fine --- model retrainig starts")
                result = retraining(file)
                result.retrain_model()
                flask_log.log(file_object, "model retrained succesfully")
                file_object.close()
                return render_template('start.htm',text = 'model_retrained_succesfully')
            else:
                flash('Allowed file types are csv,xlsx')
                return render_template('start.htm')
    except Exception as e:
        flask_log.log(file_object, "ooops liiks like ther eiis some error in model retraining---------------try again after fixing the error")
        file_object.close()
        return render_template('start.htm', text  =e)



    s
file_object = open('./flask_logs.txt', 'a+')
flask_log.log(file_object, "ending of the flask app")
file_object.close()
if __name__ == "__main__":
    ##for web
    ##app.run(host = '0.0.0.0',port = 8080)
    ##for local
    app.run(debug=True)
