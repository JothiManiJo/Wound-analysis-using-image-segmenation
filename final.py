from flask import Flask,render_template,request,redirect,url_for,session
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import mysql.connector

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mydb = mysql.connector.connect(host="localhost",user="root",password="",database="covid")
mycursor = mydb.cursor()
@app.route('/')
def homepage():
    return render_template('index1.html')
@app.route("/Nnewuser")
def Nnewuser():
    return render_template('newuser.html')
@app.route("/user", methods=['GET', 'POST'])

@app.route("/user", methods=['GET', 'POST'])
def user():
    error = None
    global data1
    if request.method == 'POST':
        data1 = request.form.get('name')
        data2 = request.form.get('password')
        session['name'] = request.form['name']
        sql = "SELECT * FROM `reg` WHERE `name` = %s AND `password` = %s"
        val = (data1, data2)
        mycursor.execute(sql, val)
        account = mycursor.fetchall()
        if account:
            return render_template('index2.html')
        else:
            return  render_template('wrong.html')
@app.route("/Newuser", methods=['GET', 'POST'])
def Admin():
    if request.method == 'POST':
        name = request.form['name']
        email=request.form['email']
        password = request.form['password']
        cpassword=request.form['cpassword']

        insertQuery = "INSERT INTO reg VALUES ('" + name + "','" + email + "','" + password + "','" + cpassword + "')"
        mycursor.execute(insertQuery)
        mydb.commit()

    return render_template('index1.html')



#classes = ['Blast', 'Blight', 'Brown Spot', 'Leaf Smut', 'Tungro']
#classes = ['Bacterialblight', 'Blast', 'Brown Spot', 'Tungro']
classes = ['abdominal-wounds', 'burns', 'epidermolysis-bullosa', 'extravasation-wound-images','foot-ulcers','haemangioma','leg-ulcer-images','malignant-wound-images','meningitis','miscellaneous','orthopaedic-wounds','pilonidal-sinus','pressure-ulcer-images-a','pressure-ulcer-images-b','toes']
@app.route('/')
def index():
    return render_template('index1.html')
@app.route('/in',methods=['GET', 'POST'])
def ind():
    if request.method == 'POST':
        if request.form.get('text1') == 'abc' and request.form.get('password') == '123':
            return render_template('index2.html')
        else:
            return render_template('index1.html', msg='Invalid Username or Password')

    else:
        return render_template('index1.html')
    #return render_template('index.html')

@app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        global imgfile
        global result
        file1 = request.files['filename']
        imgfile = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(imgfile)
        #model = load_model('trained_model.h5')
        model = load_model('model_inception.h5')
        img_ = image.load_img(imgfile, target_size=(224, 224))
        img_array = image.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.
        prediction = model.predict(img_processed)
        print(prediction)
        index = np.argmax(prediction)
        result = str(classes[index]).title()
        #sy = sym[result]
        #fer = ferti[result]
        return render_template('index2.html', msg = result, src = imgfile, view = 'style=display:block')
@app.route('/detect')
def detect():
    image = cv2.imread(imgfile)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return render_template("success.html",result=result)


if __name__ == '__main__':
    app.run(debug=True,port=4000)

