from flask import Flask, render_template, request, session,Response
from flask_mysqldb import MySQL
import MySQLdb.cursors
import cv2
from keras.models import model_from_json
import numpy as np
import pickle
from imutils.video import WebcamVideoStream

app = Flask(__name__)
webcam=cv2.VideoCapture(0)

def generate_frames():   
    json_file = open('emotiondetector.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights('emotiondetector.h5')
    haar_file = cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier("C:/Python39/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

    def extract_features(image):
        feature = np.array(image)
        feature = feature.reshape(1,48,48,1)
        return feature/255.0


    labels = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
    while True:
        i,im = webcam.read()
        if not i:
            break
        else:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
            try:
                for(p,q,r,s) in faces:
                    image = gray[q:q+s, p:p+r]
                    cv2.rectangle(im, (p,q), (p+r, q+s), (255,0,0), 2)
                    image=cv2.resize(image, (48,48))
                    img = extract_features(image)
                    pred = model.predict(img)
                    prediction_label = labels[pred.argmax()]
                    #print(prediction_label)
                    #cv2.putText(im, prediction_label)
                    cv2.putText(im,'% s' %(prediction_label), (p-10,q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
                
                ret,buffer = cv2.imencode('.jpg',im)
                frame = buffer.tobytes()
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except cv2.error:
                print('Please check your webcam',cv2.error)
                break
            
            
 
#========================================================================================================================
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')
#========================================================================================================================
@app.route('/login')
def login():
    return render_template("login.html")
#========================================================================================================================
@app.route('/reg')
def register():
    return render_template('reg.html')
#========================================================================================================================
@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')
#=======================================================================================================================
@app.route('/feedback')
def feedback():
    return render_template("feedback.html")
#========================================================================================================================

if __name__ =="__main__":
    app.run(debug=True,port=8000)
