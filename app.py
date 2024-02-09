from flask import Flask, render_template, request, session
from flask_mysqldb import MySQL
import MySQLdb.cursors

app = Flask(__name__)
 
 
@app.route('/')
@app.route('/reg.html')
def register():
    return render_template('reg.html')


@app.route('/login.html')
def login():

    return render_template("login.html")
 
 


@app.route('/index.html')
def index():
    return render_template('index.html')



if __name__ =="__main__":
    app.run(debug=True,port=8000)
