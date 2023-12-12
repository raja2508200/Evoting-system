from flask import Flask, render_template, request, redirect, url_for, session,flash
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from flask_session import Session 
import sqlite3
import easygui
from cv2 import *
import cv2
import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import tqdm
import glob
import tensorflow
import random
import smtplib
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import numpy as np
import csv
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
print("hi")
app = Flask(__name__)
app.secret_key = "12345678989"
access=["no"]
# Configuration for SQLite database
DATABASE = 'database76.db'
# Mock user database

def create_table():
    try:
        conn = sqlite3.connect(DATABASE)    
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT NOT NULL ,
                          email TEXT NOT NULL,
                          voter_id TEXT NOT NULL)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS login (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          username TEXT NOT NULL ,
                          email TEXT NOT NULL,
                          voter_id TEXT NOT NULL)''')
        print("hi")
        cursor.execute('''CREATE TABLE IF NOT EXISTS nominees (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                nominee_name TEXT,zone TEXT,
                                voter_id INTEGER ) ''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS votes (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                nominee_id INTEGER,
                                FOREIGN KEY (nominee_id) REFERENCES nominees (id))''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print("Error occurred:", e)
    finally:
        if conn:
            conn.close()
create_table()
@app.route('/')
@app.route('/vote', methods=['GET', 'POST'])
@app.route('/index')
def index():
        return render_template('index.html')
@app.route('/user', methods=['GET', 'POST'])
def user():
        return render_template('user.html')
@app.route('/success')
def success():
        return render_template('success.html')  
@app.route('/complete')
def complete():
        return render_template('complete.html')


user=[]
voter_id =[]

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username=request.form['username']
        user.append(username)
        print(username)
        voter = request.form['voter']
        voter_id .append(voter)
        v1=str(voter)
        print( len(v1))
        if len(v1)!=6:
            return display_popup1(" please enter a valid 6 digit  voter id ")
        email = request.form['email']
        confirm_email = request.form['confirm_email']
        if email != confirm_email:
            return display_popup1(" your email does not match")
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email=?", (email,))
        registered = cursor.fetchall()
        print(registered)
        cursor.execute("SELECT voter_id  FROM users WHERE voter_id=?", (voter,))
        registered1 = cursor.fetchall()
        print(registered1)
        if registered:
            return display_popup1(" your email already registered")
        elif registered1:
            return display_popup1("your voter_id  already registered")
        else:
            cursor.execute("INSERT INTO users (username, email,voter_id ) VALUES (?, ?,?)", (username, email,voter))
            conn.commit()
            conn.close()
            return render_template('data.html')
    return render_template('register.html')



@app.route('/nomine', methods=['GET', 'POST'])
def nominee_form():
    if request.method == 'POST':
        nominee_name = request.form['nominee']
        zone = request.form['zone']
        voter_id = request.form['voter']
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO nominees (nominee_name,zone,voter_id) VALUES (?,?,?)', (nominee_name,zone,voter_id))
        conn.commit()
        conn.close()
        return redirect(url_for('nominee_form'))
    return render_template('nominee_form.html')


    
                        
@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        username = user[-1]
        voter_id1= voter_id [-1]
        cam = cv2.VideoCapture(0)
        
        if not cam.isOpened():
            print("Failed to open webcam.")
            return render_template('error.html', message="Failed to open webcam.")
        
        time.sleep(5)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        
        while True:
            ret, img = cam.read()
            
            if not ret:
                print("Failed to capture frame from webcam.")
                return "Failed to capture frame from webcam."
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum +=1
                cv2.imwrite(f"TrainingImages/{sampleNum}.{voter_id1}.jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
                print("hello")
                
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 30:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        
    return render_template('success.html')

def display_popup2(message):
    flash(message)
    return redirect(url_for('index'))
def display_popup1(message):
    flash(message)
    return redirect(url_for('register'))
def display_popup(message):
    flash(message)
    return redirect(url_for('login'))
u=[]
p=[]
a=[]
l=[]
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if access[-1]=="yes":                                        
                username = request.form['username']
                u.append(username)      
                email = request.form['email']
                p.append(email)
                voter_id  = request.form['voter']
                print(voter_id ,":voter_id ")
                a.append(voter_id )
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM login WHERE voter_id =? AND email=?", (voter_id ,email))
                user1 = cursor.fetchone()
                if user1:
                    return display_popup("You have already voted  Don't cheat")
                else:
                    cursor.execute("SELECT * FROM users WHERE voter_id =? AND email=?", (voter_id , email))
                    user = cursor.fetchone()
                    if user:
                        recognizer = cv2.face_LBPHFaceRecognizer.create()#cv2.createLBPHFaceRecognizer()
                        recognizer.read("train2.yml")
                        harcascadePath = "haarcascade_frontalface_default.xml"
                        faceCascade = cv2.CascadeClassifier(harcascadePath)
                        cam = cv2.VideoCapture(0)
##                        cv2.imshow('frame', img)
                        while True:
                                ret,im =cam.read()
                                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                                faces = faceCascade.detectMultiScale(gray, 1.3,5)
                                name2=[]
                                for(x, y, w, h) in faces:
                                    cv2.rectangle(im,(x, y), (x + w, y + h), (255,0,0), 2)
                                    Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                                    print(Id,conf)
                                    if conf < 50:
                                        name=Id
                                        name1=str(name)
                                        print(name1)
                                        voter_id1=str(voter_id)
                                        print(voter_id1)
                                        if (name1==voter_id1):
                                            print("after",name1,voter_id1)
                                            cursor.execute("SELECT email FROM users WHERE voter_id =? AND email=?", (voter_id , email))
                                            user2 = cursor.fetchone() 
                                            print(user2)
                                            smtp_server = 'smtp.example.com'
                                            smtp_port = 587
                                            sender_email = 'diwa.2801@gmail.com'
                                            sender_password = 'furgqbokcooqfjkf'
                                            receiver_email = user2
                                            otp = ""
                                            for _ in range(6):
                                                otp += str(random.randint(0, 9))
                                                l.append(otp)
                                            print(l[-1])
                                            host = "smtp.gmail.com"
                                            mmail = "diwa.2801@gmail.com"        
                                            hmail = user2[0]
                                            receiver_name = username
                                            sender_name= "election commision "
                                            msg = MIMEMultipart()
                                            subject = "YOUR OTP CODE"
                                            text =  f'Your OTP code is: {otp}'
                                            msg = MIMEText(text, 'plain')
                                            msg['To'] = formataddr((receiver_name, hmail))
                                            msg['From'] = formataddr((sender_name, mmail))
                                            msg['Subject'] = 'Hello  ' + receiver_name
                                            server = smtplib.SMTP(host, 587)
                                            server.ehlo()
                                            server.starttls()
                                            password = " furgqbokcooqfjkf"
                                            server.login(mmail, password)
                                            server.sendmail(mmail, [hmail], msg.as_string())
                                            server.quit()
                                            print('send')
                                            return redirect(url_for('OTP'))
                                            cam.release()
                                            cv2.destroyAllWindows()
                                            break
                                        else: 
                                            return display_popup("User face mismatch")
                                    else:
                                        print("outof range")
                    else:
                        return display_popup("Not registered")
        
        else:
             return display_popup( "The website is currently not functioning")
    return render_template('login.html')


@app.route('/OTP', methods=['GET', 'POST'])
def OTP():
    if request.method == 'POST':
        otp1 = request.form['otp']
        print(otp1)
        if otp1==l[-1]:
            l.clear()
            print(l)
            return redirect(url_for('voting_form'))
        else:
            print("not valid")
    return render_template('OTP.html')
        

@app.route('/voting', methods=['GET', 'POST'])
def voting_form():
    username=u[0]
    email=p[0]
    voter_id =a[0]
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM nominees')
    nominees = cursor.fetchall()

    cursor.execute('SELECT nominee_id, COUNT(id) as vote_count FROM votes GROUP BY nominee_id')
    votes = {nominee_id: vote_count for nominee_id, vote_count in cursor.fetchall()}
    if request.method == 'POST':
        nominee_id = int(request.form['nominee'])
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO votes (nominee_id) VALUES (?)', (nominee_id,))
        cursor.execute("INSERT INTO login (username,email,voter_id ) VALUES (?, ?,?)", (username, email,voter_id ))
        conn.commit()
        conn.close()
        p.clear()   
        u.clear()
        a.clear()
        return display_popup2(" your vote has been sucessfully registered")
    return render_template('voting_form.html', nominees=nominees, votes=votes)

@app.route('/result1')
def result1():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT nominees.nominee_name, COUNT(votes.id) as vote_count
        FROM nominees LEFT JOIN votes ON nominees.id = votes.nominee_id
        GROUP BY nominees.nominee_name
        ORDER BY vote_count DESC
    ''')
    result = cursor.fetchall()
    print(result)
    conn.close()

    # Find the nominee with the maximum votes
    winner = max(result, key=lambda x: x[1])
    print(winner)
    for naminee ,votes in result:
        print(naminee ,votes)
        
    return render_template('result1.html', result=result, winner=winner)


ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            # Redirect to admin panel or dashboard after successful login
            return redirect('/details')
    return render_template('admin.html')


print(a)
@app.route('/details',methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        agree = request.form.get('agree')
        print(agree)
        access.append(agree)
        print(access)
    
    return render_template('details.html')


def get_table_data3():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Replace 'your_table_name' with the actual name of the database table you want to display
    cursor.execute("SELECT * FROM users")
    table_data = cursor.fetchall()

    conn.close()
    return table_data

@app.route('/table')
def table():
    table_data = get_table_data3()

    return render_template('table.html', table_data=table_data)


def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create LBPH Face Recognizer
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Id))
    print(np.array(Id))
    recognizer.save("train2.yml")
    print("trained")

def getImagesAndLabels(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
        faces=[]
        Ids=[]
        for imagePath in imagePaths:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        return faces,Ids


@app.route('/train', methods=['GET'])
def train():
    return render_template('train.html')

@app.route('/training', methods=['POST'])
def training():
    # Train the model
    TrainImages()

    # Redirect to the success page
    return render_template('complete.html')


def get_table_data1():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nominees")
    table_data = cursor.fetchall()
    conn.close()
    return table_data

@app.route('/nomine_list')
def nomine_list():
    table_data = get_table_data1()
    return render_template('nomine_list.html', table_data=table_data)


def get_table_data5():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM login")
    table_data = cursor.fetchall()
    conn.close()
    return table_data

@app.route('/voted_ist')
def voted_ist():
    table_data = get_table_data5()
    return render_template('voted_ist.html', table_data=table_data)







if __name__ == '__main__':
    app.run(debug=False,port=250)

































