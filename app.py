##!/usr/bin/env python3
## -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:56:58 2019

@author: hanfeishen and tonyliu
"""

import os

from flask import (Flask, flash, request, redirect, url_for, 
                   send_from_directory, render_template, session, g)

import pymongo
import bcrypt
import urllib.parse

from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

# TODO integrate flask-upload
UPLOAD_FOLDER = '/Users/hanfeishen/Desktop/Website/'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.secret_key = 'dev' # TODO change to something random for production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 #limit the size to 50Mb

# Database setup
mongo_uri = "mongodb://admin_jesse:long_password_for_admin_jesse_2019@localhost:27017/" #TODO hook up quadcorn to mongdb
client = pymongo.MongoClient(mongo_uri)
db = client.asklb_test

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('wait.html')
#            return redirect(url_for('uploaded_file',
#                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


"""
REGISTRATION FUNCTIONS

Manages user creation and registration.

TODO:
    - enforce usernames to be emails
    - unit tests
    - check "email already registered" bug: sometimes triggers without actually registering the user
"""

@app.route('/register', methods=('GET', 'POST'))
def register():
    """Registers the new user"""

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        if not username:
            error = 'Email is required.'
        elif not password:
            error = 'Password is required.'
        else:
            users = db.users
            existing_user = users.find_one({'name' : request.form['username']})
    
            if existing_user is None:
                hashpass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                users.insert({'name' : username, 'password' : hashpass})
                session['user_id'] = username
                return redirect(url_for('index'))
            
            else:
                error = 'That username already exists!'
            
        flash(error)
            
    else:
        if 'user_id' in session:
            return 'You are already logged in as ' + session['user_id']
        else:
            return render_template('register.html')


@app.route('/login', methods=('GET', 'POST'))
def login():
    """Logs in the user."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        if request.method == 'POST':
            users = db.users
            login_user = users.find_one({'name' : username})
    
            if login_user:
                if bcrypt.hashpw(password.encode('utf-8'), 
                                 login_user['password']) == login_user['password']:
                    session.clear()
                    session['user_id'] = request.form['username']
                    return redirect(url_for('index'))
                else:
                    error = 'Incorrect password'
            else:
                error = 'Incorrect email'
        
            flash(error)
            
    else:
        if 'user_id' in session:
            return 'You are already logged in as ' + session['user_id']
        else:
            return render_template('login.html')  
        

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.before_request
def load_user():
    if 'user_id' in session:
        user = {'username': db.users.find_one({'name' : session["user_id"]})['name']}
    else:
        user = None

    g.user = user


