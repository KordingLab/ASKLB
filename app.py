##!/usr/bin/env python3
## -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:56:58 2019

@author: hanfeishen and tonyliu
"""

import os

from flask import (Flask, flash, request, redirect, url_for, 
                   send_from_directory, render_template, session)

from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash


UPLOAD_FOLDER = '/Users/hanfeishen/Desktop/Website/'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.secret_key = 'dev' # TODO change to something random for production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 #limit the size to 50Mb

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

"""

@app.route('/register', methods=('GET', 'POST'))
def register():
    """Registers the new user

    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Email is required.'
        elif not password:
            error = 'Password is required.'
        else:
            db = get_db(username)
            if db.get('password') is not None:
                error = "Email {} is already registered.".format(username)

        if error is None:
            db['password'] = generate_password_hash(password)
            db['username'] = username
            db.commit()
            return redirect(url_for('login'))

        flash(error)

    
    return render_template('register.html')


@app.route('/login', methods=('GET', 'POST'))
def login():
    """Logs in the user."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        tablenames = SqliteDict.get_tablenames(DATABASE_PATH)
        if username not in tablenames:
            error = 'Incorrect email.'
        else:
            db = get_db(username)
            if not check_password_hash(db['password'], password):
                error = 'Incorrect password.'
            
        if error is None:
            session.clear()
            session['user_id'] = db['username']
            return redirect(url_for('index'))
        
        flash(error)

    return render_template('login.html')
        

@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db(user_id)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

    
"""
DATABASE FUNCTIONS

Manages Flask interaction with the database, currently sqlitedict.

Potential scheme:
    - each table corresponds to a user
    - have additional fields in the table for user/pass, uploaded data, results
    - use sqlitedict.get_tablenames() to iterate over all datasets when training
TODO:
    - unit test
"""

from sqlitedict import SqliteDict
from flask import g

# TODO reorg directory structure
DATABASE_PATH = 'asklb.db'

def get_db(tablename):
    """Initalizes a database connection with the given table name

    Args:
        tablename (str): the table to retrieve from the sqlitedict

    Returns:
        SqliteDict: instance of SqliteDict, with autocommits set to True.
    """
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = SqliteDict(DATABASE_PATH, 
                                      tablename=tablename,
                                      autocommit=False)
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Closes the database connection on teardown request.

    Args:
        exception (?): Exception obj
    """
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
    