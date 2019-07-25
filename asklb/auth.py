"""
Contains flask views for user registration and authentication.
"""

"""
REGISTRATION FUNCTIONS

Manages user creation and registration.

TODO:
    - enforce usernames to be emails
    - unit tests
    - check "email already registered" bug: sometimes triggers without actually registering the user
"""
from asklb import app, db

from flask import (flash, request, redirect, url_for, 
                   send_from_directory, render_template, session, g)

import bcrypt


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
