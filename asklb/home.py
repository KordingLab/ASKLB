"""
Default flask views for the index and instructions.

TODO is this file needed?
"""

from asklb import app

from flask import (render_template)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/form')
def form():
    return render_template('form.html')
