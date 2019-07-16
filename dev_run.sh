#!/bin/bash

# runs the application in developer mode on the specified port

export FLASK_APP=app.py
export FLASK_ENV=development

flask run --host=0.0.0.0 --port $1
