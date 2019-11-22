#!/bin/bash

# runs the application in developer mode on the specified port

export FLASK_APP=asklb
export FLASK_ENV=development

pip install -e .
flask run --host=0.0.0.0 --port $1
