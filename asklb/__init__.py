"""
init file for running the flask app.

@author: hanfeishen, tonyliu, jessecui
"""

from flask import (Flask, flash, request, redirect, url_for, 
                   send_from_directory, render_template, session, g)
import pymongo


app = Flask(__name__)
app.secret_key = 'dev' # TODO change to something random for production

# Database setup
mongo_uri = "mongodb://admin:super_secret_password@localhost:27018/" 
client = pymongo.MongoClient(mongo_uri)
db = client.asklb_test

import asklb.auth
import asklb.home
import asklb.upload




