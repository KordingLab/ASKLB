#!/bin/bash

conda install gxx_linux-64 gcc_linux-64 swig -y
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
pip install auto-sklearn===0.5.2 2> /dev/null
pip install bcrypt