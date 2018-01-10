#activate_this = '/home/frank/.virtualenvs/flask/bin/activate_this.py'
e#xecfile(activate_this, dict(__file__=activate_this))

import sys
sys.path.insert(0, '/home/frank/Development/udacity/deeplearning/part3/project/dog-project/upload_test')

from server import app as application
