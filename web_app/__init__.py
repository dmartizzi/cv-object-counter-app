from flask import Flask
app = Flask(__name__)
from web_app import views
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300
# No cacheing at all for API endpoints.

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response