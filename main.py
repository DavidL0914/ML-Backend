import threading
from flask import render_template, request, jsonify
from flask.cli import AppGroup
from __init__ import app, db, cors
from api.ad import predict_api

db.init_app(app)

app.register_blueprint(predict_api)
@app.errorhandler(404)
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/settings/')
def settings():
    return render_template("settings.html")
@app.before_request
def before_request():
    allowed_origin = request.headers.get('Origin')
    if allowed_origin in ['http://localhost:4100', 'http://127.0.0.1:4100', 'https://davidl0914.github.io']:
        cors._origins = allowed_origin

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="8008")
