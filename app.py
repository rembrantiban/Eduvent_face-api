from flask import Flask, jsonify

from enroll import enroll_bp
from recognize_api import recognize_bp

from extensions import limiter

app = Flask(__name__)


limiter.init_app(app)


app.register_blueprint(enroll_bp)
app.register_blueprint(recognize_bp)

@app.errorhandler(429)
def ratelimit_handler(e):

    return jsonify({
        "success": False,
        "message":
        "Too many requests. Please try again later."
    }), 429


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )