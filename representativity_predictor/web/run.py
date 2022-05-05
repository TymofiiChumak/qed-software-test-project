from flask import Flask
from .model import model_app

app = Flask(__name__)
app.register_blueprint(model_app)

if __name__ == "__main__":
    app.run()
