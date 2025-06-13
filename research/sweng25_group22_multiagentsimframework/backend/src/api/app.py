from flask import Flask
from api.routes.results import results_bp
from api.routes.create import create_bp
from api.routes.catalog import catalog_bp
from api.routes.gen_config import gen_config_bp
from api.routes.del_results import del_results_bp
from api.routes.del_catalog import del_catalog_bp

from flask_cors import CORS

app = Flask(__name__)

CORS(app, supports_credentials=True)

app.register_blueprint(results_bp, url_prefix="/sim")
app.register_blueprint(create_bp, url_prefix="/sim")
app.register_blueprint(catalog_bp, url_prefix="/sim")
app.register_blueprint(gen_config_bp, url_prefix="/sim")
app.register_blueprint(del_results_bp, url_prefix="/sim")
app.register_blueprint(del_catalog_bp, url_prefix="/sim")