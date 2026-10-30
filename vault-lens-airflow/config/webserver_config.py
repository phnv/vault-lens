"""Default Airflow webserver configuration (Flask-AppBuilder)."""
from __future__ import annotations

import os

from flask_appbuilder.const import AUTH_DB

basedir = os.path.abspath(os.path.dirname(__file__))

# Flask-WTF flag for CSRF
WTF_CSRF_ENABLED = True
WTF_CSRF_TIME_LIMIT = None

# Authentication type: user/password stored in the Airflow metadata DB
AUTH_TYPE = AUTH_DB
