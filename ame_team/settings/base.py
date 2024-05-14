# BASE_DIR/ame/settings/base.py

from pathlib import Path
import os
from dotenv import load_dotenv
from decouple import config

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
# SECRET_KEY = config('DJANGO_SECRET_KEY')

GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', os.path.join(BASE_DIR.parent, 'sec', 'serviceAccountKeys.json'))

# Application definition
ASYNC_PROCESSING = "True"
ASGI_APPLICATION = "common.socketio_app.app"

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

# Static and media files settings
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

SITE_ID = 1

WSGI_APPLICATION = "aidream.wsgi.application"
ROOT_URLCONF = "aidream.urls"
