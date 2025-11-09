from os import environ
from dotenv import load_dotenv

load_dotenv()

class Config:
        SECRET_KEY = environ.get('SECRET_KEY')
        DEBUG = environ.get('FLASK_DEBUG')