"""WSGI entrypoint for Gunicorn-based deployments."""

from kg_rag_app import create_app_from_env


app = create_app_from_env()
