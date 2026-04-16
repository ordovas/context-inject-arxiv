"""
Flask REST API application for ArXiv paper retrieval.

This is an inbound adapter that translates HTTP requests into
domain operations and returns JSON responses.
"""
from flask import Flask
from src.adapters.inbound.rest_api.routes import api_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Error handler
    @app.errorhandler(404)
    def not_found(error):
        return {
            'success': False,
            'error': 'Endpoint not found'
        }, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {
            'success': False,
            'error': 'Internal server error'
        }, 500
    
    return app


def run_server(host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
    """
    Run the REST API server.
    
    Args:
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
