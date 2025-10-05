"""Server module for running the FastAPI application with static file serving."""
from pathlib import Path
import uvicorn
from fastapi.staticfiles import StaticFiles

from api import create_app


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server with static file serving."""
    app = create_app()

    # Mount static files directory for web interface
    static_dir = Path(__file__).parent / "web"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="web")

    print(f"Starting server at http://localhost:{port}")
    print(f"API documentation available at http://localhost:{port}/docs")

    uvicorn.run(app, host=host, port=port)
