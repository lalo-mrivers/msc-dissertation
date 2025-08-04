import uvicorn
from rag.api import app as rags_app
import argparse

if __name__ == "__main__":
    """
    Start the FastAPI server for the RAG systems.
    """
    parser = argparse.ArgumentParser(description="Run RAG system")
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to run the RAG API server on"
    )
    args = parser.parse_args()

    print("Starting API RAG system...")
    uvicorn.run(rags_app, host="0.0.0.0", port=args.port, log_level="debug", timeout_keep_alive=3000)
