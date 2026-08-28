from waitress import serve
from app import app
import os

if __name__ == "__main__":
    # Ensure uploads directory exists
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting production WSGI server on port {port}...")
    print("Models will be loaded into memory before the first request.")
    
    # Serve using waitress
    # threads=4 gives a good balance for simple I/O and blocking operations on a simple VPS
    serve(app, host='0.0.0.0', port=port, threads=4)
