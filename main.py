"""CiteVerify standalone application runner."""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "127.0.0.1")
    print(f"[CiteVerify] Starting application server on http://localhost:{port}")
    uvicorn.run("api.index:app", host=host, port=port, reload=False)
