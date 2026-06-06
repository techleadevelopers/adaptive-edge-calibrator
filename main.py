"""
Ponto de entrada do Quant Brain.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

import uvicorn
from api.server import app

def main() -> None:
    port = int(os.environ.get("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
