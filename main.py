import logging
import os
from pathlib import Path

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=f"{Path(__file__).parent}/logs/retraining.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


if __name__ == "__main__":

    import src.web.main

    logging.info("Starting execution of webapp from %s", __name__)
    src.web.main.execute()
