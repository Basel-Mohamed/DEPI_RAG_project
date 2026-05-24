"""Generate the Grafana dashboard JSON file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from app.services.monitoring.grafana_service import generate_dashboard_json

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output = Path(__file__).parent.parent / "monitoring" / "grafana_dashboard.json"
    generate_dashboard_json(output)
    logger.info("dashboard written to %s", output)
    print(f"Dashboard written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
