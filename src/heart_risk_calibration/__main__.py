"""Allow `python -m heart_risk_calibration <subcommand>`."""
from __future__ import annotations

import sys

from heart_risk_calibration.cli import main

if __name__ == "__main__":
    sys.exit(main())
