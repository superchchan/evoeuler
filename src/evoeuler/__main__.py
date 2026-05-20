"""Run the Streamlit application via the "evoeuler" command."""

from __future__ import annotations

import sys
from importlib.resources import as_file, files

from streamlit.web import cli as stcli


def main() -> None:
    """Run the Streamlit application."""
    app = files("evoeuler.gui").joinpath("app.py")
    with as_file(app) as app_path:
        sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
