"""python -m recipe_opt_web"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Recipe Opt Agent web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes (dev only)",
    )
    args = parser.parse_args(argv)

    import uvicorn

    # Import string form is required for reload to pick up route changes.
    uvicorn.run(
        "recipe_opt_web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
