"""Launch with holonomy-explorer or python -m holonomy_explorer."""

from __future__ import annotations

import argparse
import os
import webbrowser

import torch

from .common import Context
from .server import create_server

# Local-only defaults requested by the explorer brief; every deployment can override.
DEFAULT_HOST = "127.0.0.1"
WILDCARD_HOST = "0.0.0.0"  # IPv4 bind-all address; browse via loopback instead.
DEFAULT_PORT = 8765
DEFAULT_DTYPE = "float64"
MAX_PORT = 65535  # TCP port field is an unsigned 16-bit integer.


def main() -> None:
    """Parse deployment settings, print the bound URL, and serve until Ctrl-C."""
    parser = argparse.ArgumentParser(
        description="Local holonomy_lib explorer. Offline UI; CPU/CUDA tensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOLONOMY_EXPLORER_HOST", DEFAULT_HOST),
        help="bind address; env HOLONOMY_EXPLORER_HOST",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=os.getenv("HOLONOMY_EXPLORER_PORT", str(DEFAULT_PORT)),
        help="TCP port, 0 chooses a free port; env HOLONOMY_EXPLORER_PORT",
    )
    parser.add_argument(
        "--device",
        default=os.getenv(
            "HOLONOMY_EXPLORER_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
        help="torch device; env HOLONOMY_EXPLORER_DEVICE; ROCm also uses cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default=os.getenv("HOLONOMY_EXPLORER_DTYPE", DEFAULT_DTYPE),
        help="tensor precision; env HOLONOMY_EXPLORER_DTYPE",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print URL without opening a browser"
    )
    args = parser.parse_args()
    if not 0 <= args.port <= MAX_PORT:
        parser.error(f"port must be between 0 and {MAX_PORT}")
    try:
        context = Context.from_request({"device": args.device, "dtype": args.dtype})
        server = create_server(args.host, args.port, str(context.device), args.dtype)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    # Wildcard bindings are not navigable destinations; loopback reaches this server.
    browser_host = DEFAULT_HOST if args.host == WILDCARD_HOST else args.host
    url = f"http://{browser_host}:{server.server_address[1]}"
    print(f"Holonomy explorer: {url} ({context.device}, {args.dtype})", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
