from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GA-ready Pygame racing prototype")
    parser.add_argument("--headless", action="store_true", help="run without a visible window")
    parser.add_argument("--frames", type=int, default=None, help="stop after this many frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    from racing.game import RacingGame

    RacingGame().run(max_frames=args.frames)


if __name__ == "__main__":
    main()
