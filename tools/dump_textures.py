"""Save every procedurally generated texture as a PNG (used for the report).

    python tools/dump_textures.py [output_dir]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame  # noqa: E402

from core import textures as T  # noqa: E402
from sports.football import FOOTBALLER_LOOK  # noqa: E402
from sports.basketball import BASKETBALLER_LOOK  # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "texture_dump"
    os.makedirs(out, exist_ok=True)
    pygame.init()
    items = {
        "football": T.football(),
        "basketball": T.basketball(),
        "jersey_football": T.jersey(**FOOTBALLER_LOOK["jersey"]),
        "jersey_basketball": T.jersey(**BASKETBALLER_LOOK["jersey"]),
        "face_football": T.face(**FOOTBALLER_LOOK["face"]),
        "face_basketball": T.face(**BASKETBALLER_LOOK["face"]),
        "socks": T.socks(**FOOTBALLER_LOOK["socks"]),
        "pitch": T.pitch(),
        "court": T.court(),
        "backboard": T.backboard(),
        "grass_tile": T.grass_tile(),
    }
    for name, arr in items.items():
        pygame.image.save(T.array_to_surface(arr), os.path.join(out, name + ".png"))
        print("saved", name, arr.shape)


if __name__ == "__main__":
    main()
