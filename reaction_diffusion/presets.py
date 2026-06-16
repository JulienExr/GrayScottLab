from __future__ import annotations

PRESETS: dict[str, dict] = {
    "spots": {
        "Du": 0.2097,
        "Dv": 0.1050,
        "F": 0.0365,
        "k": 0.0600,
        "steps_per_frame": 8,
        "description": "Isolated circular spots",
    },
    "maze": {
        "Du": 0.2097,
        "Dv": 0.1050,
        "F": 0.0290,
        "k": 0.0570,
        "steps_per_frame": 8,
        "description": "Labyrinthine maze channels",
    },
    "coral": {
        "Du": 0.2097,
        "Dv": 0.1050,
        "F": 0.0580,
        "k": 0.0650,
        "steps_per_frame": 8,
        "description": "Branching coral structures",
    },
    "worms": {
        "Du": 0.2097,
        "Dv": 0.1050,
        "F": 0.0390,
        "k": 0.0580,
        "steps_per_frame": 10,
        "description": "Squiggly worm-like patterns",
    },
    "cells": {
        "Du": 0.2097,
        "Dv": 0.1050,
        "F": 0.0260,
        "k": 0.0510,
        "steps_per_frame": 6,
        "description": "Cell division pattern",
    },
    "unstable": {
        "Du": 0.2097,
        "Dv": 0.1050,
        "F": 0.0620,
        "k": 0.0609,
        "steps_per_frame": 12,
        "description": "Unstable chaotic regime",
    },
    "fingerprints": {
        "Du": 0.1900,
        "Dv": 0.0500,
        "F": 0.0600,
        "k": 0.0625,
        "steps_per_frame": 8,
        "description": "Fingerprint ridges",
    },
}

PRESET_NAMES: list[str] = list(PRESETS.keys())


def next_preset(current: str) -> str:
    idx = PRESET_NAMES.index(current)
    return PRESET_NAMES[(idx + 1) % len(PRESET_NAMES)]
