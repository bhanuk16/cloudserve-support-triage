"""Loads versioned prompt files from prompts/. A prompt is a design artefact with an
id, a version and the requirements it serves; that id+version is written to the
decision log for every generation, so any reply can be traced to its exact prompt."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .config import ROOT

PROMPT_DIR = ROOT / "prompts"


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    system: str
    user_template: str
    serves: str

    @property
    def tag(self) -> str:
        return f"{self.id} v{self.version}"


@lru_cache(maxsize=None)
def load_prompt(filename: str) -> Prompt:
    text = (PROMPT_DIR / filename).read_text(encoding="utf-8")
    meta_block, body = text.split("---", 2)[1:]
    meta = dict(re.findall(r"^(\w+):\s*(.+)$", meta_block, re.M))
    system = body.split("## system", 1)[1].split("## user", 1)[0].strip()
    user = body.split("## user", 1)[1].strip()
    return Prompt(meta["id"], meta["version"], system, user, meta.get("serves", ""))
