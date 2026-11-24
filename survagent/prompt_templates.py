"""Load the packaged prompt files by stage."""

import re
from importlib.resources import files


def prompt_text(stage: str, name: str) -> str:
    return files("survagent").joinpath("prompts", stage, name).read_text(encoding="utf-8")


def fill_prompt(template: str, **values: str) -> str:
    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", lambda match: values.get(match[1], match[0]), template)
