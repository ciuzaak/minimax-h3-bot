"""Persist per-user generation preferences, preserving legacy shared defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tempfile


LANGUAGES = ("Chinese", "Japanese", "Korean", "English", "none")
DURATIONS = (5, 10, 15)


@dataclass(frozen=True)
class GenerationSettings:
    mode: str = "fast"
    seconds: int = 5
    dialogue_language: str = "Chinese"
    input_mode: str = "i2v"
    reference_mode: str = "standard"
    image_profile: str = "native"
    image_ratio: str = "1:1"
    image_quant: str = "w4a8"
    image_edit_size: str = "input"

    def __post_init__(self):
        if self.mode not in ("fast", "standard", "trial8") or (self.mode == "trial8" and self.input_mode != "reference"):
            raise ValueError("Unsupported generation mode")
        if type(self.seconds) is not int or self.seconds not in DURATIONS:
            raise ValueError("Unsupported duration")
        if self.dialogue_language not in LANGUAGES:
            raise ValueError("Unsupported dialogue language")
        if self.input_mode not in ("i2v", "reference", "image"):
            raise ValueError("Unsupported input mode")
        if self.reference_mode not in ("fast", "standard", "trial8"):
            raise ValueError("Unsupported reference mode")
        if self.image_profile not in ("preview", "balanced", "native"):
            raise ValueError("Unsupported image profile")
        if self.image_ratio not in ("1:1", "4:3", "3:4", "16:9", "9:16"):
            raise ValueError("Unsupported image ratio")
        if self.image_quant not in ("int8", "w4a8"):
            raise ValueError("Unsupported image quantization")
        if self.image_edit_size not in ("input", "canvas"):
            raise ValueError("Unsupported image edit sizing")


def normalize_settings(data, default: GenerationSettings) -> GenerationSettings:
    if not isinstance(data, dict):
        return default
    mode = data.get("mode", data.get("generation_mode"))
    seconds = data.get("seconds")
    language = data.get("dialogue_language")
    return GenerationSettings(
        mode=mode if mode in ("fast", "standard") else default.mode,
        seconds=seconds if type(seconds) is int and seconds in DURATIONS else default.seconds,
        dialogue_language=language if language in LANGUAGES else default.dialogue_language,
        input_mode=data.get("input_mode") if data.get("input_mode") in ("i2v", "reference", "image") else default.input_mode,
        reference_mode=data.get("reference_mode") if data.get("reference_mode") in ("fast", "standard", "trial8") else default.reference_mode,
        image_profile=data.get("image_profile") if data.get("image_profile") in ("preview", "balanced", "native") else default.image_profile,
        image_ratio=data.get("image_ratio") if data.get("image_ratio") in ("1:1", "4:3", "3:4", "16:9", "9:16") else default.image_ratio,
        image_quant=data.get("image_quant") if data.get("image_quant") in ("int8", "w4a8") else default.image_quant,
        image_edit_size=data.get("image_edit_size") if data.get("image_edit_size") in ("input", "canvas") else default.image_edit_size,
    )


class PreferenceStore:
    def __init__(self, path: Path, defaults: GenerationSettings):
        self.path = path
        self.defaults = defaults
        self.users = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        if data.get("schema_version") == 2:
            self.defaults = normalize_settings(data.get("defaults"), defaults)
            users = data.get("users")
            if isinstance(users, dict):
                for key, value in users.items():
                    if isinstance(key, str) and key.isdecimal() and int(key) > 0:
                        self.users[str(int(key))] = normalize_settings(value, self.defaults)
        elif "schema_version" not in data:
            self.defaults = normalize_settings(data, defaults)

    def get(self, user_id: int) -> GenerationSettings:
        return self.users.get(str(user_id), self.defaults)

    def update(self, user_id: int, **changes) -> GenerationSettings:
        if user_id <= 0:
            raise ValueError("User ID must be positive")
        updated = replace(self.get(user_id), **changes)
        users = dict(self.users, **{str(user_id): updated})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump({"schema_version": 2, "defaults": asdict(self.defaults),
                           "users": {key: asdict(value) for key, value in users.items()}},
                          stream, ensure_ascii=False)
                stream.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.users = users
        return updated
