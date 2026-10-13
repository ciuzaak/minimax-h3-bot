"""Ordered image collections and atomic restoration of bot inputs."""

import json
from pathlib import Path
import secrets
import tempfile

from .jobs import MAX_REFERENCE_IMAGES


def image_paths(state):
    if "image_paths" in state:
        return tuple(state["image_paths"])
    return (state["image_path"],)


def image_entries(state):
    if "images" in state:
        return tuple(state["images"])
    return ({"image_path": state["image_path"], "width": state["width"], "height": state["height"]},)


def collection_state(chat_id, user_id, images):
    images = tuple(dict(image) for image in images)
    if not 1 <= len(images) <= MAX_REFERENCE_IMAGES:
        raise ValueError("A collection needs 1–9 images")
    return {"chat_id": chat_id, "user_id": user_id, "images": images,
            "image_paths": tuple(image["image_path"] for image in images),
            "image_path": images[0]["image_path"], "width": images[0]["width"],
            "height": images[0]["height"], "revision": secrets.token_hex(4)}


class InputStore:
    def __init__(self, path, input_dir):
        self.path = Path(path)
        self.input_dir = Path(input_dir).resolve()

    def save(self, states):
        payload = {"schema_version": 1, "collections": list(states.values())}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, ensure_ascii=False)
                stream.write("\n")
            temporary.replace(self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, allowed_users):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("collections"), list):
            raise ValueError("Invalid input manifest")
        states = {}
        for row in data["collections"]:
            try:
                user_id, chat_id = row["user_id"], row["chat_id"]
                if type(user_id) is not int or user_id not in allowed_users or type(chat_id) is not int:
                    continue
                images = image_entries(row)
                if not 1 <= len(images) <= MAX_REFERENCE_IMAGES:
                    continue
                checked = []
                for image in images:
                    path = Path(image["image_path"]).resolve()
                    if path.parent != self.input_dir or not path.name.startswith(f"{user_id}_") or not path.is_file():
                        raise ValueError("Invalid stored image path")
                    if any(type(image.get(k)) is not int or image[k] < 32 or image[k] % 32 for k in ("width", "height")):
                        raise ValueError("Invalid stored canvas")
                    checked.append(dict(image, image_path=str(path)))
                states[(chat_id, user_id)] = collection_state(chat_id, user_id, checked)
            except (KeyError, TypeError, ValueError, OSError):
                continue
        return states
