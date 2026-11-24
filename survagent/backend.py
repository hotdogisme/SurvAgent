"""OpenAI-compatible text, vision and embedding calls."""

import base64
import json
import os
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Image:
    content: bytes
    media_type: str

    def data_url(self) -> str:
        encoded = base64.b64encode(self.content).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        vision_model: str | None = None,
        embedding_model: str | None = None,
        api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        timeout: float = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.vision_model = vision_model
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.embedding_base_url = (
            embedding_base_url.rstrip("/") if embedding_base_url else self.base_url
        )
        self.embedding_api_key = (
            embedding_api_key if embedding_base_url else embedding_api_key or api_key
        )
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAIBackend":
        return cls(
            os.environ.get("SURVAGENT_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get("SURVAGENT_MODEL"),
            vision_model=os.environ.get("SURVAGENT_VISION_MODEL"),
            embedding_model=os.environ.get("SURVAGENT_EMBEDDING_MODEL"),
            api_key=os.environ.get("SURVAGENT_API_KEY"),
            embedding_base_url=os.environ.get("SURVAGENT_EMBEDDING_BASE_URL"),
            embedding_api_key=os.environ.get("SURVAGENT_EMBEDDING_API_KEY"),
        )

    def _post(self, base_url: str, endpoint: str, data: dict, key: str | None) -> dict:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(
            f"{base_url}/{endpoint}",
            data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"{endpoint} returned HTTP {error.code}") from None
        except (URLError, TimeoutError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            raise RuntimeError(f"{endpoint} connection failed: {reason}") from error
        except ValueError:
            raise RuntimeError(f"{endpoint} returned invalid JSON") from None
        if not isinstance(body, dict):
            raise RuntimeError(f"{endpoint} returned an unexpected response")
        return body

    def chat(self, prompt: str, *, image: Image | None = None) -> str:
        model = self.model if image is None else self.vision_model or self.model
        if not model:
            variable = "SURVAGENT_MODEL" if image is None else "SURVAGENT_VISION_MODEL"
            raise ValueError(f"Set {variable} to a model served by the chat endpoint")
        content: str | list[dict] = prompt
        if image is not None:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image.data_url()}},
            ]
        response = self._post(
            self.base_url,
            "chat/completions",
            {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0},
            self.api_key,
        )
        try:
            value = response["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError):
            raise RuntimeError("Chat response is missing message content") from None
        if isinstance(value, str):
            value = re.sub(r"^\s*<think>.*?</think>\s*", "", value, flags=re.S).strip()
            if value:
                return value
        raise RuntimeError("Chat response contains no answer text")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_model:
            raise ValueError("Set SURVAGENT_EMBEDDING_MODEL to a model served by the embedding endpoint")
        if not texts:
            return []
        response = self._post(
            self.embedding_base_url,
            "embeddings",
            {"model": self.embedding_model, "input": texts},
            self.embedding_api_key,
        )
        try:
            rows = sorted(response["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(texts))):
                raise ValueError
            return [row["embedding"] for row in rows]
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Embedding response does not match the requested texts") from None
