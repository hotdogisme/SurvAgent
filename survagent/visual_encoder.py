"""Patch embeddings for CoSMining."""

import importlib.util
import io
from pathlib import Path

from .backend import Image
from .case_bank import required_text



class ChiefEncoder:
    """Use a local CHIEF CTransPath implementation and its separately supplied weights."""

    def __init__(self, config: dict, base_dir: Path):
        try:
            import torch
            from torchvision import transforms
        except ImportError as error:
            raise RuntimeError("The CHIEF encoder requires the chief extra and compatible CHIEF dependencies") from error

        module = base_dir / required_text(config.get("module"), "chief_encoder.module")
        checkpoint = base_dir / required_text(config.get("checkpoint"), "chief_encoder.checkpoint")
        spec = importlib.util.spec_from_file_location("_survagent_ctran", module)
        if spec is None or spec.loader is None:
            raise ValueError("chief_encoder.module must point to a local ctran.py")
        implementation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(implementation)

        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        model = implementation.ctranspath()
        model.head = torch.nn.Identity()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        self.model = model.eval().to(device)
        self.device = device
        self.torch = torch
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def embed(self, images: list[Image]) -> list[list[float]]:
        from PIL import Image as PILImage

        embeddings = []
        for start in range(0, len(images), 16):
            batch = []
            for image in images[start : start + 16]:
                with PILImage.open(io.BytesIO(image.content)) as decoded:
                    batch.append(self.transform(decoded.convert("RGB")))
            inputs = self.torch.stack(batch).to(self.device)
            with self.torch.inference_mode():
                embeddings.extend(self.model(inputs).cpu().tolist())
        return embeddings
