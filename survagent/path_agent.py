"""Single-slide entry point for the archived WSI description and merge workflow."""

import io
import mimetypes
from contextlib import ExitStack
from pathlib import Path

from .backend import Image
from .case_bank import finite_number, required_text, unit_vector
from .pathology import extract_bladder_cancer_features
from .prompt_templates import prompt_text
from .research_wsi import deduplicate_by_cosine, merge_wsi_descriptions
from .visual_encoder import ChiefEncoder
from .wsi import Bounds, candidate_regions, detail_regions, open_slide


def report_wsi(
    manifest: dict,
    backend,
    *,
    base_dir: Path = Path("."),
    visual_threshold: float = 0.9,
    text_threshold: float = 0.9,
) -> str:
    report = analyze_wsi(
        manifest, backend, base_dir=base_dir,
        visual_threshold=visual_threshold, text_threshold=text_threshold,
    )
    return report["wsi_report"]


def analyze_wsi(
    manifest: dict,
    backend,
    *,
    base_dir: Path = Path("."),
    visual_threshold: float = 0.9,
    text_threshold: float = 0.9,
) -> dict:
    visual_threshold = _threshold(visual_threshold)
    text_threshold = _threshold(text_threshold)
    source = manifest.get("chief_encoder")
    if source is not None and not isinstance(source, dict):
        raise ValueError("chief_encoder must be an object")
    encoder = ChiefEncoder(source, base_dir) if source is not None else None

    with ExitStack() as stack:
        slide = None
        if manifest.get("slide") is not None:
            slide_path = base_dir / required_text(manifest["slide"], "slide")
            slide = stack.enter_context(open_slide(slide_path, manifest.get("objective_power")))
        if manifest.get("overview") is not None:
            overview = _image_file(base_dir, manifest["overview"])
        elif slide is not None:
            overview = _encode(slide.overview())
        else:
            raise ValueError("Provide a 2.5× overview image or a whole-slide file")

        regions = manifest.get("regions")
        if regions is None:
            regions = (
                [{"bounds": region.as_list()} for region in candidate_regions(slide, manifest, base_dir)]
                if slide is not None else []
            )
        describe_tile = prompt_text("wsi", "describe_tile.txt")
        describe_and_zoom = prompt_text("wsi", "describe_and_zoom.txt")
        global_report = backend.chat(describe_tile, image=overview)

        def describe_tiles(tiles: list[dict], magnification: int) -> list[tuple[dict, dict]]:
            if not isinstance(tiles, list) or any(not isinstance(tile, dict) for tile in tiles):
                raise ValueError("regions and detail must be arrays of tile objects")
            if not tiles:
                return []
            images = [_tile_image(tile, magnification, base_dir, slide) for tile in tiles]
            raw_vectors = [tile.get("embedding") for tile in tiles]
            missing = [index for index, vector in enumerate(raw_vectors) if vector is None]
            if missing:
                if encoder is None:
                    raise ValueError("Provide chief_encoder or a precomputed image embedding for each tile")
                vectors = encoder.embed([images[index] for index in missing])
                if len(vectors) != len(missing):
                    raise ValueError("The visual encoder must return one vector per patch")
                for index, vector in zip(missing, vectors):
                    raw_vectors[index] = vector
            visual = {index: unit_vector(vector) for index, vector in enumerate(raw_vectors)}
            prompt = describe_and_zoom if magnification == 10 else describe_tile
            descriptions = [backend.chat(prompt, image=image) for image in images]
            textual = _report_vectors(descriptions, backend)
            visual_keep = set(deduplicate_by_cosine(visual, visual_threshold))
            text_keep = set(deduplicate_by_cosine(textual, text_threshold))
            return [
                (tiles[index], {
                    "description": descriptions[index],
                    "enlarge": "higher magnification" in descriptions[index],
                })
                for index in sorted(visual_keep & text_keep)
            ]

        observations = [global_report]
        regions_analyzed = []
        for region, result in describe_tiles(regions, 10):
            observations.append(result["description"])
            region_result = {**_location(region), **result, "detail": []}
            if result["enlarge"]:
                tiles = region.get("detail")
                if tiles is None and slide is not None and "bounds" in region:
                    parent = Bounds.parse(region["bounds"])
                    tiles = [{"bounds": tile.as_list()} for tile in detail_regions(slide, parent)]
                for tile, high in describe_tiles(tiles if tiles is not None else [], 20):
                    observations.append(high["description"])
                    region_result["detail"].append({**_location(tile), **high})
            regions_analyzed.append(region_result)

    retained = deduplicate_by_cosine(_report_vectors(observations, backend), text_threshold)
    checklist = manifest.get("checklist", "general")
    if checklist not in ("general", "bladder"):
        raise ValueError("checklist must be general or bladder")
    report, extractions = merge_wsi_descriptions(
        [observations[index] for index in retained],
        backend,
        prompt_text("wsi", "extract_wsi_feature.txt"),
        prompt_text("wsi", "wsi_combine_full.txt"),
    )
    result = {
        "wsi_report": report,
        "overview_report": global_report,
        "candidate_count": len(regions),
        "analyzed_regions": regions_analyzed,
        "group_extractions": extractions,
    }
    if checklist == "bladder":
        result["attributes"] = extract_bladder_cancer_features(report, backend=backend)
    return result


def _report_vectors(descriptions: list[str], backend) -> dict[int, tuple[float, ...]]:
    vectors = backend.embed(descriptions)
    if len(vectors) != len(descriptions):
        raise ValueError("The text embedding backend must return one vector per description")
    return {index: unit_vector(vector) for index, vector in enumerate(vectors)}


def _location(tile: dict) -> dict:
    return {key: tile[key] for key in ("image", "bounds") if key in tile}


def _threshold(value: float) -> float:
    value = finite_number(value, "similarity threshold")
    if not -1.0 <= value <= 1.0:
        raise ValueError("Similarity thresholds must be between -1 and 1")
    return value


def _image_file(base_dir: Path, filename: str) -> Image:
    path = base_dir / required_text(filename, "image")
    mime, _ = mimetypes.guess_type(path)
    if mime not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError("Use a JPEG, PNG or WebP image")
    return Image(path.read_bytes(), mime)


def _encode(image) -> Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Image(buffer.getvalue(), "image/png")


def _tile_image(tile: dict, magnification: int, base_dir: Path, slide) -> Image:
    if tile.get("image") is not None:
        return _image_file(base_dir, tile["image"])
    if slide is None or "bounds" not in tile:
        raise ValueError("Each tile needs an image path or slide bounds")
    return _encode(slide.tile(Bounds.parse(tile["bounds"]), magnification))
