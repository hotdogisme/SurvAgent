"""OpenSlide tiling and optional CLAM/CHIEF ROI selection."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .case_bank import finite_number, required_text


@dataclass(frozen=True)
class Bounds:
    """A tile in level-0 slide coordinates."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def parse(cls, values) -> "Bounds":
        if not isinstance(values, list) or len(values) != 4 or any(type(v) is not int for v in values):
            raise ValueError("bounds must be four level-0 integers [x, y, width, height]")
        if min(values[:2]) < 0 or min(values[2:]) <= 0:
            raise ValueError("Tile coordinates must be nonnegative and tile size positive")
        return cls(*values)

    def as_list(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@contextmanager
def open_slide(path: Path, objective_power=None):
    try:
        import openslide
    except (ImportError, OSError) as error:
        raise RuntimeError("WSI input requires the wsi extra and the OpenSlide native library") from error

    with openslide.OpenSlide(str(path)) as reader:
        if objective_power is None:
            raw = reader.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            if raw is None:
                raise ValueError("Set objective_power for a WSI without objective-power metadata")
            objective_power = float(raw)
        yield Slide(reader, objective_power)


class Slide:
    def __init__(self, reader, objective_power: float):
        self.reader = reader
        self.objective_power = finite_number(objective_power, "objective_power")
        if self.objective_power <= 0:
            raise ValueError("objective_power must be positive")

    @property
    def bounds(self) -> Bounds:
        return Bounds(0, 0, *self.reader.dimensions)

    def downsample(self, magnification: float) -> float:
        value = self.objective_power / magnification
        if value < 1:
            raise ValueError(f"The slide does not have native resolution for {magnification}× tiles")
        return value

    def overview(self):
        width, height = self.reader.dimensions
        scale = max(self.downsample(2.5), max(width, height) / 2048)
        return self.reader.get_thumbnail(
            (max(1, round(width / scale)), max(1, round(height / scale)))
        ).convert("RGB")

    def tile(self, bounds: Bounds, magnification: int):
        from PIL import Image

        if bounds.right > self.bounds.right or bounds.bottom > self.bounds.bottom:
            raise ValueError("Tile bounds must lie within the slide")
        downsample = self.downsample(magnification)
        level = self.reader.get_best_level_for_downsample(downsample)
        level_scale = self.reader.level_downsamples[level]

        def scaled_size(scale):
            return max(1, round(bounds.width / scale)), max(1, round(bounds.height / scale))

        region = self.reader.read_region((bounds.x, bounds.y), level, scaled_size(level_scale))
        if region.mode == "RGBA":
            rgb = Image.new("RGB", region.size, "white")
            rgb.paste(region, mask=region.getchannel("A"))
        else:
            rgb = region.convert("RGB")
        size = scaled_size(downsample)
        return rgb.resize(size, Image.Resampling.BILINEAR) if rgb.size != size else rgb


def _grid(region: Bounds, span: int):
    """Keep partial edge tiles that cover at least 30% of a full tile."""
    for y in range(region.y, region.bottom, span):
        for x in range(region.x, region.right, span):
            width = min(span, region.right - x)
            height = min(span, region.bottom - y)
            if width * height / span**2 >= 0.3:
                yield Bounds(x, y, width, height)


def _tissue_fraction(image) -> float:
    histogram = image.convert("L").histogram()
    return sum(histogram[:225]) / (image.width * image.height)


def _slide_tiles(
    slide: Slide, regions: list[Bounds], magnification: int, pixels: int,
    *, maximum: int | None, thumbnail=None,
) -> list[Bounds]:
    span = max(1, round(pixels * slide.downsample(magnification)))
    scale_x = thumbnail.width / slide.bounds.width if thumbnail is not None else None
    scale_y = thumbnail.height / slide.bounds.height if thumbnail is not None else None
    seen = set()
    eligible = []
    for tile in (tile for region in regions for tile in _grid(region, span)):
        if tile in seen:
            continue
        seen.add(tile)
        if thumbnail is not None:
            box = (
                math.floor(tile.x * scale_x),
                math.floor(tile.y * scale_y),
                math.ceil(tile.right * scale_x),
                math.ceil(tile.bottom * scale_y),
            )
            if _tissue_fraction(thumbnail.crop(box)) < 0.5:
                continue
        tissue = _tissue_fraction(slide.tile(tile, magnification))
        if tissue * (tile.width * tile.height / span**2) >= 0.5:
            eligible.append(tile)
        if maximum is not None and len(eligible) == maximum:
            break
    return eligible


def _clam_coordinates(slide: Slide, path: Path):
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError("CLAM coordinates require the roi extra") from error

    with h5py.File(path, "r") as source:
        dataset = source["coords"]
        coords = np.asarray(dataset[:], dtype=float)
        level = int(dataset.attrs.get("patch_level", 1))
        patch_size = int(dataset.attrs.get("patch_size", 256))
    if coords.ndim != 2 or coords.shape[1] != 2 or not (coords >= 0).all() or not np.isfinite(coords).all():
        raise ValueError("CLAM coords must be a finite N×2 array of level-0 coordinates")
    if not 0 <= level < len(slide.reader.level_downsamples) or patch_size < 1:
        raise ValueError("CLAM patch_level and patch_size must describe this slide's pyramid")
    if (coords[:, 0] >= slide.bounds.width).any() or (coords[:, 1] >= slide.bounds.height).any():
        raise ValueError("CLAM coordinates extend beyond the supplied WSI")
    return coords, level, patch_size


def _attention_regions(slide: Slide, config: dict, base_dir: Path) -> list[Bounds]:
    """Read aligned CLAM coordinates and CHIEF attention, then cluster ROIs."""
    try:
        import numpy as np
        from sklearn.cluster import DBSCAN
    except ImportError as error:
        raise RuntimeError("CLAM/CHIEF ROI inputs require the roi extra") from error

    coords_path = base_dir / required_text(config.get("clam_coords"), "clam_coords")
    attention_path = base_dir / required_text(config.get("chief_attention"), "chief_attention")
    coords, level, patch_size = _clam_coordinates(slide, coords_path)
    scores = np.load(attention_path, allow_pickle=False).reshape(-1)
    if len(scores) != len(coords) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("CHIEF attention must contain one finite score per CLAM coordinate")

    threshold = finite_number(config.get("attention_threshold", 0.6), "attention_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("attention_threshold must be between 0 and 1")
    minimum = config.get("min_samples", 10)
    if type(minimum) is not int or minimum < 1:
        raise ValueError("min_samples must be a positive integer")
    patch_span = patch_size * slide.reader.level_downsamples[level]
    spread = np.ptp(scores)
    if spread == 0:
        return []
    filtered = coords[(scores - scores.min()) / spread >= threshold]
    if len(filtered) < minimum:
        return []
    labels = DBSCAN(eps=patch_span * 4, min_samples=minimum).fit_predict(filtered)
    slide_width = slide.bounds.width
    slide_height = slide.bounds.height
    regions = []
    for label in sorted(set(labels) - {-1}):
        low = filtered[labels == label].min(axis=0)
        high = filtered[labels == label].max(axis=0)
        center_x, center_y = (low + high) / 2
        side = max(float((high - low).max()), patch_span) + 2 * patch_span
        left = max(0, round(center_x - side / 2))
        top = max(0, round(center_y - side / 2))
        right = min(slide_width, round(center_x + side / 2))
        bottom = min(slide_height, round(center_y + side / 2))
        if right > left and bottom > top:
            regions.append(Bounds(left, top, right - left, bottom - top))
    return regions


def candidate_regions(slide: Slide, config: dict, base_dir: Path, *, overview=None) -> list[Bounds]:
    budget = config.get("max_regions", 128)
    if type(budget) is not int or budget < 1:
        raise ValueError("max_regions must be a positive integer")
    has_coords = config.get("clam_coords") is not None
    if has_coords != (config.get("chief_attention") is not None):
        raise ValueError("Provide both clam_coords and chief_attention for attention-guided tiling")
    regions = _attention_regions(slide, config, base_dir) if has_coords else []
    if regions:
        return _slide_tiles(slide, regions, 10, 512, maximum=budget)

    # Score whole-slide candidate windows against the overview before reading patches.
    thumbnail = overview if overview is not None else slide.overview()
    return _slide_tiles(slide, [slide.bounds], 10, 512, maximum=budget, thumbnail=thumbnail)


def detail_regions(slide: Slide, parent: Bounds) -> list[Bounds]:
    return _slide_tiles(slide, [parent], 20, 512, maximum=None)


def tile_wsi(
    config: dict, *, base_dir: Path, tiles_dir: Path | None = None, screening: bool = False
) -> dict:
    """Export a 10× report manifest or 20× screening patches for the expert encoder."""
    slide_path = (base_dir / required_text(config.get("slide"), "slide")).resolve()
    with open_slide(slide_path, config.get("objective_power")) as slide:
        overview = slide.overview()
        clam_level = None
        if screening:
            budget = config.get("max_screening_patches")
            if budget is not None and (type(budget) is not int or budget < 1):
                raise ValueError("max_screening_patches must be a positive integer")
            magnification, pixels = 20, 256
            if config.get("clam_coords") is None:
                bounds = _slide_tiles(
                    slide, [slide.bounds], magnification, pixels, maximum=budget, thumbnail=overview
                )
            else:
                path = base_dir / required_text(config["clam_coords"], "clam_coords")
                coords, clam_level, pixels = _clam_coordinates(slide, path)
                downsample = slide.reader.level_downsamples[clam_level]
                magnification = slide.objective_power / downsample
                span = max(1, round(pixels * downsample))
                bounds = []
                for x, y in coords[:budget]:
                    x, y = int(x), int(y)
                    bounds.append(Bounds(x, y, min(span, slide.bounds.width - x), min(span, slide.bounds.height - y)))
            section = "screening_patches"
        else:
            magnification, pixels = 10, 512
            bounds = candidate_regions(slide, config, base_dir, overview=overview)
            section = "regions"
        entries = [{"bounds": tile.as_list()} for tile in bounds]
        result = {"slide": str(slide_path), "objective_power": slide.objective_power, section: entries}
        if "checklist" in config:
            result["checklist"] = config["checklist"]
        if not screening and config.get("chief_encoder") is not None:
            encoder = dict(config["chief_encoder"])
            for field in ("module", "checkpoint"):
                encoder[field] = str((base_dir / required_text(encoder.get(field), field)).resolve())
            result["chief_encoder"] = encoder
        if screening:
            result.update({"magnification": magnification, "patch_size": pixels})
        if tiles_dir is not None:
            from PIL import Image

            stage = "screening_20x" if screening else "regions_10x"
            tiles_dir = (tiles_dir / stage).resolve()
            tiles_dir.mkdir(parents=True, exist_ok=True)
            overview_path = tiles_dir / "overview.png"
            overview.save(overview_path)
            result["overview"] = str(overview_path)
            for tile, entry in zip(bounds, entries):
                if clam_level is not None:
                    # Original split_tile.py reads fixed-size patches in HDF5 order.
                    patch = slide.reader.read_region(
                        (tile.x, tile.y), clam_level, (pixels, pixels)
                    ).convert("RGB")
                else:
                    patch = slide.tile(tile, magnification)
                if patch.size != (pixels, pixels):
                    padded = Image.new("RGB", (pixels, pixels), "white")
                    padded.paste(patch, (0, 0))
                    patch = padded
                patch_path = tiles_dir / f"tile_x{tile.x}_y{tile.y}.png"
                patch.save(patch_path)
                entry["image"] = str(patch_path)
        return result
