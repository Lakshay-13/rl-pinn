from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


DEFAULT_CYCLES = 5
DEFAULT_COLUMNS = 5
SUPPORTED_PLOTS = ("loss", "potential")
BACKGROUND = "white"
TEXT = "black"
SUBTLE = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render combined cycle plot sheets for each strategy run."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Batch run root containing per-strategy folders.",
    )
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Optional strategy list. Defaults to all strategy folders under run root.",
    )
    parser.add_argument(
        "--run-name",
        default="run_1",
        help="Optional nested run directory name inside each strategy folder.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help="Number of cycles to include.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=DEFAULT_COLUMNS,
        help="Grid columns.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=480,
        help="Per-tile resize width.",
    )
    parser.add_argument(
        "--plots",
        nargs="*",
        default=list(SUPPORTED_PLOTS),
        choices=SUPPORTED_PLOTS,
        help="Plot types to render.",
    )
    return parser.parse_args()


def _discover_strategies(run_root: Path, requested: Sequence[str] | None) -> list[str]:
    if requested:
        return list(requested)
    return sorted(path.name for path in run_root.iterdir() if path.is_dir())


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("Arial.ttf", "Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_image(image: Image.Image, target_width: int) -> Image.Image:
    width, height = image.size
    ratio = target_width / float(width)
    target_height = max(1, int(height * ratio))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _render_grid(
    image_entries: Iterable[tuple[int, Path]],
    output_path: Path,
    *,
    title: str,
    tile_width: int,
    columns: int,
) -> None:
    entries = list(image_entries)
    images = [Image.open(path).convert("RGB") for _, path in entries]
    if not images:
        raise ValueError(f"No images provided for {output_path}")

    resized = [_fit_image(image, tile_width) for image in images]
    tile_height = max(image.size[1] for image in resized)
    rows = (len(resized) + columns - 1) // columns

    margin = 28
    gap_x = 24
    gap_y = 36
    title_height = 54
    label_height = 32

    canvas_width = margin * 2 + columns * tile_width + (columns - 1) * gap_x
    canvas_height = (
        margin * 2
        + title_height
        + rows * (tile_height + label_height)
        + (rows - 1) * gap_y
    )

    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(28)
    label_font = _load_font(18)

    title_width, title_text_height = _text_size(draw, title, title_font)
    draw.text(
        ((canvas_width - title_width) // 2, margin),
        title,
        fill=TEXT,
        font=title_font,
    )

    y_start = margin + max(title_height, title_text_height + 12)
    for index, image in enumerate(resized):
        cycle_number = entries[index][0]
        row = index // columns
        col = index % columns
        x = margin + col * (tile_width + gap_x)
        y = y_start + row * (tile_height + label_height + gap_y)

        centered_x = x + (tile_width - image.size[0]) // 2
        centered_y = y + (tile_height - image.size[1]) // 2
        canvas.paste(image, (centered_x, centered_y))

        label = f"Cycle {cycle_number}"
        label_width, _ = _text_size(draw, label, label_font)
        draw.text(
            (x + (tile_width - label_width) // 2, y + tile_height + 6),
            label,
            fill=SUBTLE,
            font=label_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def _discover_cycle_images(run_dir: Path, plot_name: str, max_cycles: int) -> list[tuple[int, Path]]:
    image_entries: list[tuple[int, Path]] = []
    discovered: list[tuple[int, Path]] = []
    for cycle_dir in run_dir.glob("cycle_*"):
        if not cycle_dir.is_dir():
            continue
        try:
            cycle_number = int(cycle_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if cycle_number > max_cycles:
            continue
        image_path = cycle_dir / f"{plot_name}.png"
        if image_path.exists():
            discovered.append((cycle_number, image_path))
    image_entries.extend(sorted(discovered, key=lambda item: item[0]))
    return image_entries


def _resolve_strategy_run_dir(run_root: Path, strategy: str, run_name: str) -> Path:
    nested_dir = run_root / strategy / run_name
    if nested_dir.exists():
        return nested_dir
    direct_dir = run_root / strategy
    if direct_dir.exists():
        return direct_dir
    raise FileNotFoundError(f"Missing strategy directory: {direct_dir}")


def main() -> None:
    args = parse_args()
    strategies = _discover_strategies(args.run_root, args.strategies)
    max_cycles = int(args.cycles)

    for strategy in strategies:
        run_dir = _resolve_strategy_run_dir(args.run_root, strategy, args.run_name)

        for plot_name in args.plots:
            image_entries = _discover_cycle_images(run_dir, plot_name, max_cycles)
            if not image_entries:
                raise FileNotFoundError(
                    f"Missing {plot_name} images for {strategy} under {run_dir}"
                )

            first_cycle = image_entries[0][0]
            last_cycle = image_entries[-1][0]
            title = f"{strategy} {plot_name.title()} Plots, Cycles {first_cycle}-{last_cycle}"
            output_path = run_dir / f"{plot_name}_cycles_{first_cycle}_{last_cycle}_grid.png"
            _render_grid(
                image_entries,
                output_path,
                title=title,
                tile_width=int(args.tile_width),
                columns=int(args.columns),
            )
            print(output_path)


if __name__ == "__main__":
    main()
