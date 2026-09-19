"""Command line entry point: `ink-tokenizer ...` (or `python -m ink_tokenizer`).

Exists so a fresh checkout on a lab machine can produce ink without anyone
writing a script first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from .base import PROMPT_DERENDER, PROMPT_RECOGNIZE_AND_DERENDER
from .visualize import ink_to_svg, load_ink_json


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--endpoint",
        default=None,
        help="Run against an HTTP service instead of in-process TensorFlow.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="HF repo id or local SavedModel directory (in-process only).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--prompt",
        default=PROMPT_RECOGNIZE_AND_DERENDER,
        choices=[PROMPT_RECOGNIZE_AND_DERENDER, PROMPT_DERENDER],
    )
    p.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not retry empty results with the plain derender prompt.",
    )


def _make_backend(args):
    from . import get_backend

    kwargs = {}
    if args.endpoint is None:
        if args.model:
            kwargs["model"] = args.model
        kwargs["batch_size"] = args.batch_size
    return get_backend(args.endpoint, **kwargs)


def _cmd_doctor(args) -> int:
    """Report whether this machine can run inference, and how."""
    import platform

    print(f"python      {platform.python_version()} on {sys.platform}")

    try:
        from .tf_runtime import tf

        print(f"tensorflow  {tf.__version__}")
        import tensorflow_text as tf_text

        print(f"tf-text     {tf_text.__version__}")
        gpus = [d.name for d in tf.config.list_physical_devices("GPU")]
        print(f"gpus        {gpus or 'none (CPU inference)'}")
        local_ok = True
    except ImportError as exc:
        print(f"tensorflow  NOT AVAILABLE ({exc.name or exc})")
        local_ok = False

    if args.endpoint:
        from .client import InkSightClient

        try:
            print(f"service     {InkSightClient(args.endpoint).health()}")
        except Exception as exc:
            print(f"service     unreachable at {args.endpoint}: {exc}")

    if local_ok:
        print("\nready: in-process inference available (LocalBackend)")
        return 0
    print(
        "\nnot ready for in-process inference.\n"
        "  Linux/macOS: uv sync --extra inference\n"
        "  Windows:     run the service elsewhere and pass --endpoint"
    )
    return 1


def _cmd_derender(args) -> int:
    paths = [Path(p) for p in args.images]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"no such file: {missing[0]}", file=sys.stderr)
        return 2

    backend = _make_backend(args)
    crops = []
    from .preprocess import scale_and_pad

    for p in paths:
        crops.append(scale_and_pad(Image.open(p)))

    results = backend.derender_crops(
        crops, prompt=args.prompt, fallback=not args.no_fallback
    )

    payload = [
        {
            "image": str(p),
            "text": r.text,
            "strokes": len(r.ink),
            "points": r.ink.num_points,
            "used_fallback": r.used_fallback,
            "ink": r.ink.to_dict(),
        }
        for p, r in zip(paths, results)
    ]

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    for entry in payload:
        print(
            f"{entry['image']}: {entry['strokes']} strokes, "
            f"{entry['points']} points"
            + (f", text={entry['text']!r}" if entry["text"] else "")
            + (" (fallback)" if entry["used_fallback"] else "")
        )
    backend.close()
    return 0


def _cmd_visualize(args) -> int:
    source = Path(args.ink_json)
    if not source.is_file():
        print(f"no such file: {source}", file=sys.stderr)
        return 2
    try:
        ink, inferred_image = load_ink_json(source, args.index)
        background = args.image if args.image is not None else inferred_image
        if background is not None and not Path(background).is_file():
            print(
                f"source image unavailable ({background}); drawing ink only",
                file=sys.stderr,
            )
            background = None
        svg = ink_to_svg(ink, background=background, show_order=args.show_order)
    except ValueError as exc:
        print(f"cannot visualize {source}: {exc}", file=sys.stderr)
        return 2

    output = Path(args.out) if args.out else source.with_suffix(".svg")
    output.write_text(svg, encoding="utf-8")
    print(f"wrote {output} ({len(ink)} strokes, {ink.num_points} points)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ink-tokenizer")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor", help="check whether this machine can run inference"
    )
    doctor.add_argument("--endpoint", default=None)
    doctor.set_defaults(func=_cmd_doctor)

    der = sub.add_parser("derender", help="convert word-level images to ink")
    der.add_argument("images", nargs="+")
    der.add_argument("-o", "--out", help="write results as JSON to this path")
    _add_backend_args(der)
    der.set_defaults(func=_cmd_derender)

    vis = sub.add_parser("visualize", help="render recovered ink JSON as SVG")
    vis.add_argument("ink_json", help="raw Ink JSON or output from `derender -o`")
    vis.add_argument("-o", "--out", help="SVG output path (default: INPUT.svg)")
    vis.add_argument(
        "--image",
        help="source image to embed behind the ink (defaults to the result's image)",
    )
    vis.add_argument(
        "--index", type=int, default=0, help="result index when the JSON has many inputs"
    )
    vis.add_argument(
        "--show-order", action="store_true", help="label each stroke's start point"
    )
    vis.set_defaults(func=_cmd_visualize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
