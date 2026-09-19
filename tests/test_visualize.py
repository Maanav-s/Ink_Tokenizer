import json

from PIL import Image

from ink_tokenizer.cli import main
from ink_tokenizer.ink import Ink
from ink_tokenizer.visualize import ink_to_svg, load_ink_json


def test_ink_to_svg_frames_strokes_without_an_image():
    svg = ink_to_svg(Ink.from_point_lists([[(2, 3), (7, 8)], [(4, 5)]]))

    assert 'viewBox="-6 -5 21 21"' in svg
    assert '<path d="M 2 3 L 7 8" />' in svg
    assert '<circle cx="4" cy="5" r="2" />' in svg


def test_result_json_uses_its_source_image(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (20, 10), "white").save(image)
    results = tmp_path / "ink.json"
    results.write_text(
        json.dumps(
            [
                {
                    "image": str(image),
                    "ink": {"strokes": [{"points": [[1, 2], [3, 4]]}]},
                }
            ]
        ),
        encoding="utf-8",
    )

    ink, background = load_ink_json(results)
    assert ink.num_points == 2
    assert background == str(image)
    svg = ink_to_svg(ink, background=background, show_order=True)
    assert 'viewBox="0 0 20 10"' in svg
    assert "data:image/png;base64," in svg
    assert ">1</text>" in svg


def test_visualize_command_writes_default_svg(tmp_path):
    source = tmp_path / "ink.json"
    source.write_text('{"strokes": [{"points": [[0, 0], [1, 1]]}]}')

    assert main(["visualize", str(source)]) == 0
    output = source.with_suffix(".svg")
    assert output.is_file()
    assert "Recovered ink" in output.read_text(encoding="utf-8")
