from __future__ import annotations

from ink_tokenizer.edge_ocr import Box, box_iou, greedy_matches, ocr_with_rapidocr, parse_tesseract_tsv


def test_box_iou_and_greedy_matching_are_one_to_one():
    candidates = [Box(0, 0, 10, 10), Box(1, 1, 11, 11)]
    references = [Box(0, 0, 10, 10)]

    assert box_iou(candidates[0], references[0]) == 1.0
    assert greedy_matches(candidates, references, threshold=0.3) == [(0, 0, 1.0)]


def test_parse_tesseract_tsv_filters_page_rows_and_low_confidence():
    payload = "\n".join(
        [
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t",
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t12\t92.5\tStart",
            "5\t1\t1\t1\t1\t2\t50\t20\t20\t12\t10\tnoise",
        ]
    )

    assert parse_tesseract_tsv(payload) == [Box(10.0, 20.0, 40.0, 32.0, 92.5, "Start")]


def test_rapidocr_adapter_converts_quadrilaterals_to_boxes():
    class Output:
        boxes = [[[10, 20], [40, 18], [42, 35], [12, 37]]]
        txts = ("Start",)
        scores = (0.95,)

    class Engine:
        def __call__(self, image_path):
            assert image_path == "page.png"
            return Output()

    assert ocr_with_rapidocr("page.png", Engine()) == [
        Box(10.0, 18.0, 42.0, 37.0, 95.0, "Start")
    ]
