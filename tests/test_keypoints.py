from ink_tokenizer.edge_ocr import Box
from ink_tokenizer.keypoints import FlowNode, infer_downward_links, order_flow_nodes


def test_flow_layout_prefers_the_main_lane_within_a_vertical_band():
    start = FlowNode(Box(90, 0, 110, 20), "Start")
    main = FlowNode(Box(90, 50, 110, 70), "Read")
    branch = FlowNode(Box(-10, 50, 10, 70), "No")
    end = FlowNode(Box(90, 110, 110, 130), "End")

    assert order_flow_nodes([start, branch, main, end]) == [start, main, branch, end]


def test_downward_links_choose_the_nearest_successor_in_the_same_lane():
    nodes = [
        FlowNode(Box(90, 0, 110, 20)),
        FlowNode(Box(90, 50, 110, 70)),
        FlowNode(Box(-10, 50, 10, 70)),
        FlowNode(Box(90, 110, 110, 130)),
    ]

    assert infer_downward_links(nodes)[:2] == [(0, 1), (1, 3)]
