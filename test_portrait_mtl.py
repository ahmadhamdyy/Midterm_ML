from pathlib import Path

import torch

from portrait_mtl import (
    PortraitNet,
    YEAR_MAX,
    YEAR_MIN,
    parse_record,
    split_train_records,
)


def test_parse_record(tmp_path: Path) -> None:
    path = tmp_path / "F" / "1967_TX_example.jpg"
    path.parent.mkdir()
    path.touch()
    record = parse_record(tmp_path, "F/1967_TX_example.jpg")
    assert record.year == 1967
    assert record.state == "TX"
    assert record.gender == 0


def test_year_range_constants() -> None:
    assert YEAR_MIN == 1905
    assert YEAR_MAX == 2013


def test_seeded_split_is_stable(tmp_path: Path) -> None:
    records = [parse_record(tmp_path, f"M/19{20 + i:02d}_NY_x.jpg") for i in range(10)]
    first = split_train_records(records)
    second = split_train_records(records)
    assert first == second
    assert len(first[1]) == 1


def test_model_output_shapes() -> None:
    model = PortraitNet("multi", pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 64, 64))
    assert output["year"].shape == (2,)
    assert output["gender"].shape == (2, 2)
