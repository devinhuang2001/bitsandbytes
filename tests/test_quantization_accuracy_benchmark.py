import json

from benchmarking.quantization_accuracy import affine_int4, compare_weight, distribution_stats, main, tensor_bytes
import pytest
import torch


def test_affine_int4_packing_and_stored_metadata():
    weight = torch.arange(-7, 9, dtype=torch.float16).repeat(2, 2)
    restored, packed, scale, zero = affine_int4(weight, 32)

    torch.testing.assert_close(restored, weight, atol=0, rtol=0)
    assert packed.dtype == torch.uint8
    assert packed[0].tolist() == [16, 50, 84, 118, 152, 186, 220, 254] * 2
    assert scale.tolist() == [[1.0], [1.0]]
    assert zero.tolist() == [[7.0], [7.0]]
    assert scale.dtype == zero.dtype == torch.float16
    assert tensor_bytes(packed, scale, zero) == 40


@pytest.mark.parametrize("shape,blocksize", [((2, 33), 32), ((32,), 32), ((2, 32), 0)])
def test_affine_int4_rejects_invalid_groups(shape, blocksize):
    with pytest.raises(ValueError):
        affine_int4(torch.ones(shape, dtype=torch.float16), blocksize)


def test_distribution_stats_use_population_moments_per_block():
    weight = torch.tensor([[-1.0, 1.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    stats = distribution_stats(weight, 4)
    assert stats == {"mean_block_kurtosis": 0.5, "mean_block_skew": 0.0, "mean_block_max_abs_over_std": 0.5}


def test_compare_weight_matches_group_sizes_and_counts_real_bytes():
    weight = torch.arange(-7, 9, dtype=torch.float16).repeat(2, 2)
    result = compare_weight(weight, 32)

    assert result["blocksize"] == 32
    assert result["affine_int4"]["rmse"] == 0
    assert result["affine_int4"]["packed_bytes"] == 32
    assert result["affine_int4"]["metadata_bytes"] == 8
    assert result["affine_int4"]["total_bytes"] == 40
    assert result["nf4"]["packed_bytes"] == 32
    assert result["nf4"]["metadata_bytes"] == 72
    assert result["nf4"]["total_bytes"] == 104
    assert result["nf4"]["rmse"] > 0


def test_checkpoint_cli_records_selected_layers_and_skipped_groups(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.pt"
    output = tmp_path / "accuracy.json"
    torch.save(
        {
            "q_proj.weight": torch.arange(-7, 9, dtype=torch.float16).repeat(2, 2),
            "embedding.weight": torch.ones(2, 32),
        },
        checkpoint,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["quantization_accuracy", "--checkpoint", str(checkpoint), "--blocksize", "32", "64", "--output", str(output)],
    )
    main()
    report = json.loads(output.read_text())
    assert len(report["results"]) == 2
    assert all(result["name"] == "q_proj.weight" for result in report["results"])
    assert report["results"][0]["affine_int4"]["rmse"] == 0
    assert report["results"][1]["skipped"] == "row length incompatible with group"
