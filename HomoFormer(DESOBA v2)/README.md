# HomoFormer (DESOBA v2)

This folder contains the DESOBA v2 release package for HomoFormer.

## Files

- `best_path.pth`: best HomoFormer checkpoint from the supplied release package. The checkpoint was anonymized and slimmed to remove source-path metadata.
- `test_desoba_v2_latest_metrics.py`: DESOBA v2 evaluation entrypoint for Shadow / non-Shadow / all metrics.

## Checkpoint format

`best_path.pth` contains:

- `state_dict`
- `local_detail_mix_bias`
- `model`

Training-only state and source-path metadata are intentionally omitted.

## Evaluation

Run from a compatible HomoFormer codebase with DESOBA v2 data prepared:

```bash
python test_desoba_v2_latest_metrics.py \
  --weights best_path.pth \
  --relation_annotation_root ./data/DESOBAv2_extended_annotations \
  --relation_image_root ./data/DESOBAv2_work/release256 \
  --split test
```

The release files intentionally avoid machine names, usernames, ports, absolute local paths, and absolute remote paths.
