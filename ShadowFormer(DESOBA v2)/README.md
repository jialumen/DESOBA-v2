# ShadowFormer (DESOBA v2)

This folder contains the DESOBA v2 submission bundle for ShadowFormer.

Files:

- `best_path.pth`: anonymized best checkpoint.
- `test_desoba_v2_latest_metrics.py`: DESOBA v2 evaluation script with local machine paths replaced by relative dataset placeholders.
- `ShadowFormer_best.json`: package metadata and checksums.

Notes:

- No local workstation path, remote host, SSH account, port, or password is intentionally included.
- Configure `--relation_annotation_root` and `--relation_image_root` to the DESOBA v2 dataset locations before running evaluation.
