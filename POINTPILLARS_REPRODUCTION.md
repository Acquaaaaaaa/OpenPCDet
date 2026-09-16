# PointPillars KITTI Baseline Reproduction

This branch records a reproducible PointPillars baseline built on OpenPCDet `233f849`.

## Environment

- Ubuntu 22.04 on WSL2
- NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)
- Python 3.10.21
- PyTorch 2.9.1+cu128
- OpenPCDet 0.6.0

The KITTI dataset is kept outside the repository and linked to `data/kitti`. Dataset files, generated metadata, checkpoints, predictions, logs, and TensorBoard events are intentionally excluded from Git.

## Training

Run from `tools/`:

```bash
python train.py \
  --cfg_file cfgs/kitti_models/pointpillar.yaml \
  --batch_size 2 \
  --epochs 80 \
  --workers 4 \
  --extra_tag full_80ep_bs2_seed666_run001 \
  --fix_random_seed \
  --ckpt_save_interval 1 \
  --max_ckpt_save_num 30 \
  --num_epochs_to_eval 1 \
  --logger_iter_interval 50
```

Training completed in about 6 h 47 min. The final epoch average loss was approximately `0.326`.

## KITTI validation results

3D AP R40 Moderate:

| Checkpoint | Car | Pedestrian | Cyclist | Mean |
| --- | ---: | ---: | ---: | ---: |
| Epoch 79 | 78.1260 | 46.9732 | 62.6052 | 62.5681 |
| Epoch 80 | 77.9652 | 46.9819 | 62.5610 | 62.5027 |
| OpenPCDet pretrained reference | 78.4001 | 51.4538 | 62.9422 | 64.2654 |

Epoch 80 inference time was approximately `0.0440 s/sample` on 3,769 KITTI validation samples. Recall at IoU thresholds 0.3/0.5/0.7 was `0.937692 / 0.877150 / 0.635551`.

## Compatibility changes

- Load trusted OpenPCDet checkpoints with PyTorch's restricted `weights_only` mechanism and explicit safe NumPy types.
- Avoid TorchScript compilation of the Argoverse quaternion helper, which is incompatible with the current PyTorch environment during package import.
