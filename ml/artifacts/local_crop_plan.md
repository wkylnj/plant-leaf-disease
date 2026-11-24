# 局部裁剪训练（local_crop）

- 改动 `ml/src/train_colab.py`（唯一改动的代码文件，commit `d585f1b`）：新增 `--local-crop`（默认 False）/`--local-scale-min 0.15`/`--local-scale-max 0.5`；开启时训练集首步换成 `RandomResizedCrop(224, scale=(0.15, 0.5), ratio=(0.75, 1.33))`，RandAugment / ColorJitter / RandomErasing 原样保留；val/test 仍是 `Resize(256) + CenterCrop(224)`（已断言 on/off 完全一致）；`summary.json` 追加 `local_crop` / `local_scale_min` / `local_scale_max`，默认关闭时行为与历史完全一致。
- 本地冒烟（CPU，resnet18，`--epochs 1 --limit-batches 2`，全量 val/test 评估）：关闭 exit=0（test_acc 0.0458）、开启 exit=0（test_acc 0.0444，summary `local_crop=true`），best.pt / curves.png / confusion_matrix.png / summary.json 均正常落盘。
- Kaggle：https://www.kaggle.com/code/wangkaiyu66/plant-leaf-local-crop ，kernel v1 已提交（MobileNetV3-Large 15 轮，T4，run-name `mobilenet_v3_large_localcrop_224_15ep`），提交后 status = `QUEUED`。
- 预计 75~90 分钟出结果（参照 da_aug 的 73 分钟），产物在 `/kaggle/working/runs/mobilenet_v3_large_localcrop_224_15ep`；取出后先跑 `eval_crossdomain.py` 比对基线 0.2582 / da_aug 0.2360。
