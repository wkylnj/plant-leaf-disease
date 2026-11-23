# TTA 与模型集成评估

- 视图组合：`none+hflip+vflip+hvflip`（4 个）
- 同域数据：PlantVillage test
- 跨域数据：PlantDoc all

## 同域 PlantVillage

| 配置 | top-1 | top-3 | 相对基线 |
|---|---|---|---|
| mobilenet_v3_large_224_15ep 基线 | 0.9948 | 0.9998 | — |
| mobilenet_v3_large_224_15ep + TTA | 0.9949 | 0.9998 | +0.0001 |
| resnet18_224_15ep 基线 | 0.9941 | 0.9997 | — |
| resnet18_224_15ep + TTA | 0.9939 | 0.9997 | -0.0002 |
| 集成 + TTA | 0.9965 | 0.9999 | +0.0017 |

## 跨域 PlantDoc

| 配置 | top-1 | 作物级 | top-3 | 相对基线 |
|---|---|---|---|---|
| mobilenet_v3_large_224_15ep 基线 | 0.2582 | 0.5195 | 0.4575 | — |
| mobilenet_v3_large_224_15ep + TTA | 0.2582 | 0.5226 | 0.4599 | +0.0000 |
| resnet18_224_15ep 基线 | 0.2818 | 0.5113 | 0.4901 | — |
| resnet18_224_15ep + TTA | 0.2825 | 0.5154 | 0.4952 | +0.0007 |
| 集成 + TTA | 0.2911 | 0.5473 | 0.5051 | +0.0086 |
