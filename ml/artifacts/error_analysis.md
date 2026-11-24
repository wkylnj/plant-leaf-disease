# E-09 跨域错例分析：模型到底错在哪

> 数据来自 `ml/runs/<run>/crossdomain_preds.csv`，不重新推理。
> 跨域 = PlantDoc 全量 2920 张，27 类映射，零样本（无任何微调）。

## 一句话结论

模型不是「没学会看病」，而是「学会了但按错误的依据判断」：top-1 只有 34.93%，但 top-3 有 57.05%，作物级有 53.22%。错误高度集中在**同作物不同病**与**健康/病害混淆**两类上，且答错时置信度明显低于答对时（模型自己「不确定」），这为「低置信度只给作物名 + 候选列表」的产品设计提供了直接依据。

## 四模型横向对比

| 模型 | top-1 | top-3 | 作物级 | 答对时置信度 | 答错时置信度 | 高置信度错误(≥0.8) |
|---|---:|---:|---:|---:|---:|---:|
| swin_t_224_15ep | 34.93% | 57.05% | 53.22% | 0.670 | 0.511 | 311 |
| resnet18_224_15ep | 28.18% | 49.01% | 51.13% | 0.535 | 0.398 | 124 |
| mobilenet_v3_large_224_15ep | 25.82% | 45.75% | 51.95% | 0.508 | 0.389 | 131 |
| mobilenet_v3_large_daaug_224_15ep | 23.60% | 43.08% | 49.90% | 0.497 | 0.390 | 109 |

## 错误类型分布

把每个错误按「真值作物/病名」与「预测作物/病名」的关系归类：

| 模型 | 同作物 不同病 | 同作物 健康↔病害 | 不同作物 同名病 | 不同作物 涉健康 | 不同作物 不同病 |
|---|---:|---:|---:|---:|---:|
| swin_t_224_15ep | 426 (22.4%) | 108 (5.7%) | 338 (17.8%) | 331 (17.4%) | 697 (36.7%) |
| resnet18_224_15ep | 511 (24.4%) | 159 (7.6%) | 427 (20.4%) | 408 (19.5%) | 592 (28.2%) |
| mobilenet_v3_large_224_15ep | 604 (27.9%) | 159 (7.3%) | 380 (17.5%) | 384 (17.7%) | 639 (29.5%) |
| mobilenet_v3_large_daaug_224_15ep | 534 (23.9%) | 234 (10.5%) | 400 (17.9%) | 356 (16.0%) | 707 (31.7%) |

## 最常见的错误类别对（Top 10）

### swin_t_224_15ep

| 真值 | 预测 | 次数 | 占比(占全部错误) |
|---|---|---:|---:|
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 81 | 4.3% |
| 35_Tomato_Tomato_Yellow_Leaf_Curl_Virus | 25_Squash_Powdery_mildew | 81 | 4.3% |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 67 | 3.5% |
| 20_Potato_Early_blight | 21_Potato_Late_blight | 39 | 2.1% |
| 04_Blueberry_healthy | 11_Grape_Black_rot | 37 | 1.9% |
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 36 | 1.9% |
| 28_Tomato_Bacterial_spot | 32_Tomato_Septoria_leaf_spot | 34 | 1.8% |
| 31_Tomato_Leaf_Mold | 30_Tomato_Late_blight | 34 | 1.8% |
| 23_Raspberry_healthy | 26_Strawberry_Leaf_scorch | 29 | 1.5% |
| 27_Strawberry_healthy | 23_Raspberry_healthy | 29 | 1.5% |

### resnet18_224_15ep

| 真值 | 预测 | 次数 | 占比(占全部错误) |
|---|---|---:|---:|
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 96 | 4.6% |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 75 | 3.6% |
| 35_Tomato_Tomato_Yellow_Leaf_Curl_Virus | 30_Tomato_Late_blight | 49 | 2.3% |
| 27_Strawberry_healthy | 26_Strawberry_Leaf_scorch | 44 | 2.1% |
| 23_Raspberry_healthy | 26_Strawberry_Leaf_scorch | 41 | 2.0% |
| 08_Corn_maize_Common_rust | 09_Corn_maize_Northern_Leaf_Blight | 40 | 1.9% |
| 20_Potato_Early_blight | 21_Potato_Late_blight | 40 | 1.9% |
| 17_Peach_healthy | 19_Pepper_bell_healthy | 38 | 1.8% |
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 35 | 1.7% |
| 31_Tomato_Leaf_Mold | 30_Tomato_Late_blight | 33 | 1.6% |

### mobilenet_v3_large_224_15ep

| 真值 | 预测 | 次数 | 占比(占全部错误) |
|---|---|---:|---:|
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 123 | 5.7% |
| 35_Tomato_Tomato_Yellow_Leaf_Curl_Virus | 30_Tomato_Late_blight | 121 | 5.6% |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 85 | 3.9% |
| 31_Tomato_Leaf_Mold | 30_Tomato_Late_blight | 64 | 3.0% |
| 35_Tomato_Tomato_Yellow_Leaf_Curl_Virus | 25_Squash_Powdery_mildew | 59 | 2.7% |
| 20_Potato_Early_blight | 30_Tomato_Late_blight | 51 | 2.4% |
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 50 | 2.3% |
| 09_Corn_maize_Northern_Leaf_Blight | 10_Corn_maize_healthy | 39 | 1.8% |
| 27_Strawberry_healthy | 23_Raspberry_healthy | 38 | 1.8% |
| 37_Tomato_healthy | 30_Tomato_Late_blight | 35 | 1.6% |

### mobilenet_v3_large_daaug_224_15ep

| 真值 | 预测 | 次数 | 占比(占全部错误) |
|---|---|---:|---:|
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 129 | 5.8% |
| 35_Tomato_Tomato_Yellow_Leaf_Curl_Virus | 30_Tomato_Late_blight | 110 | 4.9% |
| 09_Corn_maize_Northern_Leaf_Blight | 10_Corn_maize_healthy | 85 | 3.8% |
| 32_Tomato_Septoria_leaf_spot | 26_Strawberry_Leaf_scorch | 72 | 3.2% |
| 20_Potato_Early_blight | 30_Tomato_Late_blight | 55 | 2.5% |
| 31_Tomato_Leaf_Mold | 30_Tomato_Late_blight | 53 | 2.4% |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 47 | 2.1% |
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 47 | 2.1% |
| 27_Strawberry_healthy | 23_Raspberry_healthy | 38 | 1.7% |
| 20_Potato_Early_blight | 26_Strawberry_Leaf_scorch | 33 | 1.5% |

## 最难的类别（按 top-1 升序，至少 5 张样本）

| 模型 | 类别 | 样本数 | top-1 | 作物级 |
|---|---|---:|---:|---:|
| swin_t_224_15ep | 06_Cherry_including_sour_healthy | 57 | 0.00% | 0.00% |
| swin_t_224_15ep | 28_Tomato_Bacterial_spot | 110 | 0.00% | 47.27% |
| swin_t_224_15ep | 37_Tomato_healthy | 52 | 0.00% | 36.54% |
| swin_t_224_15ep | 17_Peach_healthy | 112 | 0.89% | 4.46% |
| swin_t_224_15ep | 24_Soybean_healthy | 65 | 1.54% | 1.54% |
| swin_t_224_15ep | 31_Tomato_Leaf_Mold | 91 | 5.49% | 54.95% |
| swin_t_224_15ep | 36_Tomato_Tomato_mosaic_virus | 54 | 5.56% | 38.89% |
| swin_t_224_15ep | 02_Apple_Cedar_apple_rust | 106 | 14.15% | 19.81% |
| resnet18_224_15ep | 37_Tomato_healthy | 52 | 0.00% | 51.92% |
| resnet18_224_15ep | 17_Peach_healthy | 112 | 0.89% | 1.79% |
| resnet18_224_15ep | 06_Cherry_including_sour_healthy | 57 | 1.75% | 1.75% |
| resnet18_224_15ep | 28_Tomato_Bacterial_spot | 110 | 1.82% | 55.45% |
| resnet18_224_15ep | 36_Tomato_Tomato_mosaic_virus | 54 | 1.85% | 64.81% |
| resnet18_224_15ep | 31_Tomato_Leaf_Mold | 91 | 4.40% | 53.85% |
| resnet18_224_15ep | 24_Soybean_healthy | 65 | 7.69% | 7.69% |
| resnet18_224_15ep | 08_Corn_maize_Common_rust | 117 | 10.26% | 90.60% |
| mobilenet_v3_large_224_15ep | 06_Cherry_including_sour_healthy | 57 | 0.00% | 0.00% |
| mobilenet_v3_large_224_15ep | 17_Peach_healthy | 112 | 0.00% | 1.79% |
| mobilenet_v3_large_224_15ep | 36_Tomato_Tomato_mosaic_virus | 54 | 0.00% | 55.56% |
| mobilenet_v3_large_224_15ep | 37_Tomato_healthy | 52 | 0.00% | 76.92% |
| mobilenet_v3_large_224_15ep | 31_Tomato_Leaf_Mold | 91 | 1.10% | 74.73% |
| mobilenet_v3_large_224_15ep | 24_Soybean_healthy | 65 | 1.54% | 1.54% |
| mobilenet_v3_large_224_15ep | 28_Tomato_Bacterial_spot | 110 | 1.82% | 60.91% |
| mobilenet_v3_large_224_15ep | 02_Apple_Cedar_apple_rust | 106 | 7.55% | 18.87% |
| mobilenet_v3_large_daaug_224_15ep | 06_Cherry_including_sour_healthy | 57 | 0.00% | 0.00% |
| mobilenet_v3_large_daaug_224_15ep | 37_Tomato_healthy | 52 | 0.00% | 75.00% |
| mobilenet_v3_large_daaug_224_15ep | 17_Peach_healthy | 112 | 0.89% | 1.79% |
| mobilenet_v3_large_daaug_224_15ep | 28_Tomato_Bacterial_spot | 110 | 0.91% | 49.09% |
| mobilenet_v3_large_daaug_224_15ep | 31_Tomato_Leaf_Mold | 91 | 1.10% | 70.33% |
| mobilenet_v3_large_daaug_224_15ep | 36_Tomato_Tomato_mosaic_virus | 54 | 1.85% | 57.41% |
| mobilenet_v3_large_daaug_224_15ep | 24_Soybean_healthy | 65 | 3.08% | 3.08% |
| mobilenet_v3_large_daaug_224_15ep | 02_Apple_Cedar_apple_rust | 106 | 4.72% | 11.32% |

## 高置信度错误（最危险的失败模式）

模型很有把握却答错，这类错误在产品里无法通过置信度阈值拦住：

### swin_t_224_15ep（共 311 例，占全部错误 16.4%）

| 真值 | 预测 | 置信度 |
|---|---|---:|
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9684 |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9615 |
| 20_Potato_Early_blight | 30_Tomato_Late_blight | 0.9615 |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9593 |
| 20_Potato_Early_blight | 29_Tomato_Early_blight | 0.9570 |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9531 |
| 24_Soybean_healthy | 26_Strawberry_Leaf_scorch | 0.9526 |
| 31_Tomato_Leaf_Mold | 30_Tomato_Late_blight | 0.9506 |

### resnet18_224_15ep（共 124 例，占全部错误 5.9%）

| 真值 | 预测 | 置信度 |
|---|---|---:|
| 20_Potato_Early_blight | 29_Tomato_Early_blight | 0.9967 |
| 32_Tomato_Septoria_leaf_spot | 29_Tomato_Early_blight | 0.9885 |
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 0.9826 |
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 0.9798 |
| 03_Apple_healthy | 19_Pepper_bell_healthy | 0.9727 |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9694 |
| 08_Corn_maize_Common_rust | 09_Corn_maize_Northern_Leaf_Blight | 0.9645 |
| 30_Tomato_Late_blight | 29_Tomato_Early_blight | 0.9621 |

### mobilenet_v3_large_224_15ep（共 131 例，占全部错误 6.0%）

| 真值 | 预测 | 置信度 |
|---|---|---:|
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9885 |
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9835 |
| 20_Potato_Early_blight | 29_Tomato_Early_blight | 0.9817 |
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 0.9795 |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9734 |
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 0.9721 |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9714 |
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 0.9681 |

### mobilenet_v3_large_daaug_224_15ep（共 109 例，占全部错误 4.9%）

| 真值 | 预测 | 置信度 |
|---|---|---:|
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9888 |
| 09_Corn_maize_Northern_Leaf_Blight | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9759 |
| 28_Tomato_Bacterial_spot | 30_Tomato_Late_blight | 0.9647 |
| 31_Tomato_Leaf_Mold | 30_Tomato_Late_blight | 0.9638 |
| 21_Potato_Late_blight | 30_Tomato_Late_blight | 0.9631 |
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9620 |
| 32_Tomato_Septoria_leaf_spot | 26_Strawberry_Leaf_scorch | 0.9612 |
| 08_Corn_maize_Common_rust | 07_Corn_maize_Cercospora_leaf_spot_Gray_leaf_spot | 0.9610 |

