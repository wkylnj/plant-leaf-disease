# 模型效率实测（E3）

- 设备：AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD
- 平台：Windows 11 / Python 3.12.10
- PyTorch：2.6.0+cpu
- 输入：224×224，batch=1，线程=8
- 延迟统计：预热后重复测量取中位数（单张毫秒）
- FLOPs 口径：1 MAC = 2 FLOPs，单张前向

| 模型 | 家族 | 参数量(M) | FLOPs(G) | 单张延迟(ms) | 吞吐(img/s) | 延迟标准差(ms) |
|---|---|---|---|---|---|---|
| resnet18 | CNN | 11.196 | 7.254 | 16.7 | 59.7 | 0.9 |
| resnet50 | CNN | 23.586 | 16.349 | 42.9 | 23.3 | 7.3 |
| mobilenet_v3_large | 轻量CNN | 4.251 | 0.861 | 13.6 | 73.3 | 1.0 |
| shufflenet_v2_x1_0 | 轻量CNN | 1.293 | 0.576 | 12.6 | 79.7 | 1.5 |
| efficientnet_b0 | 轻量CNN | 4.056 | 1.538 | 17.9 | 56.0 | 1.2 |
| swin_t | Transformer | 27.549 | 17.959 | 53.8 | 18.6 | 0.9 |

## 相对 ResNet18 的倍数

| 模型 | 参数量 | FLOPs | 延迟 |
|---|---|---|---|
| resnet18 | 1.00× | 1.00× | 1.00× |
| resnet50 | 2.11× | 2.25× | 2.56× |
| mobilenet_v3_large | 0.38× | 0.12× | 0.81× |
| shufflenet_v2_x1_0 | 0.12× | 0.08× | 0.75× |
| efficientnet_b0 | 0.36× | 0.21× | 1.07× |
| swin_t | 2.46× | 2.48× | 3.21× |
