# swin_b others 下采样对比实验

实验设置：SCIN 作为 `external_test`；训练来源为 Dermnet、Mendeley、SkinDisNet；训练阶段使用 `WeightedRandomSampler`，不启用 `loss_class_weights`。

| 方案 | 划分 | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | others Recall |
|---|---|---:|---:|---:|---:|---:|
| No downsample | test | 0.9281 | 0.8124 | 0.8414 | 0.9266 | 0.9733 |
| No downsample | external_test | 0.5241 | 0.2068 | 0.1661 | 0.4125 | 0.8958 |
| Downsample others | test | 0.8611 | 0.8437 | 0.8435 | 0.8618 | 0.8842 |
| Downsample others | external_test | 0.5020 | 0.2190 | 0.1923 | 0.4251 | 0.8290 |

## 简要结论

- 内部 test Macro-F1 变化：+0.0021。
- 外部 external_test Macro-F1 变化：+0.0261。
- 下采样方案将训练来源中的 `others` 每个数据源最多保留 3000 张；SCIN 外部测试集不参与下采样。