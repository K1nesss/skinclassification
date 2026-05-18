# convnext_base 类别平衡消融实验

| 方案 | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| No balance | 0.8472 | 0.6833 | 0.7318 | 0.8378 |
| Weighted sampler | 0.8452 | 0.6965 | 0.7362 | 0.8384 |
| Class-weighted loss | 0.8365 | 0.6600 | 0.7121 | 0.8247 |
| Sampler + class weight | 0.8306 | 0.6750 | 0.7162 | 0.8234 |

说明：该实验固定模型结构，只改变类别不平衡处理策略，用于分析采样平衡和损失加权对分类性能的影响。