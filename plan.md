项目名称：
基于多源皮肤图像数据融合的常见皮肤疾病分类与可解释性分析系统

项目定位：
本项目不只是完成一个简单的皮肤病图像分类器，而是构建一个完整的深度学习实验系统。系统基于 Dermnet、SkinDisNet、Mendeley Skin Disease Classification Dataset 构建主训练数据集，并使用 SCIN 作为外部泛化测试集。项目完成多源数据清洗、类别统一、类别不平衡处理、多模型对比、消融实验、内部测试、外部泛化测试、丰富可视化分析、Grad-CAM 可解释性分析和 Streamlit Web Demo。

项目最终目标：
1. 能区分 acne、eczema、dermatitis、pigmentation、others 五类常见皮肤问题。
2. 能展示模型训练过程、模型性能、分类错误、类别混淆、外部泛化能力和模型关注区域。
3. 能通过 Web Demo 上传图片并输出预测结果、Top-3 置信度和 Grad-CAM 热力图。
4. 报告中体现健康中国、普惠医疗、科技向善、数据伦理和医学 AI 的边界意识。
5. 项目代码结构清晰、实验结果可复现、图表丰富、报告逻辑完整，向优秀标准靠拢。

一、总体技术路线

第一阶段：数据工程
1. 整理 Dermnet、SkinDisNet、Mendeley Skin Disease Classification Dataset 和 SCIN。
2. 将不同数据集的原始标签统一映射到 acne、eczema、dermatitis、pigmentation、others 五类。
3. 记录每张图片的来源数据集、原始标签、统一标签、图片尺寸、哈希值和数据划分。
4. 删除无法打开图片、尺寸过小图片、重复图片和明显异常图片。
5. 统计每个数据集、每个类别、每个划分中的样本数量。
6. 输出数据分析图表，证明数据处理过程规范。

第二阶段：数据增强与类别平衡
1. 使用 RandomResizedCrop、RandomHorizontalFlip、ColorJitter、RandomRotation、RandomErasing。
2. 针对老师要求的背景替换，采用轻量背景扰动，不直接破坏皮肤病灶区域。
3. 进行类别不平衡处理，优先使用 WeightedRandomSampler。
4. 做类别平衡消融实验，对比普通采样、WeightedRandomSampler、class weight 的效果。
5. 输出增强前后样本对比图、类别分布图和类别平衡效果图。

第三阶段：多模型训练
1. ResNet18 作为 baseline。
2. DenseNet121 作为主力模型。
3. EfficientNet-B0 作为轻量化对比模型。
4. 可选增加 ConvNeXt-Tiny 或 MobileNetV3-Small 作为扩展实验。
5. 使用相同数据划分、相同评价指标、相同训练策略进行公平对比。
6. 最终根据验证集 Macro-F1、测试集 Macro-F1、推理速度和可解释性结果选择最佳模型。

第四阶段：模型评估
1. 在内部测试集上评估模型。
2. 在 SCIN 外部测试集上评估泛化能力。
3. 不只看 Accuracy，重点看 Macro-F1、Balanced Accuracy、每类 Recall、每类 F1。
4. 输出内部测试集和外部测试集的对比结果。
5. 分析模型在哪些类别上容易混淆，例如 eczema 与 dermatitis、pigmentation 与 others。

第五阶段：可解释性分析
1. 使用 Grad-CAM 生成热力图。
2. 对每个类别选择若干正确分类样本和错误分类样本进行解释。
3. 展示模型是否关注皮肤病灶区域，而不是背景、水印、边框、衣物等无关区域。
4. 使用 t-SNE 或 UMAP 展示深层特征分布。
5. 展示 DenseNet121 中间层特征图，体现特征提取过程。

第六阶段：前端 Demo
1. 使用 Streamlit 实现 Web Demo。
2. 支持上传皮肤图片。
3. 显示预测类别、Top-3 概率、置信度条形图。
4. 显示 Grad-CAM 热力图。
5. 显示类别说明和注意事项。
6. 显示医学免责声明：本系统仅用于课程设计和学习研究，不能替代医生诊断。
7. 可选加入模型切换功能，让用户选择 DenseNet121、EfficientNet-B0 或 ResNet18 进行预测对比。

二、优秀版数据集使用方案

主训练数据：
Dermnet
SkinDisNet
Mendeley Skin Disease Classification Dataset

外部泛化测试数据：
SCIN

不要把 SCIN 混入训练集。
SCIN 单独作为 external_test，用于体现跨数据集泛化能力。

最终类别：
0 acne
1 eczema
2 dermatitis
3 pigmentation
4 others

类别映射原则：

acne：
Dermnet 中的 Acne and Rosacea Photos
Mendeley 中的 acne
SCIN 中能映射到 acne 或 acne vulgaris 的样本

eczema：
Dermnet 中的 Eczema Photos
SkinDisNet 中的 eczema

dermatitis：
Dermnet 中的 Atopic Dermatitis Photos、Contact Dermatitis Photos、Seborrheic Dermatitis Photos
SkinDisNet 中的 atopic dermatitis、contact dermatitis、seborrheic dermatitis

pigmentation：
Mendeley 中的 hyperpigmentation
Dermnet 中包含 pigmentation、hyperpigmentation、melasma、lentigo 等关键词的类别
SCIN 中能映射到 hyperpigmentation、melasma、dark spot、post-inflammatory hyperpigmentation 的样本

others：
psoriasis
vitiligo
warts
fungal infection
tinea
scabies
nail psoriasis
SJS-TEN
unknown
其他无法归入前四类的皮肤问题

重要说明：
vitiligo 是色素脱失，不是普通色素沉着或色素斑。第一版建议放入 others，不放入 pigmentation。报告中可以说明医学概念边界，体现严谨性。

三、优秀版数据目录结构
```
skin-disease-classification/
├── README.md
├── requirements.txt
├── config.yaml
├── train.py
├── evaluate.py
├── predict.py
├── run_app.py
│
├── data/
│   ├── raw/
│   │   ├── dermnet/
│   │   ├── skindisnet/
│   │   ├── mendeley_skin/
│   │   └── scin/
│   │
│   ├── interim/
│   │   ├── dermnet_samples.csv
│   │   ├── skindisnet_samples.csv
│   │   ├── mendeley_samples.csv
│   │   ├── scin_samples.csv
│   │   ├── all_samples.csv
│   │   ├── cleaned_samples.csv
│   │   ├── duplicate_removed_samples.csv
│   │   ├── split_samples.csv
│   │   └── class_distribution.csv
│   │
│   ├── processed/
│   │   ├── train/
│   │   │   ├── acne/
│   │   │   ├── eczema/
│   │   │   ├── dermatitis/
│   │   │   ├── pigmentation/
│   │   │   └── others/
│   │   ├── val/
│   │   │   ├── acne/
│   │   │   ├── eczema/
│   │   │   ├── dermatitis/
│   │   │   ├── pigmentation/
│   │   │   └── others/
│   │   └── test/
│   │       ├── acne/
│   │       ├── eczema/
│   │       ├── dermatitis/
│   │       ├── pigmentation/
│   │       └── others/
│   │
│   └── external_test/
│       └── scin/
│           ├── acne/
│           ├── eczema/
│           ├── dermatitis/
│           ├── pigmentation/
│           └── others/
│
├── src/
│   ├── datasets/
│   │   ├── skin_dataset.py
│   │   ├── transforms.py
│   │   └── class_balance.py
│   │
│   ├── models/
│   │   └── build_model.py
│   │
│   ├── engine/
│   │   ├── trainer.py
│   │   └── evaluator.py
│   │
│   ├── visualization/
│   │   ├── plot_dataset_stats.py
│   │   ├── plot_aug_examples.py
│   │   ├── plot_curves.py
│   │   ├── plot_confusion_matrix.py
│   │   ├── plot_metrics_bar.py
│   │   ├── plot_roc_pr_curves.py
│   │   ├── plot_tsne_umap.py
│   │   ├── feature_map.py
│   │   ├── gradcam_utils.py
│   │   └── error_analysis.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logger.py
│       ├── metrics.py
│       ├── io.py
│       └── timer.py
│
├── scripts/
│   ├── prepare_dermnet.py
│   ├── prepare_skindisnet.py
│   ├── prepare_mendeley.py
│   ├── prepare_scin.py
│   ├── merge_datasets.py
│   ├── clean_images.py
│   ├── remove_duplicates.py
│   ├── split_dataset.py
│   ├── make_dataset_report.py
│   └── generate_all_figures.py
│
├── app/
│   ├── streamlit_app.py
│   ├── disease_info.py
│   └── assets/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── figures/
│   ├── gradcam/
│   ├── feature_maps/
│   ├── error_cases/
│   └── reports/
│
└── third_party/
    ├── README.md
    └── LICENSE_NOTES.md
```
四、优秀版必须输出的可视化图表

A. 数据集分析图表

1. 总类别分布柱状图
文件名：
outputs/figures/01_class_distribution_bar.png

内容：
展示 acne、eczema、dermatitis、pigmentation、others 每类样本数量。

意义：
证明你分析了类别不平衡问题。

2. 数据来源分布图
文件名：
outputs/figures/02_source_dataset_distribution.png

内容：
展示 Dermnet、SkinDisNet、Mendeley 各自贡献多少图片。

意义：
体现多源数据融合。

3. 类别-数据源热力图
文件名：
outputs/figures/03_class_source_heatmap.png

内容：
横轴为数据集来源，纵轴为统一类别，颜色表示样本数量。

意义：
说明每个类别主要来自哪个数据集。例如 pigmentation 主要来自 Mendeley，dermatitis 主要来自 SkinDisNet。

4. train/val/test 划分分布图
文件名：
outputs/figures/04_split_distribution.png

内容：
展示每个类别在 train、val、test 中的样本数量。

意义：
证明划分合理，符合训练、验证、测试流程要求。

5. 原始图片尺寸分布图
文件名：
outputs/figures/05_image_size_distribution.png

内容：
展示图片宽高分布、面积分布。

意义：
说明为什么统一 Resize 到 224 x 224 或 256 x 256。

6. 数据清洗统计图
文件名：
outputs/figures/06_data_cleaning_summary.png

内容：
原始图片数量、删除损坏图片数量、删除过小图片数量、删除重复图片数量、最终保留数量。

意义：
体现数据工程完整性。

7. 每类样本九宫格展示图
文件名：
outputs/figures/07_class_sample_grid.png

内容：
每个类别展示 9 张代表图像。

意义：
让老师直观看到数据内容。

B. 数据增强可视化图表

8. 数据增强前后对比图
文件名：
outputs/figures/08_augmentation_examples.png

内容：
同一张图片展示原图、随机裁剪、水平翻转、颜色扰动、随机擦除、轻量背景扰动后的效果。

意义：
对应老师要求的数据预处理和增强。

9. 背景扰动示意图
文件名：
outputs/figures/09_background_perturbation_examples.png

内容：
展示边缘模糊、局部擦除、弱颜色扰动等方式。

意义：
说明本项目没有破坏皮肤病灶区域，而是进行轻量背景扰动。

10. 类别平衡前后对比图
文件名：
outputs/figures/10_class_balance_before_after.png

内容：
普通采样下每类样本数量 vs WeightedRandomSampler 后每类采样频率。

意义：
体现类别不平衡处理的效果。

C. 训练过程可视化图表

11. 训练损失曲线
文件名：
outputs/figures/11_train_val_loss_curve.png

内容：
ResNet18、DenseNet121、EfficientNet-B0 的 train loss 和 val loss。

意义：
分析模型是否收敛、是否过拟合。

12. 验证准确率曲线
文件名：
outputs/figures/12_val_accuracy_curve.png

内容：
三个模型的验证 Accuracy 随 epoch 变化曲线。

意义：
展示训练过程。

13. 验证 Macro-F1 曲线
文件名：
outputs/figures/13_val_macro_f1_curve.png

内容：
三个模型的验证 Macro-F1 曲线。

意义：
因为类别不平衡，Macro-F1 比 Accuracy 更重要。

14. 学习率变化曲线
文件名：
outputs/figures/14_learning_rate_curve.png

内容：
展示 CosineAnnealingLR 或 ReduceLROnPlateau 的学习率变化。

意义：
体现训练策略完整。

15. 过拟合分析图
文件名：
outputs/figures/15_overfitting_gap.png

内容：
train accuracy 和 val accuracy 的差距，或者 train loss 和 val loss 的差距。

意义：
分析模型泛化能力。

D. 模型对比图表

16. 多模型指标对比柱状图
文件名：
outputs/figures/16_model_comparison_metrics.png

内容：
ResNet18、DenseNet121、EfficientNet-B0 的 Accuracy、Macro Precision、Macro Recall、Macro-F1。

意义：
体现模型选择过程。

17. 模型参数量与性能对比图
文件名：
outputs/figures/17_params_vs_f1.png

内容：
横轴为参数量或模型大小，纵轴为 Macro-F1。

意义：
说明模型性能与复杂度之间的权衡。

18. 推理时间对比图
文件名：
outputs/figures/18_inference_time_comparison.png

内容：
三个模型单张图片平均推理时间。

意义：
说明为什么最终 Demo 选择某个模型。

19. 模型雷达图
文件名：
outputs/figures/19_model_radar_chart.png

内容：
维度包括 Accuracy、Macro-F1、Recall、推理速度、模型大小、可解释性。

意义：
视觉效果丰富，适合 PPT 展示。

E. 测试结果图表

20. 内部测试集混淆矩阵
文件名：
outputs/figures/20_confusion_matrix_internal_raw.png

内容：
原始数量混淆矩阵。

意义：
展示每个类别的预测情况。

21. 内部测试集归一化混淆矩阵
文件名：
outputs/figures/21_confusion_matrix_internal_normalized.png

内容：
每类归一化后的混淆比例。

意义：
更清楚地看到 eczema 和 dermatitis 是否容易混淆。

22. SCIN 外部测试集混淆矩阵
文件名：
outputs/figures/22_confusion_matrix_scin.png

内容：
最终模型在 SCIN 上的混淆矩阵。

意义：
展示跨数据集泛化效果。

23. 内部测试 vs SCIN 外部测试性能对比图
文件名：
outputs/figures/23_internal_vs_external_metrics.png

内容：
内部测试集和 SCIN 外部测试集的 Accuracy、Macro-F1、Macro Recall 对比。

意义：
这是优秀项目的关键亮点，说明你不是只在同源测试集上评估。

24. 每类 Precision、Recall、F1 柱状图
文件名：
outputs/figures/24_per_class_metrics.png

内容：
每个类别对应的 Precision、Recall、F1。

意义：
可以重点分析 pigmentation 和 dermatitis 的识别难点。

25. Balanced Accuracy 对比图
文件名：
outputs/figures/25_balanced_accuracy_comparison.png

内容：
普通 Accuracy 和 Balanced Accuracy 对比。

意义：
体现类别不平衡场景下评价指标的合理性。

F. ROC / PR 曲线

26. 多类别 ROC 曲线
文件名：
outputs/figures/26_multiclass_roc_curve.png

内容：
one-vs-rest 方式绘制每个类别的 ROC 曲线和 AUC。

意义：
丰富实验分析。

27. 多类别 PR 曲线
文件名：
outputs/figures/27_multiclass_pr_curve.png

内容：
每个类别的 Precision-Recall 曲线。

意义：
在类别不平衡任务中，PR 曲线比 ROC 更有意义。

G. 模型解释与特征可视化

28. Grad-CAM 正确分类样本图
文件名：
outputs/gradcam/28_gradcam_correct_cases.png

内容：
每个类别选 3 张正确分类样本，展示原图、热力图、叠加图。

意义：
说明模型关注了病灶区域。

29. Grad-CAM 错误分类样本图
文件名：
outputs/gradcam/29_gradcam_wrong_cases.png

内容：
展示错误分类样本的 Grad-CAM。

意义：
分析模型为什么错，例如关注背景、边缘、非病灶区域。

30. 每类 Grad-CAM 汇总图
文件名：
outputs/gradcam/30_gradcam_class_summary.png

内容：
acne、eczema、dermatitis、pigmentation、others 每类若干样本的热力图。

意义：
非常适合 PPT 展示。

31. 中间层特征图可视化
文件名：
outputs/feature_maps/31_feature_maps.png

内容：
展示模型浅层、中层、深层部分 feature maps。

意义：
对应老师要求的特征图可视化。

32. t-SNE 特征分布图
文件名：
outputs/figures/32_tsne_feature_distribution.png

内容：
提取模型倒数第二层特征，用 t-SNE 降维到二维，不同类别用不同颜色表示。

意义：
展示模型学习到的特征是否具有类别可分性。

33. UMAP 特征分布图
文件名：
outputs/figures/33_umap_feature_distribution.png

内容：
可选，用 UMAP 替代或补充 t-SNE。

意义：
进一步增强可视化丰富度。

H. 错误分析图表

34. 错误样本九宫格
文件名：
outputs/error_cases/34_error_cases_grid.png

内容：
展示预测错误的图片，标注真实类别、预测类别、置信度。

意义：
体现你不是只报结果，还进行了错误分析。

35. Top confused class pairs 图
文件名：
outputs/figures/35_top_confused_pairs.png

内容：
统计最容易混淆的类别对，例如 eczema -> dermatitis、pigmentation -> others。

意义：
深入分析模型不足。

36. 置信度分布图
文件名：
outputs/figures/36_confidence_distribution.png

内容：
正确预测样本和错误预测样本的置信度分布。

意义：
分析模型是否过度自信。

37. 可靠性校准图
文件名：
outputs/figures/37_reliability_diagram.png

内容：
展示预测置信度和真实准确率之间的关系。

意义：
非常加分，体现你对模型可信度有分析。

38. ECE 指标图
文件名：
outputs/figures/38_expected_calibration_error.png

内容：
展示模型校准误差。

意义：
医学 AI 场景中模型不能只看准确率，也要看可信度。

I. 前端 Demo 可视化

39. Web Demo 上传界面截图
文件名：
outputs/figures/39_demo_upload_page.png

40. Web Demo 预测结果截图
文件名：
outputs/figures/40_demo_prediction_result.png

41. Web Demo Grad-CAM 展示截图
文件名：
outputs/figures/41_demo_gradcam_result.png

42. Web Demo 多模型对比截图，可选
文件名：
outputs/figures/42_demo_model_compare.png

五、优秀版实验设计

实验 1：数据集构建与类别分布分析

目的：
证明数据集来源清楚、类别映射合理、数据划分规范。

内容：
统计 Dermnet、SkinDisNet、Mendeley 的样本数量。
统计五个统一类别的样本数量。
统计 train、val、test 划分比例。
分析类别不平衡问题。

输出图表：
class_distribution_bar.png
source_dataset_distribution.png
class_source_heatmap.png
split_distribution.png
class_sample_grid.png

报告重点：
本项目不是直接使用单一数据集，而是进行了多源数据融合和统一标签映射。

实验 2：数据增强可视化与预处理分析

目的：
证明满足随机裁剪、水平翻转、颜色扰动、背景扰动等预处理要求。

内容：
展示增强前后图片。
说明验证集和测试集不使用随机增强。
说明背景扰动采用轻量策略，不破坏医学语义。

输出图表：
augmentation_examples.png
background_perturbation_examples.png

报告重点：
皮肤疾病图像中病灶区域与皮肤背景联系紧密，因此背景替换不能像普通物体分类一样强行替换背景。本项目采用局部擦除、边缘模糊、弱颜色扰动等轻量背景扰动策略。

实验 3：多模型对比实验

模型：
ResNet18
DenseNet121
EfficientNet-B0

训练设置：
输入尺寸 224 x 224。
ImageNet 预训练权重。
AdamW 优化器。
学习率 0.0001。
batch size 16 或 32。
训练 30 epoch。
使用早停策略。
保存验证集 Macro-F1 最优模型。

评价指标：
Accuracy
Macro Precision
Macro Recall
Macro-F1
Balanced Accuracy
Per-class F1
推理时间
参数量

输出图表：
train_val_loss_curve.png
val_accuracy_curve.png
val_macro_f1_curve.png
model_comparison_metrics.png
params_vs_f1.png
inference_time_comparison.png
model_radar_chart.png

报告重点：
ResNet18 用于验证基础流程。
DenseNet121 作为主力模型，因为 DenseNet 特征复用能力强，适合皮肤病图像中的细粒度纹理特征提取。
EfficientNet-B0 作为轻量化对比模型，分析性能与推理速度的平衡。

实验 4：类别不平衡处理消融实验

对比方案：
方案 A：普通随机采样
方案 B：WeightedRandomSampler
方案 C：CrossEntropyLoss with class weight
方案 D：WeightedRandomSampler + class weight，可选

建议主方案：
优先使用 WeightedRandomSampler。

评价重点：
Macro-F1 是否提升。
少数类 pigmentation 的 Recall 是否提升。
eczema 和 dermatitis 的混淆是否降低。
others 是否过度吸收其他类别。

输出图表：
class_balance_before_after.png
per_class_metrics_before_after_balance.png
confusion_matrix_balance_ablation.png

报告重点：
在皮肤病分类中，不同类别样本数量差异较大，仅使用 Accuracy 可能掩盖少数类识别差的问题。因此使用类别平衡策略，并以 Macro-F1 和每类 Recall 作为重要评价指标。

实验 5：外部泛化测试实验

训练数据：
Dermnet + SkinDisNet + Mendeley

外部测试数据：
SCIN

实验方式：
SCIN 不参与训练。
SCIN 不参与验证。
SCIN 只在最终模型训练完成后作为 external_test 使用。

评价指标：
SCIN Accuracy
SCIN Macro-F1
SCIN per-class Recall
SCIN confusion matrix

输出图表：
confusion_matrix_scin.png
internal_vs_external_metrics.png
scin_error_cases.png

报告重点：
内部测试集反映模型在同源数据分布下的表现；SCIN 外部测试集更接近真实手机拍摄场景，能够检验模型跨数据集、跨拍摄环境的泛化能力。

实验 6：模型可解释性实验

方法：
Grad-CAM
Feature Map Visualization
t-SNE / UMAP

内容：
对每个类别选择正确分类样本生成 Grad-CAM。
对错误分类样本生成 Grad-CAM。
展示 DenseNet121 中间层特征图。
提取倒数第二层特征，用 t-SNE 或 UMAP 可视化类别分布。

输出图表：
gradcam_correct_cases.png
gradcam_wrong_cases.png
gradcam_class_summary.png
feature_maps.png
tsne_feature_distribution.png

报告重点：
如果 Grad-CAM 热力图集中在病灶区域，说明模型判断具有一定合理性；如果热力图集中在背景、水印、边框等区域，说明模型可能学习到了无关特征，需要在后续改进中加强数据清洗和鲁棒性训练。

实验 7：模型可信度与错误分析实验

内容：
分析错误样本。
统计最常混淆类别对。
绘制置信度分布。
绘制可靠性校准图。
计算 ECE。

输出图表：
error_cases_grid.png
top_confused_pairs.png
confidence_distribution.png
reliability_diagram.png
expected_calibration_error.png

报告重点：
医学辅助识别系统不仅要关注准确率，还要关注模型是否过度自信。对于低置信度样本，系统应提示用户结果不确定，不能给出过度确定的判断。

六、优秀版模型选择策略

最终推荐：

Baseline：
ResNet18

主模型：
DenseNet121

轻量模型：
EfficientNet-B0

可选扩展模型：
MobileNetV3-Small，用于轻量部署对比
ConvNeXt-Tiny，用于更强性能对比

不建议第一版就把集成模型作为主系统。
可以做一个扩展实验：
DenseNet121 + EfficientNet-B0 soft voting

集成公式：
P_final = 0.6 * P_DenseNet121 + 0.4 * P_EfficientNetB0

但最终 Web Demo 默认使用单模型。
原因：
单模型推理速度更快。
Grad-CAM 更容易解释。
系统部署更简单。
答辩讲解更清楚。

最终模型选择依据：
1. 验证集 Macro-F1
2. 内部测试集 Macro-F1
3. SCIN 外部测试集 Macro-F1
4. 每类 Recall，尤其是 pigmentation、dermatitis
5. 推理速度
6. Grad-CAM 可解释性是否合理

最终报告推荐表述：
综合内部测试集结果、SCIN 外部泛化测试结果、推理速度和 Grad-CAM 可解释性分析，本文选择 DenseNet121 作为最终 Web Demo 默认模型。DenseNet121 在 Macro-F1 和少数类 Recall 上表现稳定，同时模型关注区域较多集中于皮肤病灶区域，具有较好的可解释性。

七、优秀版前端 Demo 功能

基础功能：
1. 上传图片。
2. 显示原图。
3. 显示预测类别。
4. 显示预测置信度。
5. 显示 Top-3 类别概率。
6. 显示 Grad-CAM 热力图。
7. 显示疾病类别说明。
8. 显示免责声明。

优秀版功能：
1. 模型选择下拉框：
   ResNet18
   DenseNet121
   EfficientNet-B0

2. 置信度进度条：
   显示 Top-3 预测结果的概率条形图。

3. 不确定性提示：
   如果最高置信度低于 0.6，提示：
   当前预测置信度较低，结果仅供参考，建议咨询专业医生。

4. Grad-CAM 对比显示：
   左侧原图。
   中间热力图。
   右侧原图叠加热力图。

5. 类别说明卡片：
   acne：痤疮类皮肤问题。
   eczema：湿疹类皮肤问题。
   dermatitis：皮炎类皮肤问题。
   pigmentation：色素沉着或色素斑类问题。
   others：其他皮肤问题。

6. 医学免责声明：
   本系统仅用于神经网络与深度学习课程设计、学习研究和辅助展示，不具备医学诊断资质，预测结果不能替代医生诊断。如存在皮肤异常，应及时咨询专业医生。

7. 实验结果展示页面：
   在 Demo 中加入模型对比表、混淆矩阵、训练曲线截图。
   这样答辩时可以直接演示完整系统。

八、优秀版报告图表清单

报告中建议至少放 15 到 20 张图表。

必放图表：
1. 技术路线图
2. 系统架构图
3. 数据处理流程图
4. 类别分布柱状图
5. 数据来源分布图
6. 类别-数据源热力图
7. train/val/test 划分图
8. 数据增强示例图
9. 多模型训练 loss 曲线
10. 多模型验证 Macro-F1 曲线
11. 模型性能对比柱状图
12. 内部测试集混淆矩阵
13. SCIN 外部测试集混淆矩阵
14. 内部测试集 vs 外部测试集指标对比图
15. 每类 Precision / Recall / F1 图
16. Grad-CAM 正确样本图
17. Grad-CAM 错误样本图
18. 错误样本分析图
19. t-SNE 特征分布图
20. Web Demo 界面截图

可选加分图表：
21. PR 曲线
22. ROC 曲线
23. 置信度分布图
24. 可靠性校准图
25. 模型参数量 vs Macro-F1 图
26. 推理时间对比图
27. 模型雷达图
28. 类别平衡前后对比图

九、优秀版报告结构

第一章 绪论
1.1 研究背景
1.2 常见皮肤疾病辅助识别的应用价值
1.3 研究意义：健康中国、普惠医疗、公共卫生服务
1.4 本项目主要工作
1.5 项目创新点

第二章 数据集与数据预处理
2.1 数据集来源
2.2 Dermnet 数据集说明
2.3 SkinDisNet 数据集说明
2.4 Mendeley Skin Disease Classification Dataset 说明
2.5 SCIN 外部测试集说明
2.6 标签统一与类别映射
2.7 数据清洗
2.8 数据划分
2.9 数据增强
2.10 背景扰动策略
2.11 类别不平衡处理
2.12 数据伦理与合规说明

第三章 方法设计
3.1 系统总体架构
3.2 迁移学习方法
3.3 ResNet18 基线模型
3.4 DenseNet121 主模型
3.5 EfficientNet-B0 对比模型
3.6 损失函数与优化器
3.7 学习率调度与早停机制
3.8 类别不平衡处理方法
3.9 Grad-CAM 可解释性方法
3.10 评价指标设计

第四章 实验设计
4.1 实验环境
4.2 数据划分
4.3 实验参数设置
4.4 多模型对比实验
4.5 类别不平衡消融实验
4.6 SCIN 外部泛化实验
4.7 模型可解释性实验
4.8 错误样本分析实验

第五章 实验结果与分析
5.1 数据集统计结果
5.2 数据增强可视化结果
5.3 训练过程分析
5.4 多模型对比结果
5.5 内部测试集性能分析
5.6 SCIN 外部测试集性能分析
5.7 混淆矩阵分析
5.8 每类指标分析
5.9 Grad-CAM 可解释性分析
5.10 错误样本与置信度分析
5.11 小结

第六章 系统实现
6.1 项目目录结构
6.2 数据处理模块
6.3 模型训练模块
6.4 模型评估模块
6.5 可视化模块
6.6 Streamlit 前端 Demo
6.7 系统运行示例

第七章 思政元素与社会价值
7.1 健康中国与普惠医疗
7.2 医学 AI 的科技向善
7.3 数据伦理与隐私保护
7.4 工匠精神：模型调优与实验严谨性
7.5 技术边界：不能替代医生诊断

第八章 总结与展望
8.1 项目完成内容
8.2 项目亮点
8.3 存在不足
8.4 后续改进方向

十、优秀版项目亮点总结

亮点一：
多源公开数据集融合。
本项目不是只使用单一数据集，而是融合 Dermnet、SkinDisNet 和 Mendeley 数据集，并使用 SCIN 进行外部泛化测试。

亮点二：
统一标签映射。
将不同数据集的原始皮肤病标签统一为 acne、eczema、dermatitis、pigmentation、others 五个类别，解决多源数据标签体系不一致问题。

亮点三：
类别不平衡处理。
使用 WeightedRandomSampler 和 class weight 进行对比，重点改善少数类识别能力。

亮点四：
多模型对比。
比较 ResNet18、DenseNet121、EfficientNet-B0，从准确率、Macro-F1、推理速度和模型大小等角度选择最终模型。

亮点五：
外部泛化验证。
SCIN 不参与训练，只作为 external_test，验证模型在真实手机拍摄皮肤图像上的泛化能力。

亮点六：
丰富可视化。
包括数据分布、数据增强、训练曲线、混淆矩阵、每类指标、ROC/PR 曲线、t-SNE、Grad-CAM、错误样本、置信度分布和前端 Demo 截图。

亮点七：
可解释性分析。
使用 Grad-CAM 分析模型关注区域，讨论模型是否真正关注皮肤病灶，而不是背景或水印。

亮点八：
可信度分析。
通过置信度分布和可靠性校准图分析模型是否过度自信，体现医学 AI 场景下的谨慎性。

亮点九：
完整 Web Demo。
实现图片上传、类别预测、Top-3 概率、Grad-CAM 热力图和医学免责声明。

亮点十：
社会价值明确。
围绕健康中国、普惠医疗、科技向善、数据伦理和模型边界进行论述。

十一、最终答辩主线

答辩时不要按代码讲，按这个逻辑讲：

第一步：
为什么做这个题？
常见皮肤问题具有较高发生率，利用深度学习进行辅助识别具有一定公共卫生和健康科普价值。

第二步：
数据怎么来？
使用 Dermnet、SkinDisNet、Mendeley 构建训练数据，SCIN 作为外部泛化测试集，所有数据来自公开数据集，来源合规。

第三步：
怎么处理数据？
统一标签、清洗异常图片、删除重复图片、划分 train/val/test、做数据增强和类别平衡。

第四步：
用了什么模型？
用 ResNet18 做 baseline，用 DenseNet121 做主模型，用 EfficientNet-B0 做轻量化对比。

第五步：
怎么评估？
不仅看 Accuracy，还看 Macro-F1、每类 Recall、混淆矩阵，并额外在 SCIN 上测试泛化能力。

第六步：
有什么可视化？
训练曲线、混淆矩阵、类别指标图、外部测试对比、Grad-CAM、t-SNE、错误样本分析和 Web Demo。

第七步：
最终结果如何？
说明 DenseNet121 或你最终选择的模型在内部测试和外部测试中的表现，并分析容易混淆的类别。

第八步：
系统怎么展示？
Streamlit Demo 上传图片，输出预测类别、Top-3 概率和 Grad-CAM 热力图。

第九步：
社会价值和边界是什么？
本系统面向健康科普和课程研究，不能替代医生诊断，体现科技向善和数据伦理。

十二、最终优秀版一句话总结

本项目基于多源公开皮肤病图像数据集，构建了一个面向痤疮、湿疹、皮炎、色素斑和其他皮肤问题的深度学习分类系统。项目不仅完成了数据清洗、标签统一、数据增强、类别不平衡处理、多模型训练和测试评估，还引入 SCIN 外部泛化测试、Grad-CAM 可解释性分析、t-SNE 特征可视化、错误样本分析和 Streamlit 前端 Demo。整体设计兼顾模型性能、实验规范、可解释性、可视化展示和医学 AI 的社会责任，符合优秀课程设计对完整性、严谨性和展示效果的要求。