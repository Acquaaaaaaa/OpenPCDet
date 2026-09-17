# PillarHist R8 完整 FP32 精度复现闭环任务书

版本：v1.2  
状态：DRAFT，执行前必须发布并冻结 R8 指导文档/预注册文件  
适用仓库：`/home/ksyang/OpenPCDet`  
前置提交：R6 `c7afac9`，R7 `b448047`

本任务书与部署指导文档独立编号：本文件的 `v1.2` 不等同于待冻结的 `PillarHist_OpenPCDet_deployment_guideline_v1.4.md`。

## 1. R8 的目标与边界

R8 只回答两个问题：

1. 在完整训练预算下，当前 OpenPCDet 中的 PillarHist 是否能够稳定完成 FP32 训练并形成有效检测模型；
2. 在相同训练协议、相同数据顺序和成对初始化下，PillarHist 相对本地官方 PointPillars baseline 的精度变化是多少，与论文公开的 KITTI validation 结果相差多少。

R8 不包括：

- PTQ、fake quant、W8A8 或其他量化实验；
- TensorRT、ONNX、目标硬件部署；
- KITTI test server 正式提交；
- nuScenes 或 Waymo 完整复现；
- 为达到论文数值而在看到结果后修改超参数；
- 将 R7 的 5,000-step AP 当作最终精度。

R8 是否闭环，只取决于协议是否冻结、正式运行是否有效、证据是否完整以及结论是否按预注册规则生成。未达到论文 AP、PillarHist 没有提升或结果为 `MIXED/NOT_SUPPORTED`，都可以构成一次有效闭环。

## 2. R7 结论的继承方式

R7 已以 `INCONCLUSIVE` 闭环：

- `PH_NORM_LINEAR` 两 seed 均值略高；
- `PH_NORM_BNRELU` 与其差距为 `0.381 AP`，低于冻结的 `1.0 AP` tie margin；
- R7 没有证明任一 projection 结构具有显著优势；
- R8 不得表述为“R7 已选出唯一胜者”。

R8 由人工评审确定主轨，不回改 R7 裁决。

### 2.1 推荐的正式主轨

| 轨道 | 作用 | VFE/坐标/projection | 是否阻塞 R8 闭环 |
|---|---|---|---|
| `PP_OFFICIAL_FULL` | 本地因果对照 | 官方 PointPillars VFE | 是 |
| `PH_PAPER_LITERAL_FULL` | 论文语义主复现 | `raw_meter_xy + Linear`，64 bins | 是 |

选择 `PH_PAPER_LITERAL_FULL` 的原因是它最接近论文中“高度直方图、强度直方图、pillar center xy、线性投影”的字面结构。该选择代表“论文忠实性优先”，不代表 R7 已证明它优于 normalized 变体。

### 2.2 非阻塞工程扩展

`PH_NORM_LINEAR_FULL` 可作为工程优化副轨，但默认不属于 R8 最小闭环：

- 只有在计算预算单独批准、并在 v1.4 冻结前预注册时才运行；
- 不得替代 `PH_PAPER_LITERAL_FULL` 的论文复现主轨；
- 不得把 normalized 版本的结果直接写成论文原方法复现结果；
- 副轨中断或未执行不阻塞 R8 主闭环。

不得在看到 seed 666 或任何正式 R8 结果后再决定是否增加 normalized 副轨。

### 2.3 人工选轨决策证据

v1.4 冻结前必须生成：

`PillarHist_R8_track_admission_decision_v1.4.json`

至少记录：

- R7 最终状态为 `INCONCLUSIVE`；
- R7 排名、两 seed 均值和冻结的 tie margin；
- R8 选择 `raw_meter_xy + Linear` 是基于论文忠实性优先；
- 该选择不表示 raw 轨在 R7 中精度更优；
- `raw_meter_xy` 是根据论文公开描述作出的实现解释，并非作者代码已经确认的唯一实现；
- normalized 副轨是否获批及其阻塞属性；
- 决策时间、决策人/执行主体、Git HEAD；
- 本任务书、v1.4、预注册和配置的 SHA-256。决策文件自身的 SHA-256 由外部 `document_hashes.json` 记录，任何文件均不得要求在自身内容中记录自身哈希。

## 3. 执行前硬入口门禁

以下条件全部满足后，才允许启动正式 R8 训练。

### 3.1 发布并冻结 v1.4

必须生成并冻结：

- `PillarHist_OpenPCDet_deployment_guideline_v1.4.md`；
- `PillarHist_R8_preregistration_v1.4.json`；
- `PillarHist_R8_track_admission_decision_v1.4.json`；
- R8 validation/split manifest；
- 本任务书的 SHA-256；
- 所有配置、脚本和数据身份的 SHA-256。

预注册必须早于任何正式 R8 checkpoint 和正式 full-validation 结果。允许在冻结前运行标记为 `EXPLORATORY` 的构建、显存和短前向测试，但不得把这些结果写入正式 R8 精度结论。

### 3.2 清理并冻结 Git 起点

R8 开始前必须：

1. 核对当前工作树中 PillarHist 工具脚本和其他未提交修改；
2. 区分真实内容修改、权限/换行差异和既有用户修改；
3. 不删除、不覆盖 Focal Sparse Conv 等无关用户修改；
4. 将 R8 所需代码、配置、文档形成唯一入口 commit；
5. 在预注册中记录 `git_head`、`git_diff_sha256` 和分支名；
6. 对未跟踪的 R6/R7 `outputs/` 做原地保留和独立备份。

正式训练期间不得修改模型数学语义、数据协议、optimizer、scheduler、epoch 数、主指标或裁决规则。仅修复不改变数学语义的运行器/记录器错误时，必须形成 amendment，说明影响范围并按第 10 节处理。

### 3.3 R6/R7 回归门禁

入口 commit 上至少完成：

- `tests/pillarhist` 全量回归；
- TST-018 继续为 PASS；
- 已知官方 Scatter 空输入限制继续作为有记录的 expected failure；
- `PP_OFFICIAL_FULL` 和 `PH_PAPER_LITERAL_FULL` 均可构建；
- 一个真实 KITTI training batch 可完成 inference、loss 和 backward；
- 两轨输出给 Scatter 的 contract 均为 `[V,64]`，坐标 contract 不变；
- checkpoint save/load smoke test 通过。

保存 pytest 文本报告和 JUnit XML，不只在闭环报告中记录摘要。

## 4. 数据集与评测身份

### 4.1 KITTI 数据身份

正式数据 manifest 至少记录：

- KITTI 根目录；
- train/val split 文件及 SHA-256；
- train、val frame 数；
- info PKL 路径、条目数及 SHA-256；
- GT database 路径、条目数及 SHA-256；
- 点云、标注和 calibration 文件抽样 checksum；
- OpenPCDet 数据处理和增强配置；
- 类别顺序和标准 IoU 阈值。

目标 split 为常用 KITTI train/val 划分：当前预期 train 3,712 帧、validation 3,769 帧。以实际冻结的 split/info 文件条目数和 SHA-256 为唯一事实来源，不因论文文字差异临时切换数据；若实测数量与预期不符，必须在冻结前查明并记录原因。

### 4.2 正式评测指标

每个正式轨道的主 checkpoint（固定为最后一个 epoch）必须在全部 3,769 个 validation frame 上输出：

- Car：3D AP_R40 Easy / Moderate / Hard；
- Pedestrian：3D AP_R40 Easy / Moderate / Hard；
- Cyclist：3D AP_R40 Easy / Moderate / Hard；
- 每个类别三个难度的算术平均；
- 三类别 Moderate 宏平均，作为与 R7 的辅助衔接指标；
- prediction count、recall 和评测脚本版本。

论文直接比对以 Car/Pedestrian 的 3D AP_R40 为主。论文没有提供同表中的 Cyclist PillarHist KITTI 数值，因此 Cyclist 只报告本地结果，不计算“论文复现误差”。

### 4.3 论文参照值

| 模型 | Car Easy | Car Mod. | Car Hard | Car 三难度均值 | Ped Easy | Ped Mod. | Ped Hard | Ped 三难度均值 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PointPillars | 87.08 | 77.90 | 74.97 | 79.98 | 54.71 | 49.01 | 44.52 | 49.41 |
| PH-PointPillars | 88.80 | 79.13 | 76.30 | 81.42* | 57.45 | 50.42 | 45.36 | 51.07** |
| 论文提升 | +1.72 | +1.23 | +1.33 | +1.44* | +2.74 | +1.41 | +0.84 | +1.66 |

`*` 主论文 Table 5 报告 Car mAP 为 `81.42`、相对 `79.98` 的提升为 `+1.44`；由公开的三个难度值重算得到 `(88.80 + 79.13 + 76.30) / 3 = 81.41`、提升 `+1.43`，补充材料也报告 `81.41/+1.43`。

`**` 主论文 Table 5 报告 Pedestrian mAP 为 `51.07`；由公开的三个难度值重算得到 `(57.45 + 50.42 + 45.36) / 3 = 51.0767`，按常规两位小数显示为 `51.08`。其相对 PointPillars 的重算提升仍为 `+1.66`（两位小数）。

正式产物对 Car 和 Pedestrian 均必须同时保存：

- `paper_reported_map`；
- `recomputed_from_displayed_ap`；
- `paper_reported_delta`；
- `recomputed_delta`。

逐难度 AP 是主要外部比较依据，不受上述舍入差异影响。

论文值只作为外部参照，不作为运行有效性的硬门槛。正式报告必须同时给出：

1. 本地绝对 AP 与论文 AP 的差；
2. 本地 PillarHist 相对本地 PointPillars 的成对差；
3. 本地相对提升与论文相对提升的差。

## 5. 完整训练协议

### 5.1 配置来源

R8 默认采用“OpenPCDet 受控复现”口径：

- 以冻结 commit 中的官方 KITTI PointPillars 完整训练配置为训练协议来源；
- 单 GPU 训练；
- 优先采用 `BATCH_SIZE_PER_GPU=4`，global batch 为 4；
- gradient accumulation 固定为 1；
- AMP 关闭；
- DataLoader workers 优先固定为 0；
- LR 固定为官方配置的 `0.003`，除非 batch 4 显存 smoke test 失败并在正式结果前重新冻结；
- 完整训练固定为 80 epochs；
- 两轨只允许 VFE 结构及其必然参数差异不同；
- Adam/OneCycle、weight decay、momentum、数据增强、anchor/head、loss、NMS 等保持一致；
- resolved config 中必须展开所有继承项，不能只保存 YAML 路径。

在 v1.4 冻结前，先对两轨分别完成 batch 4 的真实 KITTI forward/loss/backward、显存和短恢复 smoke test：

- 两轨均稳定通过时，正式协议固定为 batch 4、LR 0.003；
- 任一轨无法稳定运行时，允许在正式结果产生前将两轨统一改为 batch 2；
- 改为 batch 2 时必须明确 LR 是否仍为 0.003、重新测量吞吐/显存/时间，并记录为 `PAPER_PROTOCOL_MISMATCH`；
- 正式结果产生后不得再改变 batch、LR 或 accumulation。

每个 epoch 的实际 `len(train_loader)`、是否 `drop_last`、总 optimizer step 数必须由冻结环境实测并写入预注册，不能只按样本数手工推算。

补充材料中关于学习率和 GPU 数量的描述与不同 OpenPCDet 版本可能存在差异。R8 不静默混合两套协议：具体 LR、batch、GPU 数量和 scheduler 数值必须在 v1.4 中冻结，并将与论文文字不一致的项目列入 `PAPER_PROTOCOL_MISMATCH` 清单。因此 R8 可以进行论文数值比较，但除非全部协议可核对一致，不宣称“训练设置完全相同”。

### 5.2 严格 FP32 与确定性设置

“完整 FP32”固定表示：

- AMP/autocast 关闭；
- GradScaler 禁用；
- `torch.backends.cuda.matmul.allow_tf32 = False`；
- `torch.backends.cudnn.allow_tf32 = False`；
- PyTorch float32 matmul precision 设置被明确记录；
- cuDNN benchmark/deterministic 设置被明确记录；
- `torch.use_deterministic_algorithms` 的开关和值被明确记录；
- 两轨使用完全相同的数值和确定性设置。

若实际执行决定允许 TF32，则任务名称和报告必须改为“FP32 storage / no AMP / TF32 enabled”，不得继续宣称严格 FP32。推荐沿用 R6 的 TF32 关闭设置。

### 5.3 正式 seed、运行数量与顺序

正式 seed 固定为：

- 666；
- 667；
- 668。

最小正式运行数量为 6：

```text
seed 666: PP_OFFICIAL_FULL + PH_PAPER_LITERAL_FULL
seed 667: PP_OFFICIAL_FULL + PH_PAPER_LITERAL_FULL
seed 668: PP_OFFICIAL_FULL + PH_PAPER_LITERAL_FULL
```

不得只挑选最好的 seed。所有有效正式运行都进入汇总。

seed 666 和 667 已在 R7 中被观察，只有 seed 668 是新 seed；R8 必须披露这一事实，不将三 seed 描述为完全盲法独立确认。

为减少固定执行顺序造成的热降频和系统状态偏差，正式顺序冻结为：

```text
seed 666: PP -> PH
seed 667: PH -> PP
seed 668: PP -> PH
```

若因调度原因不能按该顺序执行，必须在正式结果前冻结另一份交替顺序；不得根据已有精度或速度结果改变顺序。

### 5.4 成对初始化

每个 seed：

1. 创建一个 canonical backend initialization；
2. 将所有 shape-compatible 的 Scatter、Backbone、Neck 和 Head 参数显式复制到两轨；
3. 保存复制前后 checksum 和未复制参数清单；
4. 官方 PFN 与 PillarHist projection 等结构不兼容参数按冻结规则初始化；
5. 完成参数复制后才创建 optimizer 和 scheduler；AMP 固定关闭，checkpoint/manifest 明确记录 GradScaler 为 disabled；
6. 保存 canonical initialization 文件和 SHA-256。

不要求结构不兼容参数具有相同 checksum，但必须记录初始化方式和差异原因。

### 5.5 数据顺序、增强与完整 replay

两轨在同一 seed 下必须使用相同的：

- epoch/sampler 顺序；
- frame index 序列；
- augmentation seed；
- GT database sampling 结果或其 checksum；
- flip、rotation、scale 参数；
- batch 和 gradient accumulation 边界。

复用 R7 replay 机制并扩展为紧凑的全序列审计格式：

- 每个 seed 生成一份共享 replay，不为两轨分别生成不同计划；
- 使用压缩 `JSONL.GZ`，避免单个展开的大型 JSON；
- 保存每个 epoch 的 sampler permutation 或其可逆生成状态；
- 保存每个样本的 augmentation seed；
- 保存 GT database sampling 的确定性 seed、采样对象 ID 或其 checksum；
- 每个 optimizer step 都计算增强后 batch 的规范化输入 checksum；
- 每一步参与连续哈希链：`H_k = SHA256(H_{k-1} || step_id || input_checksum_k)`；
- 每个 epoch 保存 hash-chain root，完整训练保存 final root；
- 每个 epoch 的首批、末批和预注册随机抽样批次额外保存 tensor shape/dtype/逐张量 checksum；
- 同 seed 两轨的每个 epoch root 和 final root 必须一致。

哈希链保证全部步骤均参与审计，展开样本用于定位和人工检查。输入 checksum 必须覆盖增强后的 points、voxel/pillar 输入、coordinates、GT boxes/classes 和 batch 边界，而不只覆盖 frame ID。

workers 默认固定为 0。若 v1.4 最终选择多 worker，必须在正式训练前通过独立的完整重放测试，并冻结 worker seed、prefetch、persistent worker 和 sampler 行为；仅设置相同全局 seed 不足以构成公平性证据。

### 5.6 checkpoint、validation 与保留策略

主 checkpoint 固定为 `LAST_EPOCH`，即 epoch 80/冻结协议的最终 epoch。不得使用 best-validation checkpoint 替代主结果。

- 每 10 epochs 保存一个可恢复 checkpoint；
- 至少保留 epoch 20、40、60、80；
- 始终保留最近一个完整恢复点，直到下一个恢复点通过完整性验证；
- 只有 epoch 80 执行正式 3,769 帧 full-validation；
- 如需 epoch 40 中间验证，必须在 v1.4 中提前批准并固定，且仅作诊断；
- 中间 AP 不得用于提前停止、修改配置或选择主 checkpoint；
- 不生成隐式 best checkpoint，也不按中间 AP 删除其他轨道。

每个 checkpoint 至少包含：

- model；
- optimizer；
- scheduler；
- AMP 状态；本任务中应明确为 disabled；
- Python/NumPy/PyTorch/CUDA RNG state；
- sampler state、epoch、数据游标；
- completed optimizer step；
- resolved config、Git commit 和 replay manifest hash。

### 5.7 恢复连续性容差

每个正式轨道至少执行一次固定 next-batch 恢复测试，冻结判据为：

- next epoch/step、sampler cursor、输入 checksum：严格相等；
- optimizer step、scheduler step、LR：严格相等；
- loss 及各分量：`rtol=1e-5, atol=1e-6`；
- finite、正 anchor 数和代表梯度有效性一致；
- 在同硬件、同软件、同确定性设置下，恢复后一次 optimizer update 的模型参数优先要求原始字节 checksum 相等；
- 若已知底层算子无法逐字节确定，则必须在 v1.4 中预先列出算子和原因，并使用逐张量 `rtol=1e-5, atol=1e-6`、global norm 相对误差和 cosine similarity，不得结果产生后放宽。

v1.4 预注册还必须固定梯度审计的具体参数层、采样频率、`finite` 条件、梯度范数是否要求严格大于零，以及连续多少次不满足条件才判失败，避免使用未定义的“代表梯度有效”作为人工裁决空间。

## 6. 分阶段执行步骤

### Step 0：入口审计

- 核对 Git 工作树和入口 commit；
- 冻结 v1.4、预注册、配置和数据 manifest；
- 冻结人工选轨决策和 normalized 副轨是否执行；
- 运行全量 PillarHist pytest；
- 完成两轨 batch 4 真实 batch smoke test并冻结最终 batch；
- 实测 DataLoader 长度、steps/s 和显存；
- 生成正式资源预算、checkpoint 保留预算和散热条件记录。

未通过时不得启动正式训练。

### Step 1：seed 666 成对 admission

先完成 seed 666 的两条完整正式训练。该结果属于正式结果，不是可丢弃 pilot。

检查：

- 两轨均完成冻结的全部 epoch/step；
- loss 及分量全程 finite；
- 正 anchor 持续存在；
- PillarHist projection 和后端代表梯度有效；
- 数据 replay/checksum 一致；
- checkpoint 恢复连续性成立；
- full validation 产生非空预测和完整 AP_R40；
- baseline 没有出现零 AP、评测空输出或明显的数据/配置损坏。

低于论文 AP 本身不是 admission 失败。只有数据、训练链、checkpoint 或评测无效才阻塞扩展。

### Step 2：seed 667/668 成对训练

seed 666 admission 通过后，按冻结协议完成另外四个正式运行。不得根据 seed 666 的胜负修改超参数、训练长度、checkpoint 选择或主指标。

### Step 3：完整 validation 与效率记录

对六个主 checkpoint 运行相同 full-validation，并记录：

- 完整 KITTI 指标；
- validation latency 和总 wall time；
- 训练 step/s；
- peak allocated/reserved memory；
- checkpoint 大小；
- GPU 型号、驱动、温度/功耗状态；
- 所有原始评测文本和解析后的 JSON。

GPU 遥测默认使用冻结版本的 `nvidia-smi` 或等价工具每 30 秒采样一次，至少记录 timestamp、temperature、power draw、utilization、memory used 和 graphics clock。采样命令、工具版本和原始日志必须保存。

R8 的 latency/显存只作为当前硬件上的 FP32 工程记录，不替代 R6 benchmark，也不构成 TensorRT/部署结论。

### Step 4：三 seed 汇总

对每个指标计算：

```text
delta_seed = PH_seed - PP_seed
mean_PP = mean(PP_666, PP_667, PP_668)
mean_PH = mean(PH_666, PH_667, PH_668)
mean_delta = mean(delta_666, delta_667, delta_668)
std_delta = sample_std(delta_666, delta_667, delta_668)
```

必须保存逐 seed 原始值，不能只报告均值。三个 seed 不足以支撑强统计显著性声明，因此置信区间只作为描述性结果，不使用“证明显著优于”等措辞。

### Step 5：最终回归和归档

- 重新运行 `tests/pillarhist`；
- 随机抽取最终 checkpoint 做新进程加载和 inference；
- 验证所有 manifest/checksum 路径仍有效；
- 生成公平性审计、机器裁决和闭环报告；
- 只显式暂存 R8 相关代码、配置和文档；
- 形成独立 R8 Git commit；
- 大型 checkpoint/outputs 保持未跟踪，但必须备份并记录位置和 checksum。

## 7. 运行有效性与科学结论分离

### 7.1 执行状态

- `PASS_COMPLETE`：六个主轨运行全部有效，证据和汇总完整；
- `PASS_WITH_DOCUMENTED_RETRY`：存在允许的等价恢复/重试，但最终六轨有效且 amendment 完整；
- `PARTIAL`：部分 seed 有效，未满足六轨闭环；
- `BLOCKED`：达到停止条件且无法在冻结协议内继续。

只有前两种状态代表 R8 正式闭环。

### 7.2 科学结果标签

在执行状态通过后，机器按以下唯一规则生成科学标签。

对每个 seed 分别计算：

```text
Car_mAP_seed = mean(Car Easy, Car Moderate, Car Hard)
Ped_mAP_seed = mean(Ped Easy, Ped Moderate, Ped Hard)
delta_car_seed = PH_Car_mAP_seed - PP_Car_mAP_seed
delta_ped_seed = PH_Ped_mAP_seed - PP_Ped_mAP_seed
mean_delta_car = mean(delta_car_666, delta_car_667, delta_car_668)
mean_delta_ped = mean(delta_ped_666, delta_ped_667, delta_ped_668)
positive_car_seeds = count(delta_car_seed > 0)
positive_ped_seeds = count(delta_ped_seed > 0)
```

零值按非正处理。

机器标签使用评测器 JSON 中未四舍五入的原始数值计算，不使用报告表格中的两位小数。任一必需 AP 缺失、为 NaN/Inf 或解析失败时，不生成科学标签，执行状态按证据完整性进入 `PARTIAL` 或 `BLOCKED`。

- `DIRECTIONALLY_SUPPORTED`：`mean_delta_car > 0`、`mean_delta_ped > 0`、`positive_car_seeds >= 2` 且 `positive_ped_seeds >= 2`；
- `NOT_SUPPORTED`：`mean_delta_car <= 0`、`mean_delta_ped <= 0`、`positive_car_seeds <= 1` 且 `positive_ped_seeds <= 1`；
- `MIXED`：不满足以上两类的所有其他有效结果。

Car/Pedestrian 六个 Easy/Moderate/Hard paired delta 必须全部报告，但机器标签只使用上述两个类别三难度均值，避免“主要指标”或“总体为正”的人工解释空间。

`PAPER_PROTOCOL_MISMATCH` 不是精度科学标签，而是 manifest 中的客观布尔字段和差异列表。只要 GPU、batch、LR、OpenPCDet 版本、数据文件、训练方式或评测方式中任一项无法确认与论文一致，就设为 `true`。绝对 AP 差距只报告数值，不另设事后阈值。

示例：`PASS_COMPLETE + MIXED + PAPER_PROTOCOL_MISMATCH=true`。

不预设最低 AP 成功线，也不因结果不理想修改标签规则。

## 8. 论文比对报告要求

最终报告必须明确区分：

1. **本地可学习性：**完整训练是否稳定完成；
2. **本地因果收益：**相同 seed 下 PH 相对 PP 的成对变化；
3. **论文方向复现：**Car/Pedestrian 是否呈现论文所述正向趋势；
4. **论文绝对值复现：**本地绝对 AP 与论文 AP 的差距；
5. **协议差异：**GPU、batch、LR、OpenPCDet 版本、数据文件和 checkpoint 规则差异；
6. **尚未回答的问题：**INT8 保真度和部署速度留到 R9。

不得：

- 用三类别 Moderate 宏平均直接减论文的单类别三难度均值；
- 用 R7 5000-step AP 代替 R8 完整训练结果；
- 用最好 seed 代替三 seed 汇总；
- 将低 GFLOPs 直接表述为真实部署加速；
- 将 KITTI FP32 结果表述为已经验证论文的 nuScenes W8A8 结论。

## 9. 产物结构

```text
outputs/pillarhist/r8/
├─ preregistration/
│  ├─ preregistration_v1.4.json
│  ├─ track_admission_decision_v1.4.json
│  ├─ document_hashes.json
│  ├─ data_manifest.json
│  └─ protocol_differences.json
├─ seed666/
│  ├─ canonical_initialization.pth
│  ├─ replay_manifest.json
│  ├─ replay_steps.jsonl.gz
│  ├─ replay_hash_chain.json
│  ├─ PP_OFFICIAL_FULL/
│  └─ PH_PAPER_LITERAL_FULL/
├─ seed667/
│  ├─ canonical_initialization.pth
│  ├─ replay_manifest.json
│  ├─ replay_steps.jsonl.gz
│  ├─ replay_hash_chain.json
│  ├─ PP_OFFICIAL_FULL/
│  └─ PH_PAPER_LITERAL_FULL/
├─ seed668/
│  ├─ canonical_initialization.pth
│  ├─ replay_manifest.json
│  ├─ replay_steps.jsonl.gz
│  ├─ replay_hash_chain.json
│  ├─ PP_OFFICIAL_FULL/
│  └─ PH_PAPER_LITERAL_FULL/
├─ R8_fairness_audit.json
├─ R8_metrics_by_seed.json
├─ R8_paper_comparison.json
├─ R8_efficiency.json
├─ R8_decision.json
└─ tests/
   ├─ test_report.txt
   └─ junit.xml
```

每个正式轨道目录至少包含：

- `manifest.json`；
- `config_resolved.yaml`；
- `environment.txt`；
- `initialization_checksums.json`；
- `training_log.jsonl`；
- `gradient_audit.json`；
- `checkpoints/last.pth`；
- `checkpoint_manifest.json`；
- `validation/full/result.txt`；
- `validation/full/metrics.json`；
- `runtime_and_memory.json`；
- `gpu_telemetry.csv`；
- `commands/`。

仓库文档目录生成：

```text
docs/pillarhist/R8_closure_report.md
docs/pillarhist/PillarHist_OpenPCDet_deployment_guideline_v1.4.md
docs/pillarhist/PillarHist_R8_preregistration_v1.4.json
docs/pillarhist/PillarHist_R8_track_admission_decision_v1.4.json
```

## 10. 重试、修改与停止条件

### 10.1 允许的恢复

以下情况允许从最近的完整 checkpoint 恢复，不视为新实验：

- 断电、系统重启或终端中断；
- CUDA OOM 的偶发外部占用，且 batch/配置不变；
- GPU 驱动暂时错误；
- 磁盘空间或日志写入的临时故障；
- validation 单独失败但训练 checkpoint 完整。

恢复前后必须验证 checkpoint、optimizer、scheduler、RNG、sampler 和 next-step 连续性。

同一瞬时故障最多允许两次恢复尝试。三次连续出现相同故障且无法在不改变冻结协议的情况下解决时，停止该轨道并标记 `BLOCKED`。

### 10.2 允许的 amendment

只允许修复不改变以下内容的记录器或运行器错误：

- 模型数学语义；
- 数据和增强结果；
- 初始化；
- optimizer/scheduler；
- epoch/step 数；
- checkpoint 选择；
- 评测指标和裁决规则。

必须保留失败尝试、修改前后源码 hash、错误原因和影响分析。若修改可能影响梯度、输入、训练轨迹或模型输出，则原受影响运行作废，并发布新的预注册版本后成对重跑。

### 10.3 不因偶然误差过早停止

以下情况不自动停止整个 R8：

- 单次 loss spike；
- 单次 validation 波动；
- 某个 seed 的 AP 低于论文；
- 某一类别没有提升；
- 一次可恢复的 CUDA/IO/进程错误；
- 训练速度低于估算；
- R8 结果与 R7 排名不一致。

### 10.4 必须暂停评审的情况

- 两轨数据顺序或增强 checksum 不一致；
- 出现持续 NaN/Inf 且完整恢复后重复出现；
- 正 anchor 长时间为零或评测输出为空；
- checkpoint 无法恢复 optimizer/scheduler/RNG/sampler；
- Git/配置/数据身份无法追溯；
- 需要改变 batch、LR、epoch、模型结构或主指标才能继续；
- 同一阻塞原因连续三次发生；
- 计算资源不足以完成剩余成对运行。

暂停后不得用已有最好结果宣称 R8 完成。

## 11. 资源预算提示

R7 的 `3.0–3.17 optimizer steps/s` 来自 batch 2，不能直接作为 R8 batch 4 的正式预算。R8 资源预算必须在 batch 4/最终 batch smoke test 后按下式重新计算：

```text
base_hours_per_run = epochs * measured_steps_per_epoch / measured_steps_per_second / 3600
budget_hours_per_run = base_hours_per_run * 1.20
total_gpu_hours = budget_hours_per_run * 6 + full_validation_budget
```

`1.20` 为恢复、日志和轻微热降频余量，不是停止线。预注册必须保存实测 batch、steps/epoch、steps/s、单轨预算、六轨预算和 full-validation 预算。原 batch 2 的 80–100 GPU 小时估算仅作历史参考，不再作为正式数字。

开工前应确认：

- 连续运行和散热条件；
- checkpoint 与日志磁盘空间；
- outputs 的备份位置；
- 是否串行运行，避免不同轨道争抢 GPU；
- 电源/休眠策略；
- 正式运行期间不同时执行其他高负载 GPU 任务。

该估算不作为超时停止线。若实际速度较慢但训练有效，可继续或从完整 checkpoint 恢复。

## 12. R8 完成清单

### 入口

- [ ] v1.4 指导文档和 R8 预注册已在正式结果前冻结
- [ ] 人工选轨决策文件已冻结，normalized 副轨已在结果前明确批准或拒绝
- [ ] Git 起点、配置、脚本和数据 manifest 已固定
- [ ] 当前未提交修改已分类处理且未覆盖用户文件
- [ ] R6/R7 回归、真实 batch 和 checkpoint smoke test 通过
- [ ] 计算、存储、散热和备份条件已确认
- [ ] 单卡、batch、global batch、accumulation、AMP/TF32、workers、LR 和 steps/epoch 已冻结
- [ ] `LAST_EPOCH`、checkpoint 保留和 validation 频率已冻结

### 正式训练

- [ ] seed 666 两轨 admission 通过
- [ ] seed 667 两轨完整训练通过
- [ ] seed 668 两轨完整训练通过
- [ ] 六轨均达到冻结的完整 epoch/step
- [ ] 三个 seed 内成对数据 replay/checksum 审计通过
- [ ] 每个 epoch replay hash-chain root 和最终 root 成对一致
- [ ] loss、正 anchor、梯度和 checkpoint 恢复有效

### 评测与报告

- [ ] 六个主 checkpoint 完成 3,769 帧 full-validation
- [ ] Car/Pedestrian/Cyclist AP_R40 Easy/Moderate/Hard 完整
- [ ] 三 seed mean/std 和 paired delta 已生成
- [ ] 与论文绝对值、相对提升和协议差异已分别报告
- [ ] R8 执行状态和科学标签已生成
- [ ] 最终 pytest/JUnit、机器裁决和公平性审计已保存
- [ ] R8 相关代码和文档形成独立 Git commit
- [ ] 大型 outputs/checkpoint 已备份并保存 checksum

## 13. R8 的正式终点

当以下条件同时满足时，R8 闭环结束：

1. 两个正式主轨、三个 seed 共六个完整训练均有效；
2. full KITTI validation 和论文同口径比较完成；
3. 公平性、可恢复性、测试和产物审计通过；
4. 形成 `PASS_COMPLETE` 或 `PASS_WITH_DOCUMENTED_RETRY`；
5. 按确定性规则给出 `DIRECTIONALLY_SUPPORTED`、`MIXED` 或 `NOT_SUPPORTED`，并报告 `PAPER_PROTOCOL_MISMATCH`；
6. 独立 Git commit 和 outputs 备份完成。

到达该终点后暂停评审，不自动进入 R9。R9 必须另行冻结量化范围、校准集、首尾层/PFE 保留精度、TensorRT 版本、目标硬件和延迟测量协议。
