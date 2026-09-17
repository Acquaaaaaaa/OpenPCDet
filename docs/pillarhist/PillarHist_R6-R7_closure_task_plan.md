# PillarHist R6–R7 闭环任务书

> 前置状态：R0–R5 已完成  
> 基线提交：`e90de060bceba2a228095eea01718b4ded55b68b`  
> 基线验收运行：`20260917T012049Z_seed666`  
> 当前范围：PH_OPT 正确性/性能与训练结构裁决  
> 终点：R7 完成后暂停评审，不自动进入 R8/R9

## 1. 总目标

本轮只完成两件事：

1. **R6：**实现一个与 `PH_PAPER_LITERAL_REF` 数学语义对齐的 `PH_OPT`，正式关闭 TST-018，并客观判断其是否具有局部或端到端性能价值；
2. **R7：**用统一短程训练协议裁决坐标模式和 projection 结构；若证据充分则冻结 R8 唯一主配置，否则形成 `INCONCLUSIVE` 或 `NO_PROMOTION` 结论。

本轮不进行 KITTI 完整训练、三种子正式 AP、PTQ、TensorRT 或目标硬件部署，不作相关结论。

## 2. 入口与不变量

### 2.1 冻结入口

R6–R7 必须从上述 R0–R5 提交开始，并保留以下 reference：

- 固定 validation fixture：`000001`、`000002`；
- 固定 training fixture：`000000`、`000003`；
- `PH_PAPER_LITERAL_REF` 的 tensor contract 和 checksum；
- PointPillar 后端通道：cls 18、box 42、dir 12；
- `PillarHistVFE` 输出：`[V,64]`；
- checksum、随机性和环境记录方法沿用 R0–R5。

**硬入口门禁：**在任何正式 R6 profile、性能测量或 R7 训练开始前，必须发布并冻结新的指导文档 minor 版本（v1.3）。

v1.3 必须在看到正式结果前冻结：

- R6 loss/gradient 对齐容差；
- correctness 与 performance 的运行模式；
- benchmark 方法和性能晋升门禁；
- R7 训练步数、validation subset frame ID；
- R7 主指标、tie margin 和 checkpoint 选择规则；
- 数据重放、初始化和恢复协议。

在 v1.3 发布前只允许进行标记为 `EXPLORATORY` 的工具验证或瓶颈探查；其结果不得进入正式 R6/R7 报告，也不得用于反向修改门禁。

### 2.2 不得改变的数学语义

PH_OPT 不得改变：

- active-pillar 准入集合及其行序；
- all-points-in-admitted-pillars 口径；
- Z/XY 半开边界；
- 64 个高度 bin；
- count 与 mean-intensity 定义；
- batch 隔离；
- `V=0/1`、dtype、AMP 和坐标契约；
- projection 输入布局及 `[V,64]` 输出。

改变上述任一项都属于新算法，而不是 R6 优化实现。

## 3. R6：PH_OPT

### Step 1：Reference profile

在写优化代码前，对 `deterministic_segment` reference 做 profile：

- histogram key/lookup；
- count reduction；
- intensity sum/mean；
- concat/center；
- projection；
- VFE 总时间；
- 完整 PointPillar inference 时间；
- 峰值显存。

至少覆盖：

- 小型合成边界输入；
- 高点数/overflow 输入；
- 固定真实 validation fixture；
- 固定真实 training fixture。

产物必须能说明主要瓶颈位于哪里。没有 profile 证据时，不进入自定义 CUDA。

### Step 2：选择优化路径

按以下优先级尝试：

1. 基于 compact active-pillar row 的 `scatter_add`；
2. 内存可控的 `bincount`；
3. 其他 PyTorch 原生实现；
4. 只有原生实现无法达到预设性能门禁时，才单独评审是否编写 CUDA kernel。

本轮最多保留一个 PH_OPT 主实现。不同候选不得同时进入 R7。

配置中应明确区分：

```text
REDUCTION_MODE: deterministic_segment   # PH_REF
REDUCTION_MODE: <selected_optimized>    # PH_OPT
```

### Step 3：关闭 TST-018

REF/OPT 对齐必须覆盖：

| 对象 | 通过标准 |
|---|---|
| active-pillar row/order | 精确一致 |
| `Hp` count histogram | 精确一致 |
| `Hi` mean-intensity | `rtol=1e-5, atol=1e-6` |
| concat/projection input | `rtol=1e-5, atol=1e-6` |
| `pillar_features` | `rtol=1e-5, atol=1e-6` |
| Scatter/Backbone/Head pre-NMS | 满足既有 tensor contract 容差 |
| loss | finite；`rtol=1e-5, atol=1e-6` |
| projection gradient | 逐参数比较；`rtol=1e-4, atol=1e-6` |
| backend gradient | 代表参数逐张量比较，并比较整体 norm；`rtol=1e-4, atol=1e-6` |

梯度还必须满足：

- 两侧 norm 均大于 `1e-8` 时，cosine similarity 不低于 `0.9999`；
- 两侧绝对值均低于 `1e-8` 的元素按近零处理，只检查绝对误差，不计算无意义的相对误差；
- projection 至少检查 weight 及实际存在的 bias；
- backend 至少检查 Backbone 一组参数和 Head 一组参数；
- 同时报告逐张量误差、global gradient norm 和 norm 相对差异。

上述数值作为 v1.3 默认值；若需调整，必须在正式 PH_OPT 结果产生前，以 FP32 reference 噪声分析为依据完成。

测试输入至少包含：

- `V=0`、`V=1`；
- 空 bin、半开边界、40+ points/pillar；
- overflow/admission cap；
- 打乱 points/coords；
- 中间空 batch；
- 最大合法 packed key；
- 固定真实 validation/training fixture；
- FP32 和外层 AMP。

若 PH_OPT 使用 atomic 或非固定归约顺序，必须拆分两种运行模式：

**Correctness/repeatability mode：**

- REF/OPT 必须从同一份完整 model state 构建，projection/backend 参数、train/eval 模式和输入完全一致；
- `Hp` 仍必须精确；
- `Hi`/features 使用规定容差；
- 同一输入至少重复 20 次，报告最大误差和输出分布；
- 若路径包含 atomic 或其他非固定归约，loss 和代表梯度也至少重复 20 次；每次 backward 前清空梯度并恢复相同模型/输入状态，报告最坏误差、gradient norm 分布和最低 cosine similarity；
- 明确记录 `torch.use_deterministic_algorithms`、cuDNN、TF32 和 CUBLAS 设置；
- 不得把非确定性隐藏为 checksum 失败。

**Performance/deployment-like mode：**

- REF/OPT 使用同一套预先冻结的 deterministic/TF32/AMP 设置；
- 若 deterministic algorithms 会拒绝 atomic，则性能模式可统一关闭该限制，但不能只对 OPT 关闭；
- correctness 结论和 performance 结论分轨记录，不能用性能模式替代正确性检查。

TST-018 只有在上述比较全部通过后才能标记为 `PASS`。

### Step 4：性能基准

REF 与 OPT 必须使用：

- 相同代码提交、GPU、驱动、PyTorch/CUDA；
- 相同 batch、dtype、AMP 和 deterministic 设置；
- 相同输入 fixture；
- 使用 CUDA Event 计时，并在每轮前后显式 synchronize；
- 独立 warmup 和测量阶段；
- 无 DataLoader/I/O 干扰的 microbenchmark，以及完整模型 benchmark。

默认最小测量协议：

- warmup：30 次；
- VFE-only：至少 200 次；
- end-to-end：至少 100 次；
- 独立重复：5 轮；
- 报告 mean、p50、p95、标准差；
- REF/OPT 在每轮中交替或按预先生成的顺序随机化测量，降低热漂移；
- 保存每轮原始测量样本，而不只保存聚合统计；
- 报告 samples/s、points/s、pillars/s；
- 同时报告 peak allocated 和 peak reserved memory；
- 记录 GPU 时钟、温度、功耗模式和是否存在其他 GPU 负载；
- 报告加速比、轮间变异系数，并优先给出 paired bootstrap 95% 置信区间。

### Step 5：R6 性能门禁

正确性通过是绝对门禁。PH_OPT 是否晋升到 R7，再按预先冻结的性能门禁判断。

建议默认门禁：

- VFE-only p50 至少提升 10%；
- 5 轮中至少 4 轮保持正收益；
- end-to-end paired 加速的 95% 置信区间下界大于 0；若不计算置信区间，则至少要求 5 轮中 4 轮为正收益，且收益大于 reference 的轮间噪声；
- 峰值显存不增加超过 10%，或增加部分有明确收益解释。

项目可以在 v1.3 中调整数值，但必须在看到优化结果前冻结。

R6 可能得到四类结论：

- `PASS_PROMOTED`：TST-018 通过且端到端收益超过测量噪声，PH_OPT 进入 R7；
- `PASS_LOCAL_ONLY`：TST-018 通过且 VFE 局部收益明确，但端到端收益处于噪声范围；不得宣称部署加速，R7 默认使用 PH_REF；
- `PASS_NOT_PROMOTED`：TST-018 通过，但端到端稳定退化或总体收益不足，R7 统一使用 PH_REF；
- `BLOCKED/FAIL`：正确性不满足或环境阻塞，不进入 R7 训练。

不得把“数值对齐但没有加速”或“只有 VFE 局部加速”报告为端到端部署加速成果。

### Step 6：R6 产物

```text
outputs/pillarhist/r6/<implementation>/<timestamp>_<seed>/
├─ config_resolved.yaml
├─ manifest.json
├─ environment.txt
├─ data_manifest.json
├─ input_checksum.json
├─ source_commits.json
├─ correctness/
│  ├─ tst018_report.json
│  ├─ tensor_comparison.json
│  ├─ repeatability.json
│  └─ gradient_comparison.json
├─ profile/
│  └─ reference_profile.json
├─ benchmark/
│  ├─ vfe_latency.json
│  ├─ end_to_end_latency.json
│  ├─ throughput.json
│  ├─ memory.json
│  ├─ raw_samples.json
│  └─ gpu_state.json
├─ commands/
│  ├─ profile_command.txt
│  └─ benchmark_command.txt
└─ tests/
   ├─ test_report.txt
   └─ junit.xml
```

R6 完成后形成独立 Git commit，不与 R7 训练配置混合。

`manifest.json` 还必须记录 REF/OPT 代码提交、profiler/benchmark 工具及版本、correctness/performance 模式和完整命令行。

## 4. R7：训练与结构裁决

### Step 1：冻结四条实验轨道

| Track | 坐标 | Projection |
|---|---|---|
| `PH_RAW_LINEAR` | `raw_meter_xy` | Linear |
| `PH_NORM_LINEAR` | `normalized_xy` | Linear |
| `PH_RAW_BNRELU` | `raw_meter_xy` | Linear + BN + ReLU |
| `PH_NORM_BNRELU` | `normalized_xy` | Linear + BN + ReLU |

另增加一个非候选校准轨道：

| Track | 结构 | 用途 |
|---|---|---|
| `PP_SHORT_CONTROL` | 原始 PointPillar/PillarVFE | 判断短程训练协议是否本身异常，不参与四种 PillarHist 结构排名 |

四个 PillarHist 候选轨道必须使用相同的 reduction path；`PP_SHORT_CONTROL` 固定使用原始 `PillarVFE`，不参与该约束：

- R6 为 `PASS_PROMOTED`：统一使用选定 PH_OPT；
- R6 为 `PASS_LOCAL_ONLY` 或 `PASS_NOT_PROMOTED`：统一使用 PH_REF；
- 不允许四个轨道混用 REF/OPT。

### Step 2：成对初始化

建立一个 canonical master initialization。不得依赖“相同 seed 连续构建四次”，因为 RNG 状态会持续前进。

1. 从固定 seed 只构建一次 canonical backend master；
2. 将所有 shape-compatible 的 Scatter、Backbone 和 Head 权重显式复制到五个轨道；
3. 单独生成一份 canonical projection Linear weight，并复制到四个 PillarHist 轨道的兼容 Linear 层；
4. raw/normalized 且 projection 类型相同时，projection 的全部兼容参数必须相同；
5. v1.3 明确冻结 Linear-only 是否带 bias、BN-ReLU 中 Linear 是否带 bias；
6. Linear-only bias 与 BN-ReLU 不兼容时，不伪造复制关系，而是按冻结规则初始化并记录；
7. 固定 BN `weight=1`、`bias=0`、running mean=0、running var=1、momentum 和更新行为；
8. 保存复制前后 checksum 和未复制参数清单；
9. 全部权重复制完成后才创建 optimizer、scheduler 和 AMP GradScaler。

不要求结构不兼容的参数拥有相同 checksum，但其初始化规则、RNG 状态和差异原因必须被记录。

### Step 3：冻结短程训练协议

v1.3 在运行前至少固定：

- KITTI split 和 info PKL；
- 单卡/多卡和 GPU 型号；
- per-GPU/global batch；
- BN 或 SyncBN 行为；
- seed、shuffle、worker seed；
- GT sampling 和全部数据增强顺序；
- optimizer、weight decay、基础学习率；
- scheduler、warmup；
- AMP、梯度裁剪；
- 固定训练步数；
- validation 子集和评测频率；
- last/peak checkpoint 选择规则；
- 异常重试和提前终止条件。

此外必须预先生成可重放的 `training_replay_manifest`，逐 optimizer step 记录：

- 数据索引和顺序；
- sampler epoch/cursor；
- 每个样本的 augmentation seed；
- GT database 采样对象或其 checksum；
- 翻转、旋转、缩放等增强参数或结果 checksum；
- gradient accumulation 的 micro-batch 边界；
- optimizer step 编号。

四个 PillarHist 轨道及 `PP_SHORT_CONTROL` 必须消费同一训练计划。建议对开头、结尾和随机抽取的 step 比较增强后输入 checksum，证明数据重放一致，而不是只比较 seed。

建议默认使用：

- 单一主 seed `666`；
- 每轨固定 5,000 optimizer steps；
- 每 1,000 steps 在固定 validation subset 上检查；
- 最终 step 在完整 validation split 上做一次初步评测；
- last checkpoint 为 primary，peak checkpoint 只作 secondary；
- 不为单个轨道单独调学习率、增强或 scheduler。

这是受控短程筛选协议，不得称为论文训练协议或正式 KITTI 训练。

训练 checkpoint 必须包含：

- model；
- optimizer；
- scheduler；
- AMP GradScaler；
- Python/NumPy/PyTorch CPU/CUDA RNG state；
- sampler state、数据游标和当前 optimizer step；
- gradient accumulation 状态（若使用）。

只有这些状态完整恢复后，才能将中断前后视为同一条训练序列。

### Step 4：训练过程检查

四个轨道统一记录：

- 总 loss 与 cls/loc/dir 分量；
- projection、Backbone、Head 梯度 norm；
- NaN/Inf、梯度爆炸和异常 step；
- BN running statistics；
- 吞吐、显存和 wall time；
- 固定 step checkpoint；
- 初步 validation 指标。
- 训练计划位置和抽查 step 的输入 checksum；

任何轨道因偶发环境错误中断，可从同一 checkpoint 重试；不得改变该轨道超参数后继续参与公平比较。

### Step 5：预先冻结裁决规则

R8 主配置按以下顺序裁决：

1. **有效性门禁：**无 NaN/Inf、梯度链完整、checkpoint 可恢复；
2. **主要指标：**v1.3 预先定义的 validation 主指标；
3. **稳定性：**loss/梯度和不同检查点趋势；
4. **效率：**训练/推理 latency、吞吐和显存；
5. **简洁性：**结果接近时优先更简单、解释更明确的结构。

主指标明确为：**KITTI validation 上 Car、Pedestrian、Cyclist 的 3D AP_R40 Moderate 宏平均**，使用冻结的 OpenPCDet 评测脚本和 KITTI 标准类别 IoU 阈值。

同时必须冻结：

- 每 1,000 steps 使用的 validation subset frame ID；
- subset 也使用 3D AP_R40 Moderate，还是仅使用固定 loss/recall；
- `peak_secondary` 根据 subset 还是完整 validation 选择；
- 最终 step 对完整 validation split 的评测命令；
- tie margin 的单位为绝对 AP point。

该指标只用于短程候选排序，不是最低精度门禁。训练 loss 不能单独决定胜者，也不要求短程 PillarHist 超过完整训练 80 epoch 的 baseline。

若第一、第二名主指标差距小于预先冻结的 tie margin（默认 1.0 个绝对 AP point），只对并列候选增加第二 seed 复核；不得自动把四条轨道全部扩展为多 seed。tie margin 是不确定性区间，不是“必须提升 1 AP”的成功线。

`PP_SHORT_CONTROL` 只用于判断 5,000-step 协议是否异常，不得与完整训练 baseline 直接比较或参与 PillarHist 四轨排名。

v1.3 必须在训练前冻结 `PP_SHORT_CONTROL` 的协议异常判据，至少包括：loss 及分量全程 finite、正 anchor 持续存在、梯度链完整、loss 相对初始化阶段具有可解释趋势、完整 validation 的 AP/recall 未退化到无效状态、checkpoint 恢复连续性成立。具体阈值须在正式结果产生前固定；不得看到五轨结果后再定义“异常”。

R7 最终状态为：

- `SELECTED`：存在稳定且可区分的唯一 R8 主配置；
- `INCONCLUSIVE`：候选差异处于不确定范围，短程实验不足以裁决；
- `NO_PROMOTION`：候选均不稳定、短程协议表现异常，或没有足够理由进入完整训练；
- `BLOCKED`：公平性、环境或数据重放失败。

`SELECTED/INCONCLUSIVE/NO_PROMOTION` 都可以作为一次有效的 R7 实验结论。后两者必须暂停评审，不自动启动完整训练。

### Step 6：R7 产物

```text
outputs/pillarhist/r7/<track>/<run_id>/
├─ config_resolved.yaml
├─ manifest.json
├─ initialization_checksums.json
├─ data_order_manifest.json
├─ training_replay_manifest.json
├─ environment.txt
├─ metrics/
│  ├─ train_curve.json
│  ├─ gradient_stats.json
│  ├─ validation_summary.json
│  └─ efficiency.json
├─ checkpoints/
│  ├─ last.pth
│  └─ peak_secondary.pth
├─ resume_state_report.json
└─ logs/
```

另生成：

```text
outputs/pillarhist/r7/R7_decision_report.md
```

决策报告必须包含四轨公平性校验、`PP_SHORT_CONTROL` 校准结果、结果表、最终状态，以及不能从短程实验推出的结论。

## 5. 停止与重试规则

### R6

- 偶发 CUDA/环境错误允许清理后重试 3 次；
- 同一优化实现允许 3 轮针对性修复；有新证据或误差持续缩小时可增加 1 轮最终验证；
- 最多尝试两种 PyTorch 原生优化路径；
- 是否进入自定义 CUDA 必须单独评审，不能因原生实现收益不足自动扩展；
- 正确性未通过时禁止发布性能结论。

### R7

- 基础设施错误允许从相同 checkpoint 恢复；
- 每个轨道最多因实现问题重新开始 1 次；
- 不允许通过修改单轨超参数挽救结果；
- 四轨协议无法保持一致时，状态为 `BLOCKED`；
- 结果无法超过 tie margin 时，只扩展并列候选；仍无法裁决则标记 `INCONCLUSIVE`；不得通过改变单轨超参数强行产生胜者。

出现以下情况立即暂停评审：

- 需要改变 PillarHist 数学语义；
- 需要重做 R0–R5 reference；
- 需要进入完整训练才能解释当前异常；
- 需要 PTQ/TensorRT 才能继续；
- 需要未经授权的环境、数据或破坏性操作。

## 6. 完成标准

### R6 完成

- [ ] v1.3 已在正式 R6 profile/benchmark 前发布并冻结
- [ ] Reference profile 已完成
- [ ] PH_OPT 主实现唯一且配置可切换
- [ ] TST-018 已正式 PASS
- [ ] REF/OPT 全链路与梯度对齐
- [ ] latency、吞吐和显存报告完整
- [ ] PH_OPT 获得 `PASS_PROMOTED`、`PASS_LOCAL_ONLY` 或 `PASS_NOT_PROMOTED`
- [ ] R6 独立 Git commit 已生成

### R7 完成

- [ ] v1.3 已在正式 R6 profile/benchmark 前发布并冻结
- [ ] loss/gradient 容差及 correctness/performance 模式已冻结
- [ ] 四轨使用相同数据顺序、后端初始化和训练协议
- [ ] `PP_SHORT_CONTROL` 已按相同短程协议运行
- [ ] training replay 抽查证明增强后输入一致
- [ ] optimizer 在权重复制之后创建
- [ ] 四轨短程训练和初步 validation 完成
- [ ] 公平性、训练稳定性和效率记录完整
- [ ] `SELECTED`、`INCONCLUSIVE` 或 `NO_PROMOTION` 结论已形成
- [ ] R7 决策报告和独立 Git commit 已生成

## 7. 本轮终点

R7 完成后必须暂停并进行人工评审。

本轮不得宣称：

- 已达到论文或官方 KITTI AP；
- 短程实验结果等同于完整训练；
- 单 seed 足以证明统计优势；
- fake INT8、真实 INT8 或 TensorRT 已验证；
- 未经 TST-018 和性能门禁的实现具有部署加速价值。

只有人工接受 R7 决策报告后，才另行制定 R8 正式训练任务书。
