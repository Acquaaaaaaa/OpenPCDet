# PillarHist × OpenPCDet 复刻与部署指导规范 v1.4

> 生效日期：2026-09-18  
> 适用工程：`/home/ksyang/OpenPCDet`  
> R7 闭环提交：`b448047eb8ee85ae84e83b4a5ed6b71587894bc2`（分支 `pointpillars-pillarhist`）  
> 当前目标：在不改变 R0–R7 已冻结语义的前提下完成 R8 三种子、双轨、80 epoch 严格 FP32 正式训练与完整 KITTI validation。  
> 本文件是 R8 实现、验证和验收的规范入口；详细执行顺序与产物以 `PillarHist_R8_closure_task_plan.md` 和预注册 JSON 为准。v1.0–v1.3 只用于追溯。

## 1. 规范使用规则

### 1.1 条款状态与证据标签

| 状态 | 含义 | 执行要求 |
|---|---|---|
| `[已冻结]` | 已有充分依据或已作工程决策 | 当前实现必须遵守；改变时升级版本 |
| `[临时默认]` | 论文缺失，但需要一个可运行起点 | 当前按此实现；不得称为作者真实设置 |
| `[对照项]` | 用于验证假设或工程优化 | 独立配置和命名，不进入主轨默认值 |
| `[开放项]` | 当前不应定案 | 不阻塞当前结构与数据流阶段 |

所有关键结论继续使用证据标签：`[原文明示]`、`[公式推导]`、`[合理推断]`、`[待验证假设]`。条款状态描述“现在怎么做”，证据标签描述“为什么这样做”。

测试门禁分为两类：[已冻结][合理推断]

- `blocking`：当前阶段必须通过；失败即不得宣称该阶段完成。
- `diagnostic/expected-failure`：用于记录固定上游模块的已知限制；允许受控 `xfail`，但必须在 test report 和 manifest 中写明触发条件、实际异常及对应代码依据。

除非测试条款显式标为 `diagnostic/expected-failure`，所有 TST 默认均为 `blocking`。

### 1.2 冲突优先级

发生冲突时依次采用：

1. 本文件带 ID 的规范条款；
2. 本文件的公式、伪代码和测试门禁；
3. 固定提交下的本地 OpenPCDet 代码与配置；
4. PillarHist 正文与补充材料；
5. v1.0、评审补充和审计依据版。

所有偏离必须进入配置、实验名和 manifest，不得在运行后追溯修改其解释。[已冻结][合理推断]

## 2. 当前范围与阶段门禁

### 2.1 当前完成目标

R0–R5 已由提交 `e90de060bceba2a228095eea01718b4ded55b68b` 关闭，R6/R7 已分别由 `c7afac9` 与 `b448047` 闭环。R7 结论为 `INCONCLUSIVE`；当前阶段完成的定义是：[已冻结][合理推断]

> `PP_OFFICIAL_FULL` 与 `PH_PAPER_LITERAL_FULL` 在三个 seed 下完成成对 80 epoch 训练、完整 validation、公平性与恢复审计，并按预注册规则输出执行状态、科学标签和论文协议差异。

当前阶段不要求：[已冻结][合理推断]

- 完成 PTQ、TensorRT 或目标硬件部署。
- 达到论文 AP 或保证 PillarHist 必须优于本地 PointPillars；
- KITTI test server 提交、nuScenes/Waymo 完整复现或统计显著性声明。

### 2.2 实施阶段

| 阶段 | 目标 | 完成条件 |
|---|---|---|
| R0 | 冻结官方 baseline | 固定 batch 的输入、中间 shape/statistics、最终预测 checksum 已保存 |
| R1 | CPU histogram oracle | 点数、平均强度、边界、all-points 和空 bin 测试通过 |
| R2 | GPU deterministic reference | 与 CPU oracle 数值对齐；行序、batch 隔离和 dtype 正确 |
| R3 | `PillarHistVFE` 接入 | registry、YAML、build、forward、输出 `[V,64]` 全部正确 |
| R4 | 端到端数据流 | Scatter/Backbone/Head shape 不变；预测无 NaN/Inf |
| R5 | 最小可微与可恢复性 | loss forward、backward、梯度和 checkpoint round-trip 通过；不要求收敛 |
| R6 | optimized path | TST-018、全链路/梯度对齐与冻结性能门禁完成，输出 PROMOTED/LOCAL_ONLY/NOT_PROMOTED |
| R7 | 训练准备与结构假设裁决 | 五轨公平短程实验完成，输出 SELECTED/INCONCLUSIVE/NO_PROMOTION |
| R8 | 正式训练与评价 | KITTI 完整训练、三种子、AP 和时延报告 |
| R9 | PTQ 与目标部署 | nuScenes fake W8A8、真 INT8、TensorRT/目标硬件 |

R7 已完成人工评审。R8 主轨选择 `raw_meter_xy + Linear` 是论文忠实性优先的人工决策，不表示 R7 已证明 raw 轨优于 normalized 轨；R8 完成后暂停评审，不自动进入 R9。[已冻结][合理推断]

## 3. 环境、数据与不变量

| ID | 状态与证据 | 规范值 |
|---|---|---|
| ENV-001 | [已冻结][原文明示] 本地检查 | WSL Ubuntu 22.04；工程 `/home/ksyang/OpenPCDet` |
| ENV-002 | [已冻结][原文明示] 本地检查 | R7 闭环 Git HEAD `b448047eb8ee85ae84e83b4a5ed6b71587894bc2`；R8 入口实现提交与冻结提交由预注册/运行 manifest 记录；分支 `pointpillars-pillarhist` |
| ENV-003 | [已冻结][原文明示] 本地检查 | Python 3.10.21；PyTorch 2.9.1+cu128；spconv 2.3.8 |
| ENV-004 | [已冻结][合理推断] | 运行时记录 GPU、显存、driver、CUDA runtime、cuDNN、编译器、自定义 op、TF32 和 deterministic 状态 |
| ENV-005 | [已冻结][原文明示] 本地检查 | 既有 dirty 修改属于用户；不得覆盖、回滚或混入 PillarHist 差异 |
| DATA-001 | [已冻结][原文明示] 本地配置 | KITTI classes：Car、Pedestrian、Cyclist |
| DATA-002 | [已冻结][原文明示] 本地代码 | batch 后 `points [P,5]=(batch_id,x,y,z,intensity)`；`voxel_coords [V,4]=(batch_id,z,y,x)` |
| DATA-003 | [已冻结][原文明示][合理推断] | ROI 为 `[0,69.12)×[-39.68,39.68)×[-3,1)` m；所有上界采用半开区间 |
| DATA-004 | [已冻结][原文明示][公式推导] | pillar size `(0.16,0.16,4)` m；grid `(N_x,N_y,N_z)=(432,496,1)` |
| DATA-005 | [已冻结][原文明示] | active pillars：train 16000、test 40000；hard voxelizer max points 32 |
| DATA-006 | [已冻结][合理推断] | 运行时记录 info 条目数、info SHA-256、点云根目录和 GT database；不把论文自相矛盾的样本数硬编码为本地事实 |
| INV-001 | [已冻结][合理推断] | 当前只替换 VFE；Scatter、BEV Backbone、AnchorHeadSingle、anchors、target assigner、loss、decode、NMS 均保持固定提交行为 |
| INV-002 | [已冻结][合理推断] | `pillar_features[i]` 必须始终对应 `voxel_coords[i]` |

## 4. PillarHist 数学与结构规范

### 4.1 主数学定义

对准入 pillar `v` 和高度 bin `b`：

\[
h=\frac{z_{max}-z_{min}}{B},\qquad
b_i=\left\lfloor\frac{z_i-z_{min}}{h}\right\rfloor.
\]

\[
S_{v,b}=\{i\mid(c_{x,i},c_{y,i})=v,\ b_i=b\},\qquad
H^p_{v,b}=|S_{v,b}|.
\]

\[
H^i_{v,b}=\begin{cases}
\dfrac{1}{H^p_{v,b}}\sum_{i\in S_{v,b}}r_i,&H^p_{v,b}>0,\\
0,&H^p_{v,b}=0.
\end{cases}
\]

pillar 物理中心为

\[
x_c=x_{min}+(c_x+\tfrac12)\delta_x,\qquad
y_c=y_{min}+(c_y+\tfrac12)\delta_y.
\]

拼接向量和输出为

\[
g_v=[H^p_v\mid H^i_v\mid x_c\mid y_c]\in\mathbb R^{2B+2},
\qquad Y_v=f_\theta(g_v)\in\mathbb R^D.
\]

以上 histogram、mean intensity、pillar center 和 linear projection 来自论文；`z-z_min` 是在 KITTI 负高度范围下使公式可实现的严格修正。[原文明示][公式推导]

### 4.2 单一结构配置源

| ID | 状态与证据 | 规范值 |
|---|---|---|
| PFE-001 | [已冻结][原文明示] | `B=64`；KITTI `h=4/64=0.0625 m` [公式推导] |
| PFE-002 | [已冻结][公式推导] | 先按半开 ROI 过滤，再计算 `floor((z-z_min)/h)`；保护性 clamp 必须计数，正常应为 0 |
| PFE-003 | [已冻结][原文明示] | `H^p` 使用 raw count；不归一化、不 `log1p` |
| PFE-004 | [已冻结][原文明示] | `H^i` 使用 bin 内 mean raw intensity；空 bin 为 0 |
| PFE-005 | [已冻结][合理推断] | intensity 沿用 loader 原始尺度，不 `/255`、不标准化 |
| PFE-006 | [已冻结][合理推断] | hard voxelizer 的 `voxel_coords` 只定义准入 pillar；准入 pillar 内统计全部 `points`，不受 32 点上限限制 |
| PFE-007 | [临时默认][待验证假设] | 当前论文最字面主轨使用 `raw_meter_xy`；`normalized_xy` 为工程对照 |
| PFE-008 | [临时默认][待验证假设] | 当前论文最字面主轨使用 `Linear(130,64,bias=True)`；bias 必须显式配置 |
| PFE-009 | [对照项][合理推断] | 工程结构：`Linear(130,64,bias=False)→BN1d(eps=1e-3,momentum=0.01)→ReLU` |
| PFE-010 | [已冻结][合理推断] | CPU oracle 与 GPU reference 只负责正确性；不得据其时延声称部署优势 |
| PFE-011 | [已冻结][合理推断] | GPU reference 使用无碰撞 int64 key、stable sort/search 和 deterministic segmented reduction |
| PFE-012 | [已冻结][合理推断] | 排序后必须恢复 original row；输出 `[V,64]` 与 `voxel_coords` 原行序一致 |
| PFE-013 | [已冻结][合理推断] | `V=1` 时输出仍为 `[1,64]`；禁止无维度约束的 `squeeze()` |
| PFE-014 | [已冻结][合理推断] | reference 路径显式关闭 autocast；仅把输入转为 `.float()` 不足以保证 `Linear` 不被 autocast |
| PFE-015 | [已冻结][合理推断] | `V=0` 时 VFE 返回 `pillar_features [0,64]`，保留 `voxel_coords [0,4]`；不得在 sort/max/index 处意外失败 |

PFE-007/PFE-008 是当前结构测试默认，而非作者真实实现的结论。R5 前只要求两种坐标和两种 projection 都能由配置正确构建；训练裁决推迟到 R7。[已冻结][待验证假设]

### 4.3 dtype 与 key 规则

| 数据 | 强制 dtype |
|---|---|
| OpenPCDet batch 中的 `points`、`voxel_coords` | 沿用 float32 输入 |
| 校验后的 `batch_id`、`cx/cy/bz` | `torch.int64` |
| packed pillar key、row、bin、segment key、permutation | `torch.int64` |
| `H^p` 计数存储 | int32；进入 concat 前转 float32 |
| intensity sum/mean | float32 |
| center、concat、projection | float32 reference path |

离散列转 long 前必须验证其为整数值：[已冻结][合理推断]

```python
coords_long = voxel_coords.round().long()
if not torch.equal(coords_long.to(voxel_coords.dtype), voxel_coords):
    raise ValueError('voxel_coords must contain integer-valued indices')

point_batch = points[:, 0].round().long()
if not torch.equal(point_batch.to(points.dtype), points[:, 0]):
    raise ValueError('points[:, 0] must contain integer-valued batch ids')
```

推荐无碰撞编码：[已冻结][公式推导]

```text
pillar_key  = batch_id * (Ny * Nx) + cy * Nx + cx
segment_key = original_row * NUM_BINS + bin_id
```

packed key 只在以下前置条件成立时无碰撞；所有断言必须在建 key 前执行：[已冻结][公式推导][合理推断]

```text
0 <= voxel batch_id < batch_size
0 <= point batch_id < batch_size
0 <= cx < Nx
0 <= cy < Ny
0 <= bz < NUM_BINS
KITTI 当前路径：grid_size[2] == 1 且 voxel z index == 0
active voxel coordinate 不得重复
```

点侧先执行半开 ROI 过滤，再计算离散索引和范围断言。保护性 clamp 只能吸收合法边界附近的浮点误差并计数；不得用 clamp 静默修复明显越界的 batch、x、y、z 或 bin。[已冻结][合理推断]

## 5. OpenPCDet 接口与文件边界

### 5.1 VFE 接口契约

`PillarHistVFE` 必须满足固定提交的 `Detector3DTemplate.build_vfe()` 接口：[已冻结][原文明示] 本地代码

```python
class PillarHistVFE(VFETemplate):
    def __init__(
        self,
        model_cfg,
        num_point_features,
        voxel_size,
        point_cloud_range,
        grid_size=None,
        **kwargs,
    ):
        ...

    def get_output_feature_dim(self):
        assert self.num_filters[-1] == 64  # 当前 KITTI 后端契约
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        ...
        batch_dict['pillar_features'] = pillar_features
        return batch_dict
```

强制输入：`points [P,5]`、`voxel_coords [V,4]`、`batch_size`；强制输出：`pillar_features [V,64]`。输出维数从 `NUM_FILTERS[-1]` 取得，但当前配置必须断言为 64。[已冻结][合理推断]

### 5.2 注册、配置与计划文件

进入编码时只新增或修改以下范围：[已冻结][合理推断]

```text
pcdet/models/backbones_3d/vfe/pillar_hist_vfe.py          # 新增
pcdet/models/backbones_3d/vfe/__init__.py                 # 注册
tools/cfgs/kitti_models/pointpillar_pillarhist.yaml       # 新增，不覆盖官方配置
tests/pillarhist/test_histogram.py                        # 新增
tests/pillarhist/test_boundaries.py                       # 新增
tests/pillarhist/test_scatter_alignment.py                # 新增
tests/pillarhist/test_integration.py                      # 新增
```

建议配置：[临时默认][待验证假设]

```yaml
VFE:
  NAME: PillarHistVFE
  NUM_BINS: 64
  NUM_FILTERS: [64]
  COUNT_MODE: raw
  INTENSITY_MODE: mean_raw
  COORD_MODE: raw_meter_xy
  USE_ALL_POINTS_IN_ADMITTED_PILLARS: true
  HISTOGRAM_DTYPE: float32
  REDUCTION_MODE: deterministic_segment
  PROJECTION:
    TYPE: linear
    BIAS: true
```

必须支持但不作为当前默认的工程组合：[对照项][合理推断]

```yaml
COORD_MODE: normalized_xy
PROJECTION:
  TYPE: linear_bn_relu
  BIAS: false
  BN_EPS: 0.001
  BN_MOMENTUM: 0.01
```

所有 enum 必须校验，未知值直接抛错。当前不建立 worktree；若用户后续改变决定，再升级文件布局说明。[已冻结][合理推断]

## 6. 张量契约与端到端数据流

### 6.1 依赖图

```text
points [P,5]
   │
   ├─ unchanged hard voxelizer ──► voxel_coords [V,4]
   │                                  │ active set + row order
   ▼                                  ▼
all-point XY/height mapping ──► Hp/Hi [V,64]
                                      │
                         center XY [V,2]
                                      │
                         concat G [V,130]
                                      │
                         projection [V,64]
                                      │ same row order as voxel_coords
                                      ▼
                         PointPillarScatter
                                      ▼
                         BEV Backbone → AnchorHeadSingle
                                      ▼
                         decode/NMS（当前仅验证可贯通）
```

### 6.2 张量尺寸

| 阶段 | 输入 | 输出 |
|---|---:|---:|
| batch raw points | 每样本 `[P_b,4]` | `[P,5]` |
| hard voxelizer | `[P_b,4]` | voxels `[V,32,4]`、counts `[V]`、coords `[V,4]` |
| histogram | points `[P,5]`、coords `[V,4]` | `H^p/H^i [V,64]` |
| concat | two hist + XY center | `[V,130]` |
| projection | `[V,130]` | `[V,64]` |
| scatter | feature + coords | `[B_s,64,496,432]` |
| backbone block 1 | scatter output | `[B_s,64,248,216]` |
| backbone block 2 | previous | `[B_s,128,124,108]` |
| backbone block 3 | previous | `[B_s,256,62,54]` |
| deblocks concat | 3 scales | `[B_s,384,248,216]` |
| class head | fused BEV | `[B_s,18,248,216]` |
| box head | fused BEV | `[B_s,42,248,216]` |
| direction head | fused BEV | `[B_s,12,248,216]` |

### 6.3 Scatter 与 batch 完整性

首版保持官方 `PointPillarScatter` 不变。[已冻结][合理推断]

固定提交中的官方 Scatter 通过 `coords[:,0].max()+1` 推断 batch size：[原文明示] 本地代码

- batch 中间的空样本只要其后仍有非空样本，Scatter 会为其产生零画布；
- 尾部空样本会使输出 batch 维度小于 `batch_dict['batch_size']`；
- 整个 batch 的 `V=0` 会在 `coords[:,0].max()` 处失败。

因此 VFE 的空输入能力与官方 Scatter 的空输入限制必须分开验收。首版保持官方 Scatter 不变，其尾部/全局空样本限制作为诊断性 `xfail`；若未来修 Scatter，必须建立独立补丁和实验轨道，不能混入“仅替换 VFE”的主结果。[已冻结][合理推断]

## 7. Reference 算法

### 7.1 CPU Oracle

```text
INPUT: points [P,5], active_coords [V,4]
validate batch/x/y/z ranges and uniqueness before packed keys
initialize Hp[V,64]=0, Hi_sum[V,64]=0
IF V == 0: return empty Hp/Hi with shape [0,64]
build active_key -> original_row

FOR point in points:
  check half-open XYZ ROI
  compute batch,cx,cy,bz
  IF point pillar is in active set:
    row = active key's original row
    Hp[row,bz] += 1
    Hi_sum[row,bz] += intensity

Hi = where(Hp>0, Hi_sum/Hp, 0)
RETURN Hp, Hi in active_coords original row order
```

CPU oracle 只用于测试真值，不进入训练和时延测量。[已冻结][合理推断]

### 7.2 GPU Deterministic Reference

```text
0. validate integer-valued inputs, ranges, KITTI z==0 and active-key uniqueness
1. if V==0, return pillar_features [0,64] without sort/max/search
2. construct active_key[V] in voxel_coords original row order
3. sorted_key, perm = stable sort(active_key)
4. ROI-filter points; construct validated point_key[P_valid]
5. pos = searchsorted(sorted_key, point_key)
6. valid = pos<V and sorted_key[pos]==point_key
7. original_row = perm[pos[valid]]
8. segment_key = original_row*64 + bin[valid]
9. stable sort by segment_key
10. deterministic segmented count/sum
11. write Hp/Hi to [V,64] using original_row
12. with autocast disabled, concat center and project; return [V,64]
```

正式 GPU reference 不得使用 Python dict；不得把排序后的 pillar 顺序直接交给 Scatter。[已冻结][合理推断]

### 7.3 Optimized Path

`PH_OPT` 可使用 `scatter_add`、`bincount`、atomic、自定义 CUDA 或 TensorRT plugin，但只有在通过 TST-018 后才允许用于时延和部署结论。[已冻结][合理推断]

R6 优先顺序固定为 compact-row `scatter_add`、内存可控 `bincount`、其他 PyTorch 原生实现；只有原生路径无法通过预注册性能门禁时，才单独评审自定义 CUDA。本轮最多保留一个 PH_OPT 主实现。[已冻结][合理推断]

TST-018 的冻结容差为：`Hp` 与 active row/order 精确一致；`Hi`、concat/projection input、pillar features 采用 `rtol=1e-5,atol=1e-6`；loss 采用 `rtol=1e-5,atol=1e-6`；projection/backend gradient 逐张量采用 `rtol=1e-4,atol=1e-6`。两侧 gradient norm 均大于 `1e-8` 时 cosine similarity 必须不低于 `0.9999`。[已冻结][合理推断]

REF/OPT correctness 必须从同一完整 model state、相同 train/eval 模式和相同输入开始。若 PH_OPT 使用 atomic 或非固定归约，同一输入至少重复 20 次；loss/gradient 的每次 backward 前必须清空梯度并恢复相同模型/输入状态，报告最坏误差、gradient norm 分布和最低 cosine similarity。[已冻结][合理推断]

correctness/repeatability 与 performance/deployment-like 分轨。性能模式允许为 REF/OPT 同时关闭 deterministic algorithms 以容纳 atomic，但不得只对 OPT 关闭，也不得用性能模式替代正确性结论。[已冻结][合理推断]

R6 正式 benchmark 使用 CUDA Event、显式 synchronize、30 次 warmup、每轮至少 200 次 VFE-only 与 100 次 end-to-end、5 轮独立重复；REF/OPT 交替测量并保存原始样本。预注册晋升门禁为：VFE-only p50 至少提升 10%，至少 4/5 轮正收益，end-to-end paired 95% CI 下界大于 0，峰值显存增幅不超过 10%或有明确收益解释。[已冻结][合理推断]

R6 状态定义为 `PASS_PROMOTED`、`PASS_LOCAL_ONLY`、`PASS_NOT_PROMOTED`、`BLOCKED/FAIL`。只有 `PASS_PROMOTED` 在 R7 使用 PH_OPT；前两种非晋升状态统一使用 PH_REF；正确性失败不得进入 R7。[已冻结][合理推断]

## 8. 强制测试与当前验收门禁

### 8.1 Histogram 与边界

| ID | 必测行为 | 通过标准 |
|---|---|---|
| TST-001 | 点守恒 | `sum(Hp)` 精确等于实际进入模型且落入 active set 的点数 |
| TST-002 | 准入守恒 | 达 pillar 上限时仍按 active set 计数，不等于 `sum(min(Nv,32))` |
| TST-003 | Z 半开边界 | `z_min` 有效、`z_max` 无效、内部边界只进入唯一 bin |
| TST-004 | XY 半开边界 | `x_min/y_min` 有效、`x_max/y_max` 无效 |
| TST-005 | 空 bin | mean intensity 精确为 0；无 NaN/Inf |
| TST-006 | all-points | 单 pillar 40 点时主路径 count=40；max32 对照 count=32 |
| TST-007 | 点顺序不变性 | 固定 active coords 后打乱 points，Hp 精确相同，Hi 满足容差 |
| TST-008 | clamp audit | 正常 ROI 过滤后 clamp count 为 0；非零必须报错或进入诊断 |

TST-001 的点数以模型实际收到的 `points` 为准，不以磁盘原始 `.bin` 为准。[已冻结][合理推断]

### 8.2 Key、row 与 batch

| ID | 必测行为 | 通过标准 |
|---|---|---|
| TST-009 | row alignment | 打乱 active coords 后，feature 仍逐行对应原 coords |
| TST-010 | batch isolation | 不同 batch 的相同 `(y,x,bin)` 不串扰 |
| TST-011 | discrete dtype | 非整数 batch/coords 必须报错；整数 float32 可无损转 long |
| TST-012 | packed key 与范围前置条件 | 最大合法 batch/grid/bin 组合无碰撞；`cx=Nx`、`cy=Ny`、`z!=0`、`batch_id=batch_size`、负 batch/x/y、非法 `bz` 和重复 active coordinate 均显式报错 |
| TST-013 | V=1 shape | 输出严格 `[1,64]` |
| TST-014A | VFE empty-sample isolation | `blocking`：空样本不污染其他 batch，VFE 无跨 batch 统计 |
| TST-014B | 官方 Scatter 尾部/全局空样本 | `diagnostic/expected-failure`：固定提交暴露已知限制时明确 `xfail` 并记录，不阻塞 R5 |

### 8.3 数值与集成

| ID | 必测行为 | 通过标准 |
|---|---|---|
| TST-015 | CPU/GPU reference | Hp 精确相等；Hi `rtol=1e-5,atol=1e-6` |
| TST-016 | determinism | 同硬件/软件环境：Hp 精确一致，Hi 在相同归约顺序下优先精确一致，pillar features 满足 `rtol=1e-5,atol=1e-6`；跨环境只要求规定容差 |
| TST-017 | AMP/dtype | autocast 开启的外层环境中，reference histogram、concat、projection 仍显式保持 float32 |
| TST-018 | REF/OPT 一致性 | Hp/row 精确；Hi/features、全链路、loss 和 gradient 按 7.3 冻结容差通过；非固定归约完成 20 次重复性审计 |
| TST-019 | registry/build | YAML 能构建完整 PointPillar 模型 |
| TST-020 | end-to-end shape | head channels 仍为 cls 18、box 42、dir 12，anchor 数不变 |
| TST-021 | baseline regression | 未启用 PH 时固定 input/checkpoint 输出 `rtol=1e-6,atol=1e-7` |
| TST-022 | loss forward | 总 loss 与各分量为 finite；不要求下降 |
| TST-023 | backward | projection 参数梯度存在且 finite；后端梯度链未断 |
| TST-024 | checkpoint round-trip | 保存/加载后固定输入输出在 TST-016 容差内一致 |
| TST-025 | Scatter spatial alignment | 人工 `2×2` 网格严格满足 `[b,z,y,x]→spatial_features[b,:,y,x]`；同步打乱 feature/coords 时 BEV 不变，仅打乱一侧时必须检测到错误 |
| TST-026 | V=0 VFE contract | `blocking`：VFE 返回 `pillar_features [0,64]`，无意外 sort/max/index 异常 |

R5 的完成条件是全部 `blocking` 测试通过；`diagnostic/expected-failure` 必须按预期触发并进入 test report 与 manifest。当前唯一预先认可的 `xfail` 是 TST-014B 所描述的固定提交官方 Scatter 限制。R5 不包含训练收敛或 AP 要求。[已冻结][合理推断]

## 9. Baseline 冻结与调试产物

R0 必须固定至少一个真实 KITTI batch，并保存：[已冻结][合理推断]

- 原始 batch_dict 键、shape、dtype、min/max/mean；
- `points`、`voxel_coords`、`voxel_num_points` 的 checksum；
- baseline `pillar_features`、scatter、三层 backbone、fused BEV 和 head 输出 shape/statistics；
- 最终预测 box/score/label checksum；
- 当前配置、Git HEAD、dirty diff 摘要和环境信息。

不得默认保存完整训练数据内容；固定 batch 工件应只包含复现测试所需张量，并遵循本地数据许可。[已冻结][合理推断]

## 10. 当前实验轨道

R6–R7 启用以下结构与训练轨道：[已冻结][合理推断]

| 名称 | 当前用途 | 是否用于训练/AP |
|---|---|---|
| `PP_OFFICIAL_SANITY` | 冻结官方 baseline 数据流 | 否；当前只做 forward/checksum |
| `PH_PAPER_LITERAL_REF` | raw-meter + Linear-only + deterministic reference；R6 正确性与性能基线 | R6 benchmark；若 PH_OPT 未晋升则作为 R7 reduction path |
| `PH_ENGINEERING_REF` | normalized/BN-ReLU 等可配置结构对照 | 仅保留 R0–R5 追溯，不直接进入 R7 排名 |
| `PH_OPT` | reference 对齐后的性能实现 | TST-018、VFE/end-to-end benchmark；仅 `PASS_PROMOTED` 时进入 R7 |
| `PH_RAW_LINEAR` | raw-meter + Linear | R7 候选 |
| `PH_NORM_LINEAR` | normalized + Linear | R7 候选 |
| `PH_RAW_BNRELU` | raw-meter + Linear/BN/ReLU | R7 候选 |
| `PH_NORM_BNRELU` | normalized + Linear/BN/ReLU | R7 候选 |
| `PP_SHORT_CONTROL` | 原始 PillarVFE | R7 协议校准，不参与候选排名 |

四个 PillarHist 候选必须使用同一 reduction path；`PP_SHORT_CONTROL` 固定使用原始 PillarVFE，不参与该约束。不得把本轮 OpenPCDet OneCycle 短程协议称为论文训练协议。[已冻结][合理推断]

## 11. 随机性与可追溯性

即使当前不训练，reference/integration 测试也必须记录：[已冻结][合理推断]

- Python、NumPy、PyTorch CPU/CUDA seeds；
- DataLoader generator 和 worker seed；
- cuDNN benchmark/deterministic、TF32、deterministic algorithms；
- 配置解析结果和输入 checksum；
- reference 或 optimized reduction mode。

R7 进行成对训练时，必须先复制形状兼容的后端初始权重，再创建 optimizer，并保存复制前后 checksum。相同 seed 不自动保证数据库采样和 augmentation 序列相同。[已冻结][合理推断]

## 12. 当前运行产物

结构与数据流阶段的每次运行使用独立目录：[已冻结][合理推断]

```text
outputs/pillarhist/structure/<track>/<timestamp>_<seed>/
├─ config_resolved.yaml
├─ manifest.json
├─ environment.txt
├─ data_manifest.json
├─ input_checksum.json
├─ tensor_contract.json
├─ diagnostics/
│  ├─ histogram_stats.json
│  ├─ overflow_and_clamp.json
│  └─ comparison.json
├─ tests/
│  └─ test_report.txt
└─ checkpoints/
   └─ round_trip_test.pth
```

`manifest.json` 至少包含：规范版本、track、Git HEAD、dirty diff 摘要、命令行、seed、reduction mode、开始/结束时间、退出状态、偏离条款，以及每个 `xfail` 的测试 ID、触发条件、异常摘要和代码依据。[已冻结][合理推断]

## 13. R7 短程训练与裁决规范

### 13.1 初始化与数据重放

从 seed 666 只构建一次 canonical backend master，并显式复制所有 shape-compatible 的 Scatter、Backbone 和 Head 权重。四个 PH 候选共享一份 canonical projection Linear weight。Linear-only 使用 `bias=True`；BN-ReLU 中 Linear 使用 `bias=False`，BN 初始化固定为 weight=1、bias=0、running mean=0、running var=1、eps=1e-3、momentum=0.01；单 GPU使用普通 BN，不启用 SyncBN。所有复制完成后才创建 optimizer、scheduler 和 AMP GradScaler。[已冻结][合理推断]

训练使用 seed 666、单张 RTX 5060 Laptop GPU、per-GPU/global batch=2、无 gradient accumulation、workers=0、AMP 关闭。优化器和调度器沿用冻结 OpenPCDet native Adam OneCycle：LR=0.003、weight decay=0.01、momentum=0.9、MOMS=[0.95,0.85]、PCT_START=0.4、DIV_FACTOR=10、gradient clip=10、无额外 warmup；总计 5000 optimizer steps。[已冻结][合理推断]

执行前生成单一 `training_replay_manifest`，逐 step 固定数据索引、sampler cursor、每样本 augmentation seed、GT database 采样 checksum、增强参数/结果 checksum、micro-batch 边界和 optimizer step。五轨消费同一计划；开头、结尾及以 seed 666 固定抽取的 20 个 step 必须比较增强后输入 checksum。[已冻结][合理推断]

checkpoint 必须包含 model、optimizer、scheduler、GradScaler 状态（即使禁用 AMP也记录）、Python/NumPy/PyTorch CPU/CUDA RNG、sampler cursor、optimizer step 和 accumulation state。只有完整恢复并在固定下一 batch 上重现 loss（`rtol=1e-5,atol=1e-6`），才视为同一训练序列。[已冻结][合理推断]

### 13.2 Validation 与裁决

每 1000 steps 在预注册 validation subset 上计算 3D AP_R40 Moderate；最终第 5000 step 在完整 KITTI validation 上评测。`last` checkpoint 是 primary，`peak_secondary` 只按 subset 三类 3D AP_R40 Moderate 宏平均选择。[已冻结][合理推断]

validation subset 固定为 KITTI `val.txt`（3769 帧，源文件 SHA-256 `657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86`）中按 `floor(i*(N-1)/(K-1)+0.5), i=0..K-1` 取得的 256 帧。完整 frame ID、零基 source index 与生成算法记录在 `PillarHist_R7_validation_subset_v1.3.json`；该文件 SHA-256 为 `92d3e3cb3c4154a236c2e9b2e7845c423a59fed6cf63c09dec4bf26df235adb8`，frame-ID 列表 SHA-256 为 `2217461dbcda24e92a69980c5db2815c2a1b81b81c7509bea1d0a81c47d0bcdd`。训练结果产生后不得更换子集。[已冻结][合理推断]

R7 主指标为完整 validation 上 Car、Pedestrian、Cyclist 的 3D AP_R40 Moderate 宏平均，使用冻结 OpenPCDet 评测脚本和 KITTI 标准类别 IoU 阈值。tie margin 为 1.0 个绝对 AP point；若前两名差距小于该值，只对并列候选增加 seed 667 复核。[已冻结][合理推断]

`PP_SHORT_CONTROL` 的预注册 sanity 门禁为：所有 backward step 的 loss/分量 finite 且正 anchor>0；每 100 steps 检查的 Backbone/Head 代表梯度 finite 且非零；最后 200 step 的 total-loss 中位数低于最初 200 step；最终完整 validation 产生非空预测，Car 3D AP_R40 Moderate 与三类宏平均均大于 0；checkpoint 完整恢复并复现固定下一 batch loss。任一失败即判定短程协议异常，不得在结果产生后修改该定义。[已冻结][合理推断]

R7 输出 `SELECTED`、`INCONCLUSIVE`、`NO_PROMOTION` 或 `BLOCKED`。单 seed 只用于候选筛选，不能证明统计优势；R7 后必须人工评审，不自动启动 R8。[已冻结][合理推断]

## 14. R8 完整 FP32 训练与评价规范

### 14.1 主轨、数据与训练协议

R8 只运行两个阻塞主轨：`PP_OFFICIAL_FULL` 使用冻结官方 PointPillars VFE；`PH_PAPER_LITERAL_FULL` 使用 64-bin、raw count、raw mean intensity、`raw_meter_xy` pillar center 与带 bias 的 Linear projection。normalized 副轨未获本轮预算批准，不属于 R8 闭环。[已冻结][合理推断]

正式 seed 固定为 666、667、668；执行顺序为 seed 666 `PP→PH`、seed 667 `PH→PP`、seed 668 `PP→PH`。seed 666/667 已在 R7 被观察，只有 seed 668 是新 seed，最终报告必须披露。[已冻结][合理推断]

本地 manifest 实测 train 3,712 帧、validation 3,769 帧。正式协议固定为单 GPU、per-GPU/global batch 4、gradient accumulation 1、workers 0、drop-last false、每 epoch 928 optimizer steps、80 epochs、总计 74,240 optimizer steps/轨。优化器采用官方 Adam OneCycle：LR 0.003、weight decay 0.01、MOMS `[0.95,0.85]`、PCT_START 0.4、DIV_FACTOR 10、gradient clip 10、无额外 warmup。[已冻结][原文明示][合理推断]

AMP/GradScaler 关闭，TF32 关闭，float32 matmul precision 为 `highest`，cuDNN benchmark=false、deterministic=true，`torch.use_deterministic_algorithms(true)`，`CUBLAS_WORKSPACE_CONFIG=:4096:8`。两轨必须使用完全相同的设置。[已冻结][合理推断]

batch 4 的冻结依据是探索性 smoke `outputs/pillarhist/r8/exploratory/20260918T_batch4_smoke_v5`：双轨均通过真实 KITTI forward/loss/backward、代表梯度、成对输入 checksum、哈希链和逐字节恢复更新检查；PP/PH 峰值 reserved 显存分别约 5.70/4.52 GB，低于 8,151 MiB 设备容量。[已冻结][原文明示]

### 14.2 初始化、重放与恢复

每个 seed 只生成一个 canonical backend initialization，并将所有 shape-compatible 的 Scatter、Backbone、Neck 和 Head 参数复制到两轨；复制完成后才创建 optimizer/scheduler。PillarVFE 与 PillarHist projection 按冻结 seed 独立初始化并保存未复制原因与 checksum。[已冻结][合理推断]

每个 seed 生成一份共享 `replay_steps.jsonl.gz`，逐 epoch 保存完整 permutation，逐样本保存 augmentation seed；每个 optimizer step 对增强后 points、voxels、coordinates、GT boxes/classes、frame ID 与 batch 边界计算 checksum，并进入 `H_k=SHA256(H_{k-1}||step_id||input_checksum_k)` 连续哈希链。每个 epoch 的首批、末批和一批预注册随机样本额外展开保存。两轨逐步 checksum 序列、epoch root 和 final root 必须完全一致。[已冻结][合理推断]

每 10 epoch 保存完整 checkpoint，至少保留 epoch 20/40/60/80；主 checkpoint 唯一固定为 epoch 80 `LAST_EPOCH`。checkpoint 包含 model、optimizer、scheduler 派生状态、RNG、GT database sampler、下一 epoch/batch cursor、replay/hash-chain 状态与输入文件 hash。[已冻结][合理推断]

每轨在 epoch 10 checkpoint 上执行固定 next-batch 恢复检查：输入、step、LR/momentum 严格相等；loss/分量使用 `rtol=1e-5,atol=1e-6`；梯度审计一致；恢复后一次 optimizer update 的完整模型 checksum 必须逐字节相等。本轮没有预注册非逐字节确定算子例外。[已冻结][合理推断]

梯度审计每 100 optimizer steps 检查 PillarHist projection（如存在）、第一个可训练 2D backbone weight 和 `dense_head.conv_cls.weight`，要求 present、finite 且 norm 严格大于零；一次失败即暂停该轨道，不允许结果产生后放宽。[已冻结][合理推断]

### 14.3 Validation、机器裁决与资源预算

只有 epoch 80 对全部 3,769 validation frames 执行正式评测，不生成或使用 best-validation checkpoint。必须报告 Car/Pedestrian/Cyclist 3D AP_R40 Easy/Moderate/Hard、每类三难度均值、三类 Moderate 宏平均、prediction count、recall、原始评测文本和未四舍五入 JSON。[已冻结][合理推断]

科学标签使用未四舍五入的 Car/Pedestrian 三难度均值 paired delta：两类 mean delta 均大于 0 且各至少 2/3 seed 为正时为 `DIRECTIONALLY_SUPPORTED`；两类 mean delta 均不大于 0 且各至多 1/3 seed 为正时为 `NOT_SUPPORTED`；其余为 `MIXED`。任一必需 AP 缺失或非 finite 时不生成科学标签。[已冻结][合理推断]

`PAPER_PROTOCOL_MISMATCH` 与科学标签分离。论文报告值与从公开逐难度 AP 重算值同时保存：PH Car 为 81.42 vs 81.41，PH Pedestrian 为 51.07 vs 51.0767（常规两位显示 51.08）。本地绝对 AP、成对收益和相对论文提升差分别报告。[已冻结][原文明示][公式推导]

30-step 正式负载近似 smoke 的 PP/PH 吞吐分别为 1.1612/1.1459 optimizer steps/s。以较慢值计算，单轨基础训练约 18.0 GPU 小时，加 20% 余量约 21.6 GPU 小时；六轨训练预算约 129.6 GPU 小时，另预留完整 validation 与恢复时间，总预算冻结为 132 GPU 小时。该预算不是停止线。[已冻结][原文明示][公式推导]

R8 的详细入口、重试、产物和终点以 `PillarHist_R8_closure_task_plan.md` v1.2 与 `PillarHist_R8_preregistration_v1.4.json` 为准。只有六轨有效且公平性、恢复、validation 与证据审计全部通过时，执行状态才能为 `PASS_COMPLETE` 或 `PASS_WITH_DOCUMENTED_RETRY`。[已冻结][合理推断]

## 15. R9 以后才生效的 PTQ 与部署规范

PTQ 只在 FP32 结构和训练配置冻结后启用。[开放项][合理推断]

保留的论文约束：[原文明示]

- W8A8 signed symmetric uniform；zero-point 0；`[-128,127]`；
- `s=2t/255`；
- weight 先于 activation；
- nuScenes val 均匀 64 帧；
- first/last layer 保持浮点。

仍需后续裁决：[开放项][待验证假设]

- per-tensor/per-channel；
- first/last layer 的准确范围；
- BN folding、bias 和 layer input/output activation 搜索；
- 100 候选与补充材料 `/i` 冲突；
- TensorRT 版本、GPU、plugin 和 engine 边界；
- FPGA/NPU/ASIC 的位宽、吞吐、功耗、存储和带宽预算。

fake quant 与真 INT8 必须分轨报告；只有真实 optimized runtime 可以用于部署时延结论。[已冻结][合理推断]

## 16. 当前风险门禁

| 等级 | 风险 | 当前门禁 |
|---:|---|---|
| S0 | 用 max32 点统计 histogram | TST-002/TST-006 |
| S0 | 负高度未减 `z_min` | TST-003 |
| S0 | sorted row 与 `voxel_coords` 错位 | TST-009/TST-020 |
| S0 | batch 间 key 冲突或非法坐标造成 packed-key 碰撞 | TST-010/TST-012 |
| S0 | feature/coords 空间位置或 x/y 语义错位 | TST-025 |
| S0 | registry/shape 不兼容 | TST-019/TST-020 |
| S1 | float coords 未验证就转 long | TST-011 |
| S1 | reference 使用非确定 atomic | TST-015/TST-016 |
| S1 | VFE 空样本串扰或 `V=0` 意外崩溃 | TST-014A/TST-026，均为 blocking |
| S1 | 官方 Scatter 尾部/全局空样本限制 | TST-014B 受控 xfail；当前只记录，不混改 Scatter |
| S1 | raw/normalized、projection 不确定 | 两种模式均可配置；R7 再训练裁决 |
| S2 | reference 时延被当作部署性能 | 实验轨道和报告名称隔离 |

## 17. 仍保留的开放问题

- [开放项][待验证假设] 作者真实 projection 是否包含 bias、BN、ReLU。
- [开放项][待验证假设] 作者实际使用 raw-meter 还是 normalized center。
- [开放项][待验证假设] 作者每卡 batch、SyncBN、epoch、scheduler 和 checkpoint selection。
- [开放项][待验证假设] 摘要中的 information entropy guidance 和 Figure 4 attention score 没有可实现定义；当前不得添加模块。
- [待 R6 裁决][合理推断] optimized reduction 按第 7.3 节冻结优先级实现；是否晋升由预注册性能门禁决定，不再作为可事后改写的开放项。
- [开放项][待验证假设] PTQ 和目标硬件参数见第 15 节。

这些问题不追溯改变 R0–R5；其中结构、坐标模式与 optimized reduction 必须分别由 R7、R6 按本版门禁裁决，其余不阻塞本轮闭环。[已冻结][合理推断]

## 18. 版本与变更控制

- v1.4 取代 v1.0–v1.3 和对应评审补充，作为 R8 唯一生效规范；R6/R7 结果仍按各自冻结版本追溯。[已冻结][合理推断]
- 改变数学语义、all-points/admission、tensor contract 或当前验收门禁时升级 major。[已冻结][合理推断]
- 补充测试、日志和不改变当前语义的实现说明时升级 minor。[已冻结][合理推断]
- R9 PTQ 规范冻结时必须发布新版本。[已冻结][合理推断]
- 每次更新记录：变更条款、原因、证据、受影响工件和旧结果可比性。[已冻结][合理推断]

### 18.1 v1.2 相对 v1.1 的规范变更

- 引入 `blocking` 与 `diagnostic/expected-failure` 两类测试，拆分 TST-014A/B，消除官方 Scatter 已知限制与 R5 门禁的冲突。
- 冻结 packed-key 的 batch/x/y/z/bin 范围和 active-coordinate 唯一性前置断言。
- 冻结 VFE 的 `V=0 → [0,64]` 契约，并将官方 Scatter 的空输入限制单独记录。
- 新增 TST-025 Scatter 人工空间对齐和 TST-026 VFE 空输入测试。
- 明确 reference 路径显式关闭 autocast、同环境/跨环境确定性标准、动态输出维数接口和 R7 初始化冻结时点。

### 18.2 v1.3 相对 v1.2 的规范变更

- 将 R0–R5 的完成提交与验收运行冻结为 R6–R7 入口。
- 冻结 PH_OPT 路径优先级、TST-018 的全链路/梯度容差、atomic 重复性和 correctness/performance 双模式。
- 冻结 CUDA Event 公平 benchmark、性能晋升门禁及 PROMOTED/LOCAL_ONLY/NOT_PROMOTED 状态。
- 冻结四条 PillarHist 候选、PP_SHORT_CONTROL、canonical 初始化、数据重放、完整恢复与 5000-step 单卡短程协议。
- 冻结完整 validation 的 3D AP_R40 Moderate 宏平均、1.0 AP tie margin、seed 667 并列复核及 PP_SHORT_CONTROL sanity 门禁。
- 明确 R7 只输出筛选结论，完成后人工评审，不自动进入 R8/R9。

### 18.3 v1.4 相对 v1.3 的规范变更

- 继承 R7 `INCONCLUSIVE`，以论文忠实性而非 R7 胜负选择 `raw_meter_xy + Linear` 主复现轨。
- 冻结三 seed、双轨、batch 4、80 epoch、严格 FP32 与交替执行顺序。
- 冻结全步骤 replay 哈希链、canonical backend、epoch-10 逐字节恢复检查和 epoch-80 唯一正式 validation。
- 冻结 Car/Pedestrian 机器科学标签、论文报告/重算值双记录和 `PAPER_PROTOCOL_MISMATCH` 独立字段。
- 根据 batch 4 双轨探索性 smoke 冻结显存可行性与 132 GPU 小时总预算。

## 19. 证据来源

### 19.1 论文

- [PillarHist — CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_PillarHist_A_Quantization-aware_Pillar_Feature_Encoder_based_on_Height-aware_Histogram_CVPR_2025_paper.html)
- [PillarHist — 官方补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Zhou_PillarHist_A_Quantization-aware_CVPR_2025_supplemental.pdf)
- 原 PDF：`D:\Zotero\storage\LW4SQGCL\Zhou 等 - 2025 - PillarHist A Quantization-aware Pillar Feature Encoder based on Height-aware Histogram.pdf`
- Zotero attachment：ID `836`，key `LW4SQGCL`

### 19.2 本地代码

- `/home/ksyang/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml`
- `/home/ksyang/OpenPCDet/pcdet/datasets/processor/data_processor.py`
- `/home/ksyang/OpenPCDet/pcdet/datasets/dataset.py`
- `/home/ksyang/OpenPCDet/pcdet/models/backbones_3d/vfe/pillar_vfe.py`
- `/home/ksyang/OpenPCDet/pcdet/models/backbones_2d/map_to_bev/pointpillar_scatter.py`
- `/home/ksyang/OpenPCDet/pcdet/models/detectors/detector3d_template.py`
- `/home/ksyang/OpenPCDet/pcdet/models/dense_heads/anchor_head_template.py`
- `/home/ksyang/OpenPCDet/pcdet/utils/loss_utils.py`

---

**v1.4 当前验收结论：** R8 以冻结的三种子双轨协议判断本地完整 FP32 可学习性、成对精度变化和论文方向/绝对值差距。低于论文 AP、没有提升或科学标签为 `MIXED/NOT_SUPPORTED` 不使有效闭环失败；任何数据重放、恢复、完整 validation 或证据门禁失败则不得宣称 R8 完成。本轮结果不等同于 PTQ、INT8、TensorRT 或目标硬件部署结论。
