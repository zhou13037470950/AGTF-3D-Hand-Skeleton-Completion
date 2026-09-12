# RepairFormer 消融实验与多模型可视化工具

该工具针对当前纯骨架修补版 RepairFormer：

- 时间线性插值锚点；
- 局部图几何分支；
- 时序 Transformer 分支；
- 自适应门控融合；
- 时空 Transformer 解码；
- 图结构与局部时间卷积细化；
- 坐标、速度、加速度、骨长、骨方向、一致性和指尖加权损失。

## 一、放置文件

把压缩包内容复制到现有项目根目录，保持结构：

```text
项目根目录/
├─ configs/
│  ├─ ablation_experiment.yaml
│  └─ visualization_models.yaml
├─ src/
│  └─ ablation_models.py
├─ run_ablation.py
├─ plot_ablation.py
├─ visualize_model_outputs.py
└─ smoke_test_ablation.py
```

原项目应继续保留：

```text
src/model.py
src/losses.py
src/factory.py
src/config.py
src/metrics.py
src/utils.py
configs/default_optimized.yaml
```

## 二、消融究竟怎么“拆”

### 1. 结构单元消融

结构消融不是加载完整模型后在测试时把某个输出设为0，而是：

1. 不实例化被删除模块；
2. 用同一数据划分、种子、训练预算从头训练；
3. 用相同固定测试掩码统一评价。

本工具支持：

| 变体 | 删除内容 | 科学问题 |
|---|---|---|
| `full` | 无 | 完整模型基准 |
| `no_anchor` | 插值锚点残差路径 | 插值先验是否有效 |
| `no_geometry` | 图几何编码分支 | 局部手部拓扑是否有效 |
| `no_temporal` | 时序Transformer分支 | 时序运动建模是否有效 |
| `no_gate` | 自适应门控 | 自适应融合是否优于固定平均 |
| `no_refinement` | 二次细化模块 | 图—时间细化是否有效 |

`no_geometry`、`no_temporal` 会真正减少参数量；`no_gate` 保留两个分支，但固定使用 0.5 平均。

### 2. 损失项消融

损失消融保持模型结构完全相同，只把对应权重设为0，然后重新训练：

| 变体 | 修改 |
|---|---|
| `no_velocity_loss` | `velocity=0` |
| `no_acceleration_loss` | `acceleration=0` |
| `no_structure_loss` | `bone=0, direction=0` |
| `no_consistency_loss` | `consistency=0` |
| `no_fingertip_weight` | `fingertip_weight=1` |

## 三、先运行单元测试

```powershell
python smoke_test_ablation.py
```

它会对每个变体检查：

- 能否实例化；
- 前向输出形状；
- 损失是否为有限数；
- 能否反向传播；
- 是否存在有效梯度；
- 参数量是否正常。

只检查部分变体：

```powershell
python smoke_test_ablation.py --variants "full,no_geometry,no_temporal"
```

## 四、训练与统一测试

先检查：

```yaml
# configs/ablation_experiment.yaml
ablation:
  variants: [...]
  repeats: 3
  full_grid_val_every: 5
```

一键从头训练并测试：

```powershell
python run_ablation.py --mode all
```

指定数据路径：

```powershell
python run_ablation.py --mode all --data-root "G:/ges_re/dhg2016_repairformer/DHG2016"
```

只跑部分变体：

```powershell
python run_ablation.py --mode all --variants "full,no_geometry,no_temporal,no_gate"
```

中断后继续：

```powershell
python run_ablation.py --mode train --resume
```

跳过已有 `best.pt`：

```powershell
python run_ablation.py --mode all --skip-existing
```

仅重新统一测试：

```powershell
python run_ablation.py --mode evaluate
```

输出：

```text
runs/ablation/
├─ full/seed_42/
│  ├─ best.pt
│  ├─ latest.pt
│  ├─ history.csv
│  ├─ used_config.yaml
│  ├─ variant.yaml
│  └─ test_detail.csv
├─ no_geometry/seed_42/
└─ results/
   ├─ ablation_detail.csv
   ├─ ablation_runs.csv
   ├─ ablation_summary.csv
   └─ ablation_summary.json
```

`best.pt` 不是按一次随机验证保存，而是每隔若干轮执行：

```text
5种mask_type × 4种ratio
```

并按照完整验证平均 MPJPE 保存。

## 五、重复实验次数

正式论文建议：

```yaml
repeats: 3
start_seed: 42
```

等价于种子：

```text
42、43、44
```

也可以显式写：

```yaml
seeds: [42, 43, 44, 45, 46]
```

最终 `ablation_summary.csv` 自动输出均值和标准差。

## 六、绘制消融指标图

```powershell
python plot_ablation.py
```

生成：

```text
runs/ablation/results/plots/
├─ mpjpe.png
├─ pck_at_0.1.png
├─ velocity_error.png
├─ bone_error.png
└─ mpjpe_delta_vs_full.png
```

## 七、加载多个模型生成可视化图

先修改：

```text
configs/visualization_models.yaml
```

可混合加载：

- 提出模型；
- 任意消融模型；
- GRU-AE、TCN-AE、ST-GCN-AE、Transformer-MAE、Motion-MAE、Anatomy-MAE。

基本运行：

```powershell
python visualize_model_outputs.py
```

指定模型：

```powershell
python visualize_model_outputs.py --models "proposed,anatomy_mae,no_geometry"
```

指定样本数、帧数和缺失类型：

```powershell
python visualize_model_outputs.py `
  --num-samples 5 `
  --num-frames 6 `
  --mask-type whole_finger `
  --mask-ratio 0.30
```

指定固定样本和固定帧：

```powershell
python visualize_model_outputs.py `
  --sample-indices "0,20,100" `
  --frames "0,4,8,12,16,20,24,31"
```

调节实验次数：

```powershell
python visualize_model_outputs.py `
  --num-samples 5 `
  --mask-repeats 3
```

含义是5个样本，每个样本生成3个不同固定掩码，共15组可视化实验。

### 行列调整

自动排版：

```powershell
python visualize_model_outputs.py --rows 0 --cols 0
```

默认每一行对应一个帧，每一列对应：

```text
Ground truth | Masked | Model 1 | Model 2 | ...
```

固定为4行5列，超出的面板自动分页：

```powershell
python visualize_model_outputs.py --rows 4 --cols 5
```

固定为3行4列：

```powershell
python visualize_model_outputs.py --rows 3 --cols 4
```

输出目录：

```text
runs/model_visualization/
├─ sample_0000_repeat_00_page_01.png
├─ sample_0000_repeat_00_page_02.png
├─ visualization_metrics.csv
└─ ...
```

保存原始输出数组：

```powershell
python visualize_model_outputs.py --save-npz
```

## 八、论文消融表建议

主表可只保留当前核心指标：

| Variant | Anchor | Geometry | Temporal | Gate | Refinement | MPJPE↓ | Velocity↓ | Bone↓ | PCK@0.1↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o anchor |  | ✓ | ✓ | ✓ | ✓ |  |  |  |  |
| w/o geometry | ✓ |  | ✓ | ✓ | ✓ |  |  |  |  |
| w/o temporal | ✓ | ✓ |  | ✓ | ✓ |  |  |  |  |
| w/o gate | ✓ | ✓ | ✓ |  | ✓ |  |  |  |  |
| w/o refinement | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |
| Full model | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |

损失消融建议另放一张表，避免结构与损失项混在一起难以解释。

## 九、公平性底线

- 所有变体使用同一训练/验证/测试受试者划分；
- 相同训练轮数、batch size、学习率和优化器；
- 相同掩码分布；
- 测试使用相同 `fixed_seed`；
- 每个结构变体必须重新训练；
- 不根据测试集选择 checkpoint；
- 至少3个种子报告 `mean ± std`；
- 参数量不同属于结构消融的自然结果，需要同时报告。
