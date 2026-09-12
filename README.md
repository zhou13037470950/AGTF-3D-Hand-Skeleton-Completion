# DHG2016 三维骨架缺失修补训练项目

本项目只读取 DHG2016 / DHG-14/28 的 `skeleton_world.txt` 三维骨架，不使用 RGB 和深度图。包含：

- 递归读取 `gesture/finger/subject/essai` 目录；
- 数据统计、三维骨架 PNG/GIF 预览；
- 随机关节点、整根手指、连续时间块、指尖、整帧缺失；
- 图卷积局部几何分支 + 时序 Transformer 分支 + 门控融合；
- 坐标、速度、骨长、动作分类、手指配置联合目标；
- 训练/验证日志、最佳模型、定期模型、自动续训；
- 不同缺失类型与比例的批量评估；
- 修补前后 PNG/GIF 可视化。

## 1. 数据目录

常见结构：

```text
DHG2016/
└── gesture_1/
    └── finger_1/
        └── subject_1/
            └── essai_1/
                └── skeleton_world.txt
```

读取器兼容：

- 每一帧一行、每行 66 个数字（22×3）；
- 每 22 行组成一帧、每行 3 个数字。

## 2. 安装

```bash
cd dhg2016_repairformer
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 修改数据路径

编辑 `configs/default.yaml`：

```yaml
data:
  root: "D:/datasets/DHG2016"
```

也可以运行时覆盖：

```bash
python inspect_dataset.py --data-root "D:/datasets/DHG2016"
```

## 4. 先检查数据和关节连线

```bash
python inspect_dataset.py \
  --config configs/default.yaml \
  --data-root "D:/datasets/DHG2016" \
  --index 0 \
  --output-dir dataset_preview
```

输出：

```text
dataset_preview/skeleton_preview.png
dataset_preview/skeleton_preview.gif
```

**重要：**配置中的 `edges` 默认采用“腕部+掌心+每根手指4关节”的常见22点排列。不同版本的关节编号可能不同。请通过预览确认连线；若不正确，只修改 `data.edges`、`finger_groups` 和 `fingertips`，其余代码不用改。

## 5. 开始训练

```bash
python train.py \
  --config configs/default.yaml \
  --data-root "D:/datasets/DHG2016" \
  --output-dir runs/experiment_01
```

训练输出：

```text
runs/experiment_01/
├── used_config.yaml
├── history.csv
├── best.pt
├── latest.pt
├── epoch_010.pt ...
└── previews/
    └── epoch_005.png ...
```

`best.pt` 按验证集缺失点 MPJPE 保存；`latest.pt` 每轮覆盖，供断点续训。

## 6. 继续训练

指定模型：

```bash
python train.py \
  --config runs/experiment_01/used_config.yaml \
  --resume runs/experiment_01/latest.pt
```

自动查找输出目录中的 `latest.pt`：

```bash
python train.py \
  --config runs/experiment_01/used_config.yaml \
  --resume auto
```

要增加总训练轮数，先修改 `used_config.yaml` 中的 `train.epochs`。

## 7. 绘制训练曲线

```bash
python plot_history.py --history runs/experiment_01/history.csv
```

输出 loss、MPJPE 曲线。

## 8. 测试集批量评估

```bash
python evaluate.py \
  --config runs/experiment_01/used_config.yaml \
  --checkpoint runs/experiment_01/best.pt \
  --output runs/experiment_01/test_results.json
```

程序会测试配置中的不同缺失类型与 10%–40% 缺失比例。

## 9. 预览单个修补过程

```bash
python predict_repair.py \
  --config runs/experiment_01/used_config.yaml \
  --checkpoint runs/experiment_01/best.pt \
  --index 0 \
  --mask-type whole_finger \
  --ratio 0.30 \
  --output-dir repair_preview
```

输出：

```text
repair_preview/repair_comparison.png
repair_preview/repair_animation.gif
```

## 10. 验证代码能否运行

不需要真实数据，运行合成数据冒烟测试：

```bash
python smoke_test.py
```

测试会完成一次数据读取、前向传播、反向传播、验证、模型保存和预览生成。

## 11. 目标函数

默认：

```text
L = 1.0 Lcoord
  + 0.5 Lvelocity
  + 0.2 Lbone
  + 0.3 Lgesture
  + 0.1 Lfinger
```

- `Lcoord`：只计算人为缺失关节点的 Smooth L1；
- `Lvelocity`：修补轨迹与真实轨迹的速度误差；
- `Lbone`：受缺失影响的骨骼边长度误差；


权重均可在 YAML 中修改，便于开展消融实验。

## 12. 论文实验建议

至少保留以下对照：

1. 线性插值；
2. 仅坐标损失；
3. 坐标 + 速度；
4. 坐标 + 速度 + 骨长；
5. 完整模型（加入动作与手指配置语义）；
6. 分别去掉几何分支、时间分支和门控融合。

报告缺失点 MPJPE、速度误差、骨长误差以及不同缺失模式下的结果。
