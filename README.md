# DHG2016 3D Skeleton Missing‑Value Completion Training Project

This project only reads the `skeleton_world.txt` 3D skeleton data from DHG2016 / DHG‑14/28. RGB and depth images are not used. Features include:

- Recursively traverse directories of `gesture/finger/subject/essai`;
- Dataset statistics, 3D skeleton preview in PNG/GIF format;
- Random missing scenarios: random joints, entire fingers, continuous time blocks, fingertips, and full‑frame loss;
- Graph convolution local geometric branch + temporal Transformer branch + gated fusion;
- Joint optimization objective: coordinates, velocity, bone length, action classification, finger configuration;
- Training/validation logging, best‑checkpoint saving, periodic checkpoint saving, automatic resume training;
- Batch evaluation across different missing types and missing ratios;
- PNG/GIF visualization for raw vs completed skeleton sequences.

## 1. Dataset Directory

Typical folder structure：

```text
DHG2016/
└── gesture_1/
    └── finger_1/
        └── subject_1/
            └── essai_1/
                └── skeleton_world.txt
```

Reader supports two data formats:

- One frame per line, each line contains 66 numbers (22 joints × 3 coordinates);
- One joint coordinate per line (3 numbers per line), every 22 lines compose one frame.

## 2. Installation

```bash
cd dhg2016_repairformer
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Modify Dataset Path

Edit `configs/default.yaml`：

```yaml
data:
  root: "D:/datasets/DHG2016"
```

You can also override the path at runtime:

```bash
python inspect_dataset.py --data-root "D:/datasets/DHG2016"
```

## 4. Inspect Dataset and Joint Edges

```bash
python inspect_dataset.py \
  --config configs/default.yaml \
  --data-root "D:/datasets/DHG2016" \
  --index 0 \
  --output-dir dataset_preview
```

Output file：

```text
dataset_preview/skeleton_preview.png
dataset_preview/skeleton_preview.gif
```

**Important Note**: The default `edges` in configuration follows the standard 22‑joint layout: wrist + palm + 4 joints per finger. Joint indices may vary across different dataset versions. Please verify skeleton connectivity via preview outputs. If connections are incorrect, only modify `data.edges`, `finger_groups` and `fingertips`. No other code modification is required.

## 5. Start Training

```bash
python train.py \
  --config configs/default.yaml \
  --data-root "D:/datasets/DHG2016" \
  --output-dir runs/experiment_01
```

Training outputs are organized as below：

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

- `best.pt`: Saved according to missing‑joint MPJPE on validation set.
- `latest.pt`: Overwritten after every epoch for training resumption

## 6. Resume Training

Resume from specified checkpoint:

```bash
python train.py \
  --config runs/experiment_01/used_config.yaml \
  --resume runs/experiment_01/latest.pt
```

Auto‑detect `latest.pt` inside output directory:

```bash
python train.py \
  --config runs/experiment_01/used_config.yaml \
  --resume auto
```

To increase total training epochs, modify `train.epochs` inside `used_config.yaml`.

## 7. Plot Training Curves

```bash
python plot_history.py --history runs/experiment_01/history.csv
```

This script generates loss and MPJPE curves.

## 8. Batch Evaluation on Test Set

```bash
python evaluate.py \
  --config runs/experiment_01/used_config.yaml \
  --checkpoint runs/experiment_01/best.pt \
  --output runs/experiment_01/test_results.json
```

The script evaluates pre‑configured missing types with missing ratios ranging from 10% to 40%.

## 9. Preview Single Sample Completion

```bash
python predict_repair.py \
  --config runs/experiment_01/used_config.yaml \
  --checkpoint runs/experiment_01/best.pt \
  --index 0 \
  --mask-type whole_finger \
  --ratio 0.30 \
  --output-dir repair_preview
```

Output files:

```text
repair_preview/repair_comparison.png
repair_preview/repair_animation.gif
```

## 10. Sanity Check without Real Dataset

Run smoke test with synthetic data to verify pipeline correctness:

```bash
python smoke_test.py
```

The smoke test covers data loading, forward pass, backward pass, validation, model checkpointing and preview rendering.

## 11. Loss Function

Default loss formulation:

```text
L = 1.0 Lcoord
  + 0.5 Lvelocity
  + 0.2 Lbone
  + 0.3 Lgesture
  + 0.1 Lfinger
```

- `Lcoord`: Smooth L1 loss calculated only on artificially masked missing joints;
- `Lvelocity`: Trajectory velocity error between completed sequence and ground truth;
- `Lbone`: Length error of bones affected by missing joints.


All loss weights can be adjusted in yaml config for ablation studies.

## 12. Suggested Experiments for Paper

At least include the following baselines and ablation groups:

1. Linear interpolation;
2. Coordinate‑only loss;
3. Coordinate + velocity loss;
4. Coordinate + velocity + bone‑length loss;
5. Full model (with gesture and finger configuration semantic losses);
6. Full model ablations: remove geometric branch, remove temporal branch, remove gated fusion respectively.

Report metrics: missing‑joint MPJPE, velocity error, bone‑length error, and quantitative results under different missing patterns.
