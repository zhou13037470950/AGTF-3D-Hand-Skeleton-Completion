# RepairFormer Ablation Study & Multi‑model Visualization Tool
This tool is designed for the skeleton‑only RepairFormer completion framework:
- Temporal linear interpolation anchor
- Local graph‑based geometric branch
- Temporal Transformer branch
- Adaptive gated fusion
- Spatio‑temporal Transformer decoder
- Graph‑structure and local temporal convolution refinement
- Weighted loss for coordinate, velocity, acceleration, bone length, bone direction, consistency and fingertips

## 1. File Placement
Copy the extracted contents into your existing project root directory, preserving the following structure:
```text
Project Root/
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

Keep original project files unchanged:

```text
src/model.py
src/losses.py
src/factory.py
src/config.py
src/metrics.py
src/utils.py
configs/default_optimized.yaml
```

## How to Perform Ablations

### 1. Architectural Ablation

Architectural ablation **does not** load a full model and zero‑out certain outputs during inference. The correct workflow is:

1. Do not instantiate the removed module;
2. Train from scratch with identical data split, random seed and training budget;
3. Evaluate under identical fixed test masks.

Supported variants:

| Variant | Removed Component | Research Question |
|---|---|---|
| `full` | None | Full‑model baseline |
| `no_anchor` | Interpolation anchor residual path| Is interpolation prior beneficial? |
| `no_geometry` | Graph‑based geometric encoder branch | Does local hand topology help performance? |
| `no_temporal` | Temporal Transformer branch | Is temporal motion modeling effective? |
| `no_gate` | Adaptive gating | Does adaptive fusion outperform fixed averaging? |
| `no_refinement` | Secondary refinement module | Is graph‑temporal refinement useful? |

`no_geometry` and `no_temporal` genuinely reduce model parameter count. `no_gate` keeps both branches but replaces adaptive gate with fixed 0.5 averaging.

### 2.Loss‑term Ablation

For loss ablation, keep the exact same model architecture, set corresponding loss weight to zero and retrain:

| Variant | Modification |
|---|---|
| `no_velocity_loss` | `velocity=0` |
| `no_acceleration_loss` | `acceleration=0` |
| `no_structure_loss` | `bone=0, direction=0` |
| `no_consistency_loss` | `consistency=0` |
| `no_fingertip_weight` | `fingertip_weight=1` |

## 3.Run Sanity Check First

```powershell
python smoke_test_ablation.py
```

It validates each variant for:

- Successful module instantiation
- Correct forward‑pass output shape
- Finite loss values
- Valid backward propagation
- Valid gradient computation
- Reasonable parameter count

Test only selected variants:

```powershell
python smoke_test_ablation.py --variants "full,no_geometry,no_temporal"
```

## 4. Training and Unified Evaluation
Check ablation configuration in:

```yaml
# configs/ablation_experiment.yaml
ablation:
  variants: [...]
  repeats: 3
  full_grid_val_every: 5
```

Launch end‑to‑end training and evaluation:

```powershell
python run_ablation.py --mode all
```

Specify dataset root path:

```powershell
python run_ablation.py --mode all --data-root "G:/ges_re/dhg2016_repairformer/DHG2016"
```

Run partial variants only:

```powershell
python run_ablation.py --mode all --variants "full,no_geometry,no_temporal,no_gate"
```

Resume interrupted training:

```powershell
python run_ablation.py --mode train --resume
```

Skip variants with existing `best.pt`:

```powershell
python run_ablation.py --mode all --skip-existing
```

Run evaluation only (skip training):

```powershell
python run_ablation.py --mode evaluate
```

Output directory structure:

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

`best.pt` is not saved based on one random validation run. Instead, evaluation is performed every several epochs over:
`5 mask_types × 4 mask ratios`
Checkpoint is selected by averaged MPJPE across the full validation grid.

## 5. Number of Repeated Runs

Recommended setting for formal paper experiments:

```yaml
repeats: 3
start_seed: 42
```

This corresponds to seeds: `42, 43, 44`.

You may also explicitly define seed list:

```yaml
seeds: [42, 43, 44, 45, 46]
```

The output `ablation_summary.csv` automatically reports mean and standard deviation.

## 6.Plot Ablation Metrics

```powershell
python plot_ablation.py
```

Generated figures under:

```text
runs/ablation/results/plots/
├─ mpjpe.png
├─ pck_at_0.1.png
├─ velocity_error.png
├─ bone_error.png
└─ mpjpe_delta_vs_full.png
```

## 7. Multi‑model Visualization

Modify configuration file first:

```text
configs/visualization_models.yaml
```

Supported model candidates for mixed comparison:

- Proposed model
- Any ablation variants
- GRU‑AE, TCN‑AE, ST‑GCN‑AE, Transformer‑MAE, Motion‑MAE, Anatomy‑MAE

Basic execution:

```powershell
python visualize_model_outputs.py
```

Select specific models:

```powershell
python visualize_model_outputs.py --models "proposed,anatomy_mae,no_geometry"
```

Configure sample count, frame count and missing pattern:

```powershell
python visualize_model_outputs.py `
  --num-samples 5 `
  --num-frames 6 `
  --mask-type whole_finger `
  --mask-ratio 0.30
```

Use fixed sample indices and selected frames:

```powershell
python visualize_model_outputs.py `
  --sample-indices "0,20,100" `
  --frames "0,4,8,12,16,20,24,31"
```

Set repeated mask generation for each sample:

```powershell
python visualize_model_outputs.py `
  --num-samples 5 `
  --mask-repeats 3
```

> Meaning: 5 source samples, each with 3 different fixed masks, producing total 15 visualization groups.

### Layout Adjustment

Auto‑arrange layout:

```powershell
python visualize_model_outputs.py --rows 0 --cols 0
```

Default layout: each row corresponds to one frame, columns follow:

```text
Ground truth | Masked | Model 1 | Model 2 | ...
```

Fixed layout: 4 rows × 5 columns, overflow panels will be paginated automatically:

```powershell
python visualize_model_outputs.py --rows 4 --cols 5
```

Fixed layout: 3 rows × 4 columns:

```powershell
python visualize_model_outputs.py --rows 3 --cols 4
```

Visualization outputs:

```text
runs/model_visualization/
├─ sample_0000_repeat_00_page_01.png
├─ sample_0000_repeat_00_page_02.png
├─ visualization_metrics.csv
└─ ...
```

Save raw prediction arrays as npz:

```powershell
python visualize_model_outputs.py --save-npz
```

## 8. Suggested Ablation Table for Paper

Main table for core architectural metrics:

| Variant | Anchor | Geometry | Temporal | Gate | Refinement | MPJPE↓ | Velocity↓ | Bone↓ | PCK@0.1↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o anchor |  | ✓ | ✓ | ✓ | ✓ |  |  |  |  |
| w/o geometry | ✓ |  | ✓ | ✓ | ✓ |  |  |  |  |
| w/o temporal | ✓ | ✓ |  | ✓ | ✓ |  |  |  |  |
| w/o gate | ✓ | ✓ | ✓ |  | ✓ |  |  |  |  |
| w/o refinement | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |
| Full model | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |

It is recommended to place loss‑ablation results in a separate table, to avoid mixing architectural and loss modifications.

## 9. Fairness Constraints

- All variants share identical train‑val‑test subject split
- Identical training epochs, batch size, learning rate and optimizer
- Identical mask distribution
- Same `fixed_seed` for test‑time masking
- Every architectural variant must be retrained from scratch
- No checkpoint selection based on test‑set performance
- Report `mean ± std` with at least 3 random seeds
- Parameter count difference is a natural outcome of architectural ablation; parameter numbers should be reported alongside metrics