# SE(3) diffusion training and comparison strategy

## What is trained

The experiment has three trained checkpoints and four evaluated variants:

| Variant | Trained checkpoint | Conditioning | Inference guidance |
|---|---|---|---|
| `unet_no_guidance` | 1-D conditional U-Net | start and goal SE(3) poses | no |
| `unet_guidance` | the same U-Net checkpoint | start and goal SE(3) poses | MPD-style differentiable cost gradient |
| `dit_no_guidance` | trajectory DiT | start and goal SE(3) poses | no |
| `dit_cross_environment` | trajectory DiT with cross-attention | start/goal plus box-obstacle tokens | no |

The guidance comparison intentionally reuses the U-Net weights. Guidance is a
change to reverse diffusion, not a separately trained network. This separates
the effect of the learned prior from the effect of collision/smoothness costs.

## Data and leakage prevention

The default collection is 50 independently sampled start/goal pairs times six
explicit route classes, for 300 accepted trajectories. Every accepted path has
run through OMPL RRTConnect + COAL, constrained B-spline smoothing, TOPP-RA,
MPPI/MuJoCo tracking, and a final zero-margin COAL audit.

The split is by start/goal pair, not trajectory. Thus the six paths belonging
to one task can never be divided between train and test. Training-only reverse
augmentation swaps start and goal and reverses the path, which removes an
unnecessary south-to-north directional bias without changing the sampled
regions. Validation and test paths are never augmented.

Each geometric reference is arc-length resampled to 128 poses. A pose is stored
as `xyz + rotation-6D`; unlike raw quaternion regression, this has no sign
discontinuity. The condition is the concatenated start and goal in the same
representation. Both endpoints are hard-clamped in noisy training inputs and
after every DDIM update. Only the 62 interior poses contribute to noise loss.
Normalization statistics are fitted only on the training split.

This is an intentional adaptation of `mpd-splines-public`, not a literal copy.
MPD's `TrajectoryDatasetBspline` learns the interior control points of a fixed
B-spline parameterization and uses the endpoints as context; its loaders group
splits by task ID, and its DDPM/DDIM inference applies differentiable guide
costs. The present planner produces a variable number of position and
quaternion control points, so directly padding those controls would mix knot
semantics. A fixed arc-length SE(3) pose sequence keeps samples comparable and
retains the same endpoint-conditioning, task-split, and inference-guidance
principles. TOPP-RA timing and MPPI traces remain in the source dataset for a
later dynamics model but are not targets of this geometric path prior.

The local MPD references used for this design are:

- `scripts/generate_data/generate_trajectories.py` for valid task sampling;
- `mpd/datasets/trajectories_dataset_bspline.py` for endpoint context and
  interior-only path targets;
- `mpd/models/diffusion_models/diffusion_model_base.py` and
  `sample_functions.py` for late, clipped DDPM/DDIM guide updates;
- `scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-config_file_v01_00.yaml`
  for the Warehouse/Panda guidance ablation structure.

## Three training stages

1. Train the 1-D FiLM U-Net baseline. Evaluate the same EMA checkpoint with and
   without inference guidance.
2. Train a compact DiT with AdaLN modulation from diffusion time and start/goal
   condition. This tests architecture while keeping semantic conditioning the
   same as the U-Net.
3. Train a DiT whose transformer blocks cross-attend to tokens describing each
   collision box (`center xyz, size xyz, quaternion wxyz`). This tests the
   mechanism required for multi-map training.

For all checkpoints, the forward process is
`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon`; the network
predicts the velocity target
`v = sqrt(alpha_bar_t) epsilon - sqrt(1-alpha_bar_t) x_0`, with MSE evaluated
only on interior path nodes. Velocity prediction keeps clean-path recovery
well-conditioned at the near-zero-SNR end of the cosine schedule. At
inference, deterministic DDIM predicts `x_0` and advances to the next selected
noise level. Position and rotation-6D channels share the training-split
standardization, while SO(3) is projected back to an orthonormal matrix before
conversion to `wxyz`.
DDIM clean-path predictions are clipped to ±4 in standardized coordinates;
this covers the observed training range while preventing the final near-zero
SNR cosine step from amplifying a small noise-prediction error into an
out-of-workspace path.

The 128-node default was selected after an explicit compression audit: all
pilot reference paths retained their 8 cm COAL margin at 128 nodes, whereas
one of eight dropped below 8 cm when represented by only 64 piecewise-SE(3)
nodes. Defaults otherwise use a 100-step cosine diffusion process, 25-step deterministic DDIM,
4000 optimizer steps per model, batch size 64, AdamW peaking at `2e-4` after a
200-step warmup followed by cosine decay, EMA `0.995`, gradient clipping, and
mixed precision on CUDA. The defaults are deliberately
small enough for the available RTX 4080 Laptop GPU and the 300-path dataset.
The best EMA checkpoint is selected on a fixed-noise validation batch; training
stops after eight validation intervals without improvement.
The experiment wrapper reruns this compressed-representation audit over all
300 paths and refuses to train if any path collides, loses its topology, or
falls below 8 cm.

## Guidance and collision guarantees

Following MPD's separation between learned diffusion and inference-time
optimization, guidance differentiates a weighted cost through the predicted
clean path during the final 40% of DDIM updates. The default cost contains:

- penetration/clearance penalties from a differentiable copy of the active
  URDF primitives (one body OBB, two rotor spheres, and four finite cylinders)
  against every oriented obstacle box;
- position second-difference smoothness;
- path-length regularization;
- sampling-bound penalties.

Each guidance gradient is RMS-normalized, endpoints receive zero gradient, and
the total update is clipped. Starting guidance late preserves the topology and
diversity learned from demonstrations; applying a strong obstacle cost from the
first noisy step tends to collapse all samples into the same broad corridor.
Concretely, a late DDIM clean-path prediction is updated by
`x_0 <- clamp(x_0 - eta * grad(C)/RMS(grad(C)))`, then converted back to the
equivalent noise prediction for the DDIM update. Two clipped cost steps are used
per guided diffusion step, with a normalized step of `0.02` and maximum local
perturbation of `0.12`. Three configurations were compared only on the six
validation endpoint pairs; the selected 6 cm proxy target is conservative and
gave the best exact COAL safety rate for the true 8 cm planning margin.

The differentiable primitive/SAT model is a fast surrogate, not the safety certificate.
The full URDF geometry and COAL are non-differentiable. Consequently every
reported sample is densely interpolated and rescored by COAL at zero, 8 cm,
and 10 cm margins. Deployment/data acceptance must reject or replan any sample
that fails this exact audit. In other words, guidance raises the success rate;
COAL provides the guarantee.

For execution, select a COAL-certified sample, fit the existing constrained
B-spline with its endpoint poses held fixed, audit that continuous spline again,
then run the unchanged TOPP-RA and MPPI/MuJoCo stages. The diffusion benchmark
stops before this downstream controller so planning quality is not confounded
with another smoothing or tracking pass.

## Comparison protocol

For each held-out start/goal pair, each variant draws 32 paths with identical
endpoint constraints and the same initial Gaussian-noise stream. The evaluator reports:

- physical collision-free rate and 8/10 cm clearance success;
- rate of valid separator-cut signatures and collision-free signatures;
- number of distinct collision-free topology classes recovered per task;
- pairwise positional/orientation diversity;
- distance to the nearest expert path;
- translation/rotation length, acceleration, and jerk;
- endpoint error and sampling latency.

The generated NPZ files contain no Rerun/GIF data. Exact per-sample metrics are
written to `evaluation/per_sample_metrics.csv`; the aggregate table is written
to `evaluation/COMPARISON.md` and `comparison.json`.

## Important limitation of the present map

All 300 demonstrations share one obstacle map. Environment cross-attention can
therefore validate the implementation and preserve a clean interface for map
tokens, but it cannot establish generalization to unseen maps: the environment
condition is constant throughout training. A scientifically meaningful map-
conditioned comparison requires a later dataset with randomized obstacle
layouts or multiple Blender scenes, while retaining task-grouped (and ideally
map-grouped) test splits.

## Commands

Collect or resume the 300-path dataset:

```bash
./collect_diffusion_dataset.sh \
  --pair-count 50 \
  --trajectories-per-pair 6 \
  --maximum-trajectory-attempts 24 \
  --output-dir datasets/diffusion_se3_multihomotopy_v002_300 \
  --resume
```

Train all checkpoints, generate four prediction sets, and run exact COAL
evaluation:

```bash
./run_diffusion_three_stage_experiment.sh
```

To audit and train another resolution consistently, set for example
`DIFFUSION_SEQUENCE_LENGTH=256`; the wrapper intentionally rejects a raw
`--sequence-length` override.

Training can be resumed at the sampling/evaluation phase without retraining:

```bash
/home/z017/research/diffusion_model/.envs/mpd-splines/bin/python \
  train_se3_diffusion.py --sample-only
./run_se3_diffusion_evaluation.sh
```
