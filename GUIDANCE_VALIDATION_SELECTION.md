# Guidance validation selection

Guidance hyperparameters were selected only on the six validation start/goal
pairs. Each row uses the same U-Net checkpoint, 32 samples per pair, identical
latent-noise seeds, and full-URDF COAL rescoring. The test split was not used to
choose the setting.

| Setting | Final DDIM fraction | Step | Steps | Max perturbation | Proxy target | Physical free | 8 cm safe | Best-of-32 tasks | Free modes/pair | Length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weak | 20% | 0.005 | 2 | 0.04 | 0.04 m | 13.0% | 5.2% | 6/6 | 2.50 | 10.529 m |
| medium | 30% | 0.010 | 2 | 0.08 | 0.05 m | 26.6% | 7.8% | 6/6 | 4.17 | 10.257 m |
| selected | 40% | 0.020 | 2 | 0.12 | 0.06 m | 30.2% | 10.4% | 6/6 | 4.33 | 12.294 m |

The unguided controlled baseline was 3.6% physical-free, 0.5% 8 cm safe, and
4/6 best-of-32 tasks. The selected setting prioritizes collision-free planning
and topology recovery over the medium setting's shorter mean path. Exact COAL
filtering remains mandatory because the differentiable primitive/SAT collision
model is an optimization surrogate, not a safety certificate.
