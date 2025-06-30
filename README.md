# BatchNorm Folding

This project demonstrates how to **remove BatchNorm layers after training** by folding their behaviour into the preceding `Linear` layers. This is a technique useful for simplifying and optimising models at inference time (reduce paramaters & increase speed).

## Motivation

**BatchNorm** is useful during training as primarily it helps to stabilise activations and gradients and acts as a regulariser of sorts.

During training, BatchNorm tracks running statistics (mean and variance), however, during **inference**, BatchNorm is no longer adaptive, it uses the **fixed, pre-computed statistics** (the running mean and variance accumulated during training). As a result, it becomes a **fixed affine transformation**:

$$
\text{BN}(z) = \gamma \cdot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

This transformation is just a **per-feature scale and shift**. Therefore, we can **fold it into the weights and bias of the preceding `Linear` layer**, eliminating the BatchNorm layer entirely at inference time.

---

## What This Project Does

- Builds a 3-layer MLP with `Linear → BatchNorm → ReLU` blocks
- Trains it briefly on synthetic data
- Computes equivalent `Linear` weights that include the effect of BatchNorm
- Reconstructs the model using only `Linear → ReLU` layers (no BatchNorm)
- Verifies that the predictions match exactly

---

## Math Behind the Folding

Given a `Linear` layer:

$$
z = Wx + b
$$

followed by a `BatchNorm` layer:

$$
\text{BN}(z) = \gamma \cdot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Define:

$$
\text{scale} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}
$$

Then fold the BatchNorm into the Linear layer as:

- **New weights:**
  $$
  W_{\text{folded}} = \text{scale} \cdot W
  $$

- **New biases:**
  $$
  b_{\text{folded}} = \text{scale} \cdot (b - \mu) + \beta
  $$

This produces a new `Linear` layer that behaves identically to `Linear → BatchNorm`, allowing the BatchNorm layer to be safely removed.

---

## Files

- `bn_fold.py`: training, folding, and verification script

Shows:

```bash
max_err=1.4901161193847656e-08
Parameters before folding    : 3,009
Parameters after  folding    : 2,817
Parameters saved              : 192
Inference time before folding: 0.309 ms
Inference time after  folding: 0.144 ms
Speed-up                     : 2.15x
```

---

## Notes

- Folding works **only for BatchNorm**, because its statistics are fixed at inference.
- It does **not** apply to `LayerNorm` or `GroupNorm`, which compute statistics per input and remain input-dependent at inference.
