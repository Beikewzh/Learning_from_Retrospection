# Method: Online RL Training

## Overview

We study online reinforcement learning for reasoning with a Qwen3-4B policy fine-tuned with LoRA. Our online RL algorithm is Group Relative Policy Optimization (GRPO), applied to two tasks: mathematical problem solving and Game of 24. Across both tasks, we keep the same high-level training loop: sample multiple rollouts per prompt, score them with a task reward, compute a group-normalized response-level advantage, and optionally add a token-level intrinsic term before the actor update.

The main goal of the study is not only to compare standard reward shaping baselines, but also to test whether token-level intrinsic signals can improve reasoning optimization. We therefore compare three classes of methods:

1. vanilla GRPO with only the task reward,
2. reward-side length baselines that modify the scalar task reward, and
3. intrinsic baselines that add a token-level signal to the GRPO advantage.

## Base Policy and Optimization Setup

For all online RL experiments, the actor policy is initialized from `Qwen3-4B` and trained with LoRA (`rank=64`, `alpha=64`). We use fp16 training and a learning rate of `1e-6`. Unless otherwise noted, all runs use:

- `rollout_batch_size = 16`,
- `global_batch_size = 16`,
- `n = 8` rollouts per prompt during training,
- validation every 5 steps with one sampled rollout per prompt.

For the final Math suite, we use `max_steps = 150` and `max_response_length = 1024`. For the final Game of 24 suite, we use `max_steps = 100` and `max_response_length = 512`, since the Game of 24 training set is much smaller and responses are substantially shorter.

## GRPO Training Objective

Let a prompt \(x_i\) produce \(n\) sampled responses \(\{y_{i,j}\}_{j=1}^n\). The reward function produces a token-level reward tensor, but under the default GRPO path used in this project, this reward is first collapsed to a scalar sequence score:

\[
R_{i,j} = \sum_t r_{i,j,t}.
\]

GRPO then normalizes these scores within the rollout group for the same prompt:

\[
\tilde{A}_{i,j} = \frac{R_{i,j} - \mu_i}{\sigma_i + \varepsilon},
\]

where \(\mu_i\) and \(\sigma_i\) are the mean and standard deviation of the \(n\) sequence scores for prompt \(x_i\). This scalar group-relative advantage is then broadcast over all valid response tokens:

\[
A^{\text{ext}}_{i,j,t} = \tilde{A}_{i,j}.
\]

The actor is then updated with the standard PPO-style clipped policy gradient objective over tokens, using \(A^{\text{ext}}_{i,j,t}\) or its intrinsic-augmented variant described below.

## External Reward Functions

### Math Task

For Math, the base reward combines answer correctness and output format. Responses are instructed to contain internal reasoning in `<think>...</think>` and a final boxed answer in `\boxed{}`. Let \(r_{\text{acc}}\in\{0,1\}\) denote exact answer correctness, computed by extracting the boxed answer and grading it against the reference answer, and let \(r_{\text{fmt}}\in\{0,1\}\) denote whether the response follows the expected output format. The external Math reward is

\[
r_{\text{math}} = (1-\lambda_{\text{fmt}})\, r_{\text{acc}} + \lambda_{\text{fmt}}\, r_{\text{fmt}},
\]

with \(\lambda_{\text{fmt}} = 0.1\).

### Game of 24

For Game of 24, each prompt contains four card values and asks the model either to produce a valid arithmetic expression that uses each card exactly once and evaluates to 24, or to output `NO` if the instance is unsolvable. As in Math, we also include a small format component encouraging `<think>...</think><answer>...</answer>` outputs. Let \(r_{\text{acc}}\in\{0,1\}\) be 1 if:

- the prompt is solvable and the model outputs a valid equation using exactly the provided cards once each and evaluating to 24, or
- the prompt is unsolvable and the model outputs `NO`.

Let \(r_{\text{fmt}}\in\{0,1\}\) be the format indicator. The external Game of 24 reward is

\[
r_{\text{24}} = (1-\lambda_{\text{fmt}})\, r_{\text{acc}} + \lambda_{\text{fmt}}\, r_{\text{fmt}},
\]

again with \(\lambda_{\text{fmt}} = 0.1\).

## Reward-Side Length Baselines

We include three scalar reward baselines that modify the external reward before GRPO computes the group-relative advantage.

### Correct-Only Group-Length Baseline

This baseline is adapted from a recent paper. For each prompt group, we consider only the correct responses and compute a z-score of their response lengths. If a response is incorrect, its reward is zero and it does not participate in the z-score calculation. If a response is correct, its reward is

\[
r = 1 - \alpha \, \sigma(z_{\ell}),
\]

where \(\sigma(\cdot)\) is the sigmoid function and \(z_{\ell}\) is the per-prompt z-scored response length among correct responses only. We use \(\alpha = 0.01\).

This baseline captures a relative preference for shorter correct solutions without penalizing incorrect responses directly.

### L1-Exact

This baseline follows the exact-length objective:

\[
r = \mathbb{I}(y = y^\star) - \alpha \, |n_{\text{target}} - n_y|,
\]

where \(n_y\) is the generated response length and \(n_{\text{target}}\) is a prescribed target length. We use \(\alpha = 0.01\). For Math we set \(n_{\text{target}} = 512\); for Game of 24 we set \(n_{\text{target}} = 256\).

### L1-Max

This baseline follows the soft maximum-length formulation:

\[
r = \mathbb{I}(y = y^\star)\cdot \mathrm{clip}\big(\alpha (n_{\text{target}} - n_y) + \delta,\; 0,\; 1\big),
\]

where \(\delta = 0.5\). As above, we use \(\alpha = 0.01\), with \(n_{\text{target}} = 512\) for Math and \(n_{\text{target}} = 256\) for Game of 24.

Compared with L1-Exact, this baseline imposes a softer budget constraint while preserving a preference for correct, shorter responses.

## Intrinsic Baselines

Our intrinsic baselines modify the token-level advantage rather than the scalar external reward. Let \(A^{\text{ext}}_{i,j,t}\) denote the broadcast GRPO advantage described above. We define a token-level intrinsic signal \(A^{\text{int}}_{i,j,t}\), normalize it within each prompt group, and then combine it with the external advantage as

\[
A_{i,j,t} = A^{\text{ext}}_{i,j,t} + \eta \, g_{i,j} \, A^{\text{int}}_{i,j,t}.
\]

Here \(\eta\) is the intrinsic weight and \(g_{i,j}\) is an outcome-dependent gate. In our final experiments we use an asymmetric gate:

\[
g_{i,j} =
\begin{cases}
1, & \text{if the response is incorrect}, \\
-1, & \text{if the response is correct}.
\end{cases}
\]

This means that the intrinsic term is added on failures and subtracted on successes.

### Group-Zscore Normalization

The key normalization choice in the final experiments is prompt-group normalization. For each prompt, we pool all valid token-level intrinsic source values across the \(n\) rollouts for that prompt, compute a single mean and standard deviation from that pooled set, and z-score every token in the group using those shared statistics:

\[
z_{i,j,t} = \frac{s_{i,j,t} - \mu_i}{\sigma_i + \varepsilon},
\]

where \(s_{i,j,t}\) is the raw intrinsic source at token \(t\), and \(\mu_i,\sigma_i\) are computed from the pooled token values across all rollouts for prompt \(x_i\). This makes the intrinsic signal relative within a prompt group, in the same spirit as GRPO's response-level group normalization.

### Transform Choices

We evaluate two transforms on top of the group-zscored intrinsic signal:

1. **Identity**:
   \[
   A^{\text{int}}_{i,j,t} = z_{i,j,t},
   \]
2. **Tanh**:
   \[
   A^{\text{int}}_{i,j,t} = \tanh(z_{i,j,t}).
   \]

The tanh variant preserves sign while smoothly limiting extreme magnitudes.

### LeaRS: Latent AR Error

For LeaRS, we first capture a latent representation from the actor log-probability path. A lightweight autoregressive model is then trained on buffered latent trajectories to predict the next latent state. The per-token latent prediction error is used as the intrinsic source \(s_{i,j,t}\). In other words:

\[
s_{i,j,t}^{\text{LeaRS}} = \| h_{i,j,t+1} - \hat{h}_{i,j,t+1} \|,
\]

where \(h_{i,j,t+1}\) is the target latent and \(\hat{h}_{i,j,t+1}\) is the AR model prediction. We apply temporal smoothing and then prompt-group z-scoring before combining the signal with the external advantage.

For the final Math suite, the LeaRS auxiliary AR model uses a lightweight configuration (`d_model=256`, `2` layers, `4` heads) with a lighter update schedule than our early experiments (`train_steps=30`, `train_every_n_steps=5`).

### Response-Length Intrinsic

This baseline uses the scalar response length as the intrinsic source. For each response, its length is repeated over all valid response tokens and then normalized with prompt-group z-scoring:

\[
s_{i,j,t}^{\text{len}} = n_{i,j}.
\]

Although the raw source is constant within a response, its normalized and gated contribution is still meaningful as an intrinsic control signal at the advantage level.

### Entropy Intrinsic

This baseline uses sampled-token surprise as a proxy for token entropy. If \(\log p_{i,j,t}\) is the log-probability of the sampled token under the policy, then

\[
s_{i,j,t}^{\text{ent}} = -\log p_{i,j,t}.
\]

This quantity is high when the chosen token is surprising under the current policy and low when it is confident. We again apply prompt-group z-scoring and optionally a tanh transform before adding it to the GRPO advantage.

## Final Method Sets

### Math

Our final Math online RL suite contains:

1. vanilla GRPO,
2. correct-only group-length baseline,
3. L1-Exact,
4. L1-Max,
5. LeaRS + group z-score,
6. LeaRS + group z-score + tanh,
7. length intrinsic + group z-score,
8. length intrinsic + group z-score + tanh,
9. entropy intrinsic + group z-score,
10. entropy intrinsic + group z-score + tanh.

The shared Math settings are:

- `max_steps = 150`,
- `max_response_length = 1024`,
- `rollout_batch_size = 16`,
- `global_batch_size = 16`,
- `n = 8`.

### Game of 24

Our final Game of 24 suite uses the same 10-method structure, replacing only the task reward with the Game of 24 correctness rule and using a shorter target length for the L1 baselines. The shared settings are:

- `max_steps = 100`,
- `max_response_length = 512`,
- `rollout_batch_size = 16`,
- `global_batch_size = 16`,
- `n = 8`.

The reduction from 150 to 100 steps is motivated by dataset size: the Game of 24 training set contains 1638 examples, so a single pass with `rollout_batch_size = 16` yields only about 102 optimizer steps.

## Summary

The central design choice in our online RL experiments is to separate reward-side and advantage-side interventions. Reward-side baselines change the scalar sequence reward before GRPO normalization, while intrinsic baselines preserve the task reward and instead add a token-level, outcome-gated term to the GRPO advantage. This allows us to compare whether improvements come from explicit scalar reward shaping, or from richer token-level guidance layered on top of the same response-level RL objective.
