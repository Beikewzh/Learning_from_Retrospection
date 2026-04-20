# Offline Analysis Method for MATH Trajectories

We performed an offline analysis of sampled reasoning trajectories on `MATH-500`. The purpose of this analysis was to operationalize trajectory complexity using three complementary metrics computed from each rollout: response length, eigenspectrum decay, and latent autoregressive prediction error. These metrics were then used to compare successful and failed rollouts, both within individual models and in pooled multi-model analyses.

For each problem, we sampled multiple independent rollouts from a frozen pretrained reasoning model using a prompt that required an explicit reasoning trace inside `<think>...</think>` and a final answer in `\boxed{...}`. In the canonical setting, we generated `32` rollouts per problem over the full `500` problems in `MATH-500`, with temperature `1.0` and maximum response length `4096`. Each rollout was scored by extracting the final boxed answer and comparing it against the ground-truth answer with an exact mathematical grader, yielding a binary success label.

For every sampled completion, we extracted a latent trajectory by rerunning the model with hidden states enabled and retaining one hidden-state vector per response token from the final hidden layer. When a valid `<think>...</think>` segment was present, downstream analysis was restricted to that reasoning subsequence; otherwise the full response-token trajectory was analyzed. Thus each rollout `i` is represented by a sequence

\[
Z_i = (z_{i,1}, z_{i,2}, \dots, z_{i,T_i}), \qquad z_{i,t} \in \mathbb{R}^d,
\]

where `T_i` is the analyzed trajectory length and `d` is the hidden dimension of the underlying model.

To measure predictive complexity, we trained a small offline autoregressive model separately for each base-model run. This predictor is a compact causal Transformer over latent vectors: an input projection maps the native latent dimension `d` to a fixed internal width, learned positional embeddings are added, a causal Transformer stack processes the sequence, and an output projection maps back to the original latent space. In the canonical configuration, the AR model used width `256`, `2` Transformer layers, `4` attention heads, and dropout `0.1`. Given a latent sequence `Z_i`, the AR model is trained to predict the next latent from the prefix using masked mean squared error,

\[
\mathcal{L}_{\mathrm{AR}}
=
\frac{1}{\sum_{b,t} m_{b,t}}
\sum_{b,t} m_{b,t}
\frac{1}{d}
\lVert \hat z_{b,t+1} - z_{b,t+1} \rVert_2^2,
\]

where `m_{b,t}` masks padded positions. The AR model was trained offline on up to `20000` trajectories from the corresponding merged run for `5000` optimization steps. Because a separate AR model was trained for each base model, raw AR errors are model-relative and require normalization for pooled multi-model comparisons.

We treat three rollout-level quantities as trajectory-complexity metrics. The first is response length,

\[
\ell_i = \mathrm{response\_length}_i,
\]

which captures how much textual computation the model externalizes. The second is eigenspectrum decay, which measures how concentrated the latent trajectory is in a low-dimensional subspace. Let `Z_i \in \mathbb{R}^{T_i \times d}` denote the analyzed latent matrix for rollout `i`. After centering across time,

\[
\tilde Z_i = Z_i - \mathbf{1}\mu_i^\top,
\qquad
\mu_i = \frac{1}{T_i} \sum_{t=1}^{T_i} z_{i,t},
\]

we form the empirical covariance

\[
C_i = \frac{\tilde Z_i^\top \tilde Z_i}{T_i - 1}.
\]

If the eigenvalues of `C_i` are ordered as `\lambda_{i,1} \ge \lambda_{i,2} \ge \cdots`, we fit the leading log-log eigenspectrum using

\[
\log \lambda_{i,r} \approx a_i - \alpha_i \log r,
\]

and define the decay metric as

\[
\mathrm{decay\_rate}_i = \alpha_i.
\]

Larger values indicate faster spectral decay, i.e. variance concentrated in fewer principal directions. In our complexity framing, this metric captures geometric concentration of the trajectory.

The third metric is latent autoregressive prediction error. For each valid step,

\[
e_{i,t} = \frac{1}{d}\lVert \hat z_{i,t} - z_{i,t} \rVert_2^2,
\qquad t = 2, \dots, T_i,
\]

and the rollout-level prediction error is

\[
\mathrm{ar\_error}_i = \frac{1}{T_i - 1} \sum_{t=2}^{T_i} e_{i,t}.
\]

This quantity measures how difficult the trajectory is for a small causal predictor to model from its own past. In the present framing, it captures predictive complexity rather than generalization.

A central issue in the final analysis is that raw complexity metrics are not directly comparable across all responses. Different models induce different latent scales, and different questions induce different baseline regimes of length and variability. We therefore compute z-scored versions of the metrics. For a metric `m_i` and a model `M`, within-model normalization is

\[
\mu_M = \frac{1}{|\mathcal I_M|} \sum_{i \in \mathcal I_M} m_i,
\qquad
\sigma_M = \operatorname{sd}\{m_i : i \in \mathcal I_M\},
\qquad
z_i^{(M)} = \frac{m_i - \mu_M}{\sigma_M}.
\]

This yields quantities such as `response_length_z_by_model`, `decay_rate_z_by_model`, and `ar_error_z_by_model`. For pooled multi-model analysis, however, the more appropriate reference group is the pair `(M,q)`, where `M` is the model and `q` is the problem identifier. In that case,

\[
\mu_{M,q} = \frac{1}{|\mathcal I_{M,q}|} \sum_{i \in \mathcal I_{M,q}} m_i,
\quad
\sigma_{M,q} = \operatorname{sd}\{m_i : i \in \mathcal I_{M,q}\},
\quad
z_i^{(M,q)} = \frac{m_i - \mu_{M,q}}{\sigma_{M,q}}.
\]

If a group has zero variance, all entries in that group are set to zero. This produces the three model-question normalized complexity metrics emphasized in the final pooled analysis:

- `response_length_z_by_model_question`,
- `decay_rate_z_by_model_question`,
- `ar_error_z_by_model_question`.

These normalized quantities ask whether, for a fixed model on a fixed problem, a rollout is relatively longer, more geometrically concentrated, or easier/harder to predict than its peer rollouts.

We use these complexity metrics in three main analyses. First, we compare their distributions between successful and failed rollouts. Second, we compute within-question paired gaps,

\[
\Delta_q(m)
=
\operatorname{mean}\{m_i : i \in \mathrm{failures\ on\ } q\}
-
\operatorname{mean}\{m_i : i \in \mathrm{successes\ on\ } q\},
\]

to ask whether failed rollouts are systematically more or less complex than successful ones on the same problem. Third, we relate empirical per-question success rate to per-question aggregates of the complexity metrics. In the pooled multi-model notebook, we perform both model-aware overlays and marginalized analyses after explicit normalization. Because question-normalized metrics have means near zero by construction, medians and paired-gap summaries are generally more informative than raw question-level means for these variables.

We also apply several robustness controls. We optionally exclude responses that hit the generation cap (`response_length \ge 4096`) to reduce truncation artifacts, and we often restrict analysis to mixed-outcome questions containing at least one successful and one failed rollout so that within-question comparisons are well defined. In addition to ordinary binned success-rate curves, we also examine conservative lower-confidence-bound summaries that favor bins with both high empirical success and low uncertainty.

The final combined offline analysis therefore centers on the raw complexity metrics

- `response_length`,
- `decay_rate`,
- `ar_error`,

and on their within-model and within-(model, question) normalized variants. Together, these quantities provide a compact characterization of trajectory complexity in terms of externalized computation, geometric concentration, predictive difficulty, and relative rank within a local support set.
