# Revisiting Variance Reduction in Policy Gradients for LLM Reinforcement Learning

**Author:** **Yifan Zhang**, Quanquan Gu 

**Date:** December 27, 2025

## Abstract

Reinforcement Learning (RL) has become central to aligning Large Language Models (LLMs) with complex reasoning tasks. While recent advancements have refined KL-regularized policy gradient objectives, the high variance inherent in gradient estimators remains a persistent bottleneck, often necessitating prohibitively large batch sizes or conservative update steps.

This work revisits the principles of **Stochastic Variance-Reduced Policy Gradient (SVRPG)** and adapts them to the high-dimensional domain of LLM alignment. We propose a variance-reduced estimator that leverages periodic policy snapshots to construct a control variate specifically for the KL-regularized objective. We establish a theoretical framework demonstrating that this approach mitigates the instability characteristic of REINFORCE-style estimators in vast token spaces.

## 1. Introduction: The Variance Bottleneck in Reasoning

The paradigm of post-training LLMs has evolved from simple instruction following to complex reasoning and reward-guided exploration. A dominant framework in this domain is the KL-regularized policy gradient, which balances the maximization of task-specific rewards with a constraint to maintain proximity to a reference distribution.

However, the optimization process is impeded by the enduring challenge of **high variance in gradient estimation**. In the context of LLMs, characterized by vast vocabulary-based action spaces and substantial trajectory lengths, this variance manifests as unstable training dynamics and suboptimal sample efficiency.

To address this, we reconcile standard LLM alignment with **SVRPG** (Papini et al., 2018). We formulate an algorithm specifically tailored for the KL-regularized reasoning objective, demonstrating that a periodic snapshot mechanism can be efficiently integrated to serve as a control variate, thereby stabilizing learning without incurring prohibitive memory overheads.

## 2. Theoretical Framework: Regularized Policy Gradients

To apply stochastic variance reduction, we first require a rigorous definition of the gradient for the KL-regularized objective $J(\theta) = \mathbb{E}[R] - \beta \text{KL}$ under off-policy sampling.

### 2.1 Unnormalized Divergences

In reasoning tasks, we often deal with unnormalized objectives. We derive exact gradients for both **Unnormalized Forward KL (UFKL)** and **Unnormalized Reverse KL (URKL)**.

The **Unnormalized Forward KL** is defined as:

$$
\text{UKL}(\pi_{\mathrm{old}}\|\pi_\theta) = \int_x \pi_{\mathrm{old}}(x)\log\frac{\pi_{\mathrm{old}}(x)}{\pi_\theta(x)}\,dx + \int_x \bigl(\pi_\theta(x)-\pi_{\mathrm{old}}(x)\bigr)\,dx
$$

### 2.2 Differentiable Surrogate Losses

Crucially for implementation in frameworks like PyTorch, we derive differentiable surrogate losses $\mathcal{L}(\theta)$ such that $\nabla_\theta \mathcal{L}(\theta)$ is an unbiased estimator of $-\nabla_\theta J(\theta)$.

For the **Unnormalized Reverse KL (URKL)**—which is theoretically equivalent to the $k_3$ estimator used in methods like **GRPO** (Group Relative Policy Optimization)—the surrogate loss is:

$$
\mathcal{L}_{\mathrm{URKL}}(\theta) = Z_{\mathrm{old}} \mathbb{E}_{x\sim\tilde{\pi}_{\mathrm{old}}}\left[ -w(x)R(x) + \beta\bigl(w(x)\log w(x) - w(x)\bigr) \right]
$$

Where $w(x) = \frac{\pi_\theta(x)}{\pi_{\mathrm{old}}(x)}$ is the importance weight.

## 3. Stochastic Variance Reduction (SVRPG)

Standard estimators (like REINFORCE) suffer from variance bounded by $\mathbb{E}[\|\Psi(x, \theta) \nabla \log \pi_\theta\|^2]$. In reasoning tasks, where rewards $R(x)$ are sparse, the gradient signal is often dominated by the variance of the KL penalty term.

We employ a control variate technique. Let $\mathbf{g}(\tau; \theta)$ denote the stochastic gradient estimator. We construct a semi-stochastic estimator $\mathbf{v}_k$:

$$
\mathbf{v}_k = \mathbf{g}(\tau; \theta_k) - \rho_k(\tau) \mathbf{g}(\tau; \tilde{\theta}) + \mu
$$

* $\tilde{\theta}$: Parameters of a "snapshot" policy (a lagging anchor).
* $\mu$: The exact gradient of the snapshot policy (approximated via a large anchor batch).
* $\rho_k(\tau)$: Importance sampling weights to correct distribution mismatch.

As $\theta_k$ converges toward $\tilde{\theta}$, the term $\mathbf{g}(\tau; \theta_k) - \rho_k(\tau) \mathbf{g}(\tau; \tilde{\theta})$ tends toward zero, leaving only the stable term $\mu$.

## 4. Algorithm: SVRPG for LLM Alignment

We propose an interleaved update schedule where the snapshot $\tilde{\theta}$ is updated periodically. To handle the instability of importance weights in high-dimensional token spaces, we apply a **Dual-Clip** strategy to the importance weights $\rho(\tau)$.

### Implementation Logic

1.  **Snapshot Phase (Anchor):**
    * Sample a "Large Batch" $$\mathcal{D}\_L$$ from the snapshot policy $$\pi_{\tilde{\theta}}$$.
    * Compute the anchor gradient $$\hat{\mu} = \frac{1}{B_L} \sum \nabla \mathcal{L}_{\text{Reg}}(\tau; \tilde{\theta})$$.

2.  **Inner Loop (Variance Reduced Updates):**
    * Sample a "Small Batch" $\mathcal{D}\_S$ from the current policy $\pi\_{\theta_t}$.
    * Compute Importance Sampling (IS) weights $\rho(\tau)$.
    * Apply Dual-Clip truncation to $\rho(\tau)$ to prevent exploding gradients.
    * Construct the variance-reduced gradient direction:

$$
\mathbf{v}\_t = \frac{1}{B\_S} \sum\_{\tau \in \mathcal{D}\_S} \left[\nabla \mathcal{L}\_{\text{Reg}}(\tau; \theta\_t) - \rho(\tau) \nabla \mathcal{L}\_{\text{Reg}}(\tau; \tilde{\theta}) \right] + \hat{\mu}
$$

This approach decouples the variance of the reasoning reward from the model's structural updates, isolating the contribution of the recent parameter shift.

## 5. Normalized KL Formulations 

For completeness, the repository also supports normalized KL objectives, which are suitable for standard RLHF workflows.

| Regularization | Surrogate Loss (sampling $x\sim \pi_{\mathrm{old}}$) |
| :--- | :--- |
| **Forward KL** | $\mathbb{E}\left[ -w(x) R(x) - \beta \log \pi_\theta(x) \right]$ |
| **Reverse KL** | $\mathbb{E}\left[ w(x)\,(-R(x) + \beta \log w(x)) \right]$ |

## Citation

```bibtex
@article{zhang2025revisiting,
  title={Revisiting Variance Reduction in Policy Gradients for LLM Reinforcement Learning},
  author={Zhang, Yifan and Gu, Quanquan},
  year    = {2025},
  month   = {Dec},
  journal = {Github},
  url     = {https://yifanzhang-pro.github.io/Revisiting-SVRPG-LLM-RL}
}
```
