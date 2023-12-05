import torch.nn as nn
import torch
import numpy as np
from ...subNets.transformers_encoder.transformer import TransformerEncoder


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the temporal resolution decreases
        self.conv1 = nn.Conv1d(32, channels[0], 3, stride=1, padding=1, bias=False)
        self.attention_1 = TransformerEncoder(embed_dim=channels[0],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.conv2_cond = nn.Conv1d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.attention_2 = TransformerEncoder(embed_dim=channels[1],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.conv3_cond = nn.Conv1d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.attention_3 = TransformerEncoder(embed_dim=channels[2],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv1d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.conv4_cond = nn.Conv1d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.attention_4 = TransformerEncoder(embed_dim=channels[3],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the temporal resolution increases
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.tconv4_cond = nn.ConvTranspose1d(channels[3], channels[2], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.attention_t4 = TransformerEncoder(embed_dim=channels[2],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose1d(channels[2] + channels[2], channels[1], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.tconv3_cond = nn.ConvTranspose1d(channels[2], channels[1], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.attention_t3 = TransformerEncoder(embed_dim=channels[1],
                                              num_heads=8,
                                              layers=2,
                                              attn_dropout=0.0,
                                              relu_dropout=0.0,
                                              res_dropout=0.0,
                                              embed_dropout=0.0,
                                              attn_mask=True)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose1d(channels[1] + channels[1], channels[0], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.tconv2_cond = nn.ConvTranspose1d(channels[1], channels[0], 3, stride=2, padding=1, bias=False, output_padding=1)
        self.attention_t2 = TransformerEncoder(embed_dim=channels[0],
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose1d(channels[0] + channels[0], 32, 3, stride=1, padding=1)
        self.tconv1_cond = nn.ConvTranspose1d(channels[0], 32, 3, stride=1, padding=1)
        self.attention_t1 = TransformerEncoder(embed_dim=32,
                                               num_heads=8,
                                               layers=2,
                                               attn_dropout=0.0,
                                               relu_dropout=0.0,
                                               res_dropout=0.0,
                                               embed_dropout=0.0,
                                               attn_mask=True)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, condition=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        if condition is not None:
            h1_with_cond = self.attention_1(h1.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h1 += h1_with_cond.permute(1, 2, 0)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        if condition is not None:
            condition = self.conv2_cond(condition)  # align condition with h2
            h2_with_cond = self.attention_2(h2.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h2 += h2_with_cond.permute(1, 2, 0)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        if condition is not None:
            condition = self.conv3_cond(condition)  # align condition with h3
            h3_with_cond = self.attention_3(h3.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h3 += h3_with_cond.permute(1, 2, 0)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        if condition is not None:
            condition = self.conv4_cond(condition)  # align condition with h4
            h4_with_cond = self.attention_4(h4.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h4 += h4_with_cond.permute(1, 2, 0)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        if condition is not None:
            condition = self.tconv4_cond(condition)
            h_with_cond = self.attention_t4(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        if condition is not None:
            condition = self.tconv3_cond(condition)
            h_with_cond = self.attention_t3(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        if condition is not None:
            condition = self.tconv2_cond(condition)
            h_with_cond = self.attention_t2(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))
        if condition is not None:
            condition = self.tconv1_cond(condition)
            h_with_cond = self.attention_t1(h.permute(2, 0, 1), condition.permute(2, 0, 1), condition.permute(2, 0, 1))
            h += h_with_cond.permute(1, 2, 0)

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None]
        return h

def loss_fn(model, x, marginal_prob_std, condition=None, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None]
  if condition is not None:
    perturbed_condition = condition + z * std[:, None, None]
    score = model(perturbed_x, random_t, perturbed_condition)
  else:
    score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None] + z)**2, dim=(1,2)))
  return loss


#Define the Euler-Maruyama sampler
## The number of sampling steps.
num_steps = 100
def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           batch_size=64,
                           num_steps=num_steps,
                           device='cuda',
                           condition=None,
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 32, 48, device=device) \
             * marginal_prob_std(t)[:, None, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)  # batch_time_step = t
            if condition is not None:
                perturbed_condition = condition + torch.randn(batch_size, 32, 48, device=device) * marginal_prob_std(batch_time_step)[:, None, None]
                mean_x = x + (g ** 2)[:, None, None] * score_model(x, batch_time_step, perturbed_condition) * step_size
            else:
                mean_x = x + (g ** 2)[:, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)
            # Do not include any noise in the last sampling step.
    return mean_x