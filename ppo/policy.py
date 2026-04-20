"""
policy.py — ActorCriticPolicy supporting all three methods.

Methods:
  no_concept        — features → mlp_extractor → actor/critic
  vanilla_freeze    — features → FlexibleMultiTaskNetwork → mlp_extractor → actor/critic
  concept_actor_critic — features → ConceptActorCritic (GRU) → mlp_extractor → actor/critic

CNN feature extractor: NatureCNN for image observations, MLP for vector.
Dict observations are handled by extracting the 'images' key (primary modality).
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .networks import FlexibleMultiTaskNetwork, ConceptActorCritic


# ---------------------------------------------------------------------------
# NatureCNN  (DQN-style)
# ---------------------------------------------------------------------------

class NatureCNN(nn.Module):
    """
    CNN following the DQN / Nature paper architecture.
    Input:  [B, C, H, W]  (float, already normalized to [0,1])
    Output: [B, features_dim]
    """

    def __init__(self, obs_shape: tuple, features_dim: int = 512):
        super().__init__()
        n_input_channels = obs_shape[0]
        # For small images (< 100px), use stride=2 in first layer to preserve
        # fine spatial detail (e.g. LunarLander lander is ~12px at 84×84).
        # For larger images (CartPole 160×240, DynamicObstacles 160×160) keep stride=4.
        h = obs_shape[1] if len(obs_shape) > 1 else 84
        first_stride = 2 if h < 100 else 4
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=first_stride, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            cnn_out = self.cnn(dummy).shape[1]
        self.linear = nn.Sequential(nn.Linear(cnn_out, features_dim), nn.ReLU())
        self.features_dim = features_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


# ---------------------------------------------------------------------------
# MlpExtractor  (actor / value MLP heads after bottleneck)
# ---------------------------------------------------------------------------

class MlpExtractor(nn.Module):
    """Shared trunk + separate pi/vf heads."""

    def __init__(self, input_dim: int, net_arch: List[int] = None):
        super().__init__()
        if net_arch is None:
            net_arch = [64, 64]
        layers = []
        in_dim = input_dim
        for h in net_arch:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# ActorCriticPolicy
# ---------------------------------------------------------------------------

class ActorCriticPolicy(nn.Module):
    """
    Unified actor-critic policy for all three methods.

    Parameters
    ----------
    obs_shape    : tuple or dict of tuples
    n_actions    : int  (discrete action space size)
    method       : 'no_concept' | 'vanilla_freeze' | 'concept_actor_critic'
    task_types   : list of 'classification' | 'regression'  (one per concept)
    num_classes  : list of ints  (K per concept; 0 for regression)
    concept_dim  : total number of concepts
    features_dim : CNN output dimension
    net_arch     : hidden sizes for mlp_extractor
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        method: str,
        task_types: List[str],
        num_classes: List[int],
        concept_dim: int,
        temporal_concepts: Optional[List[int]] = None,
        temporal_encoding: str = "none",
        features_dim: int = 512,
        net_arch: Optional[List[int]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.obs_is_dict = isinstance(obs_shape, dict)
        self.n_actions = n_actions
        self.method = method
        self.task_types = task_types
        self.num_classes = num_classes
        self.concept_dim = concept_dim
        self.temporal_concepts = temporal_concepts
        self.temporal_encoding = temporal_encoding
        self.features_dim = features_dim
        self._device = torch.device(device) if isinstance(device, str) else device

        # ---- Feature extractor ----
        if self.obs_is_dict:
            # Use images key
            img_shape = obs_shape["images"]
            self.features_extractor = NatureCNN(img_shape, features_dim)
        elif len(obs_shape) >= 3:
            self.features_extractor = NatureCNN(obs_shape, features_dim)
        else:
            # Vector observations
            in_dim = int(np.prod(obs_shape))
            self.features_extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_dim, features_dim),
                nn.ReLU(),
            )
            self.features_extractor.features_dim = features_dim  # type: ignore

        # ---- Concept module ----
        self.concept_net: Optional[nn.Module] = None
        if method == "vanilla_freeze":
            self.concept_net = FlexibleMultiTaskNetwork(
                features_dim, task_types, num_classes, temporal_concepts=temporal_concepts
            )
            mlp_input_dim = concept_dim          # integer concept vector → policy
        elif method == "concept_actor_critic":
            self.concept_net = ConceptActorCritic(
                features_dim, task_types, num_classes,
                temporal_encoding=temporal_encoding,
                temporal_concepts=temporal_concepts,
            )
            mlp_input_dim = concept_dim
        else:
            # no_concept
            mlp_input_dim = features_dim

        # ---- MLP extractor ----
        arch = net_arch if net_arch is not None else [64, 64]
        self.mlp_extractor = MlpExtractor(mlp_input_dim, arch)
        latent_dim = self.mlp_extractor.output_dim

        # ---- Policy / value heads ----
        self.action_net = nn.Linear(latent_dim, n_actions)
        self.value_net  = nn.Linear(latent_dim, 1)

        # ---- Optimizers ----
        # optimizer: all parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        # optimizer_exclude_concept: parameters NOT in concept_net
        non_concept_params = [
            p for name, p in self.named_parameters()
            if "concept_net" not in name
        ]
        self.optimizer_exclude_concept = torch.optim.Adam(
            non_concept_params, lr=3e-4
        )

        # optimizer_concept_only: concept_net parameters only
        # Used by train_concept_actor_critic so the concept AC loss does not
        # corrupt mlp_extractor / action_net / value_net.
        if self.concept_net is not None:
            self.optimizer_concept_only = torch.optim.Adam(
                self.concept_net.parameters(), lr=3e-4
            )
        else:
            self.optimizer_concept_only = None

        # optimizer_concept_and_features: concept_net + features_extractor parameters
        # Used by concept_actor_critic training so that concept signals shape the
        # feature extractor (mirrors vanilla_freeze where concept supervision also
        # flows through features_extractor), without touching mlp_extractor /
        # action_net / value_net.
        if self.concept_net is not None:
            concept_and_feature_params = (
                list(self.features_extractor.parameters()) +
                list(self.concept_net.parameters())
            )
            self.optimizer_concept_and_features = torch.optim.Adam(
                concept_and_feature_params, lr=3e-4
            )
        else:
            self.optimizer_concept_and_features = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, obs) -> torch.Tensor:
        """
        obs: tensor [B, *obs_shape] or dict of tensors
        Returns [B, features_dim].
        """
        if self.obs_is_dict:
            imgs = obs["images"].float() / 255.0
            return self.features_extractor(imgs)
        else:
            x = obs.float()
            # Check if image (3+ dims besides batch)
            if x.ndim > 2:
                x = x / 255.0
            return self.features_extractor(x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (actions, values, log_prob, h_new).
        actions: sampled action [B]
        values:  [B]
        log_prob:[B]
        h_new:   [B, hidden_dim] or None
        """
        features = self.extract_features(obs)
        latent, h_new, _, _ = self._get_latent(features, h_prev)

        action_logits = self.action_net(latent)
        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)

        values = self.value_net(latent).flatten()
        return actions, values, log_prob, h_new

    def _get_latent(
        self,
        features: torch.Tensor,
        h_prev: Optional[torch.Tensor],
    ):
        """
        Returns (latent, h_new, c_t, concept_extras)
        concept_extras: (V_c, concept_dists) or (None, None)
        """
        h_new = None
        V_c = None
        concept_dists = None

        if self.method == "no_concept":
            latent = self.mlp_extractor(features)
            c_t = None
        elif self.method == "vanilla_freeze":
            c_t = self.concept_net(features)          # [B, concept_dim]
            latent = self.mlp_extractor(c_t)
        elif self.method == "concept_actor_critic":
            c_t, h_new, concept_dists, V_c = self.concept_net(features, h_prev)
            latent = self.mlp_extractor(c_t)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return latent, h_new, c_t, (V_c, concept_dists)

    # ------------------------------------------------------------------
    # evaluate_actions  (used during training)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs,
        actions: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Returns:
          concepts  [B, concept_dim] or None
          values    [B]
          log_prob  [B]
          entropy   [B]
          h_new     [B, hidden_dim] or None
          V_c       [B, 1] or None
          concept_dists: list or None
        """
        features = self.extract_features(obs)
        latent, h_new, c_t, (V_c, concept_dists) = self._get_latent(features, h_prev)

        action_logits = self.action_net(latent)
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(actions.long().flatten())
        entropy  = dist.entropy()

        values = self.value_net(latent).flatten()
        return c_t, values, log_prob, entropy, h_new, V_c, concept_dists

    # ------------------------------------------------------------------
    # predict_concepts
    # ------------------------------------------------------------------

    def predict_concepts(
        self,
        obs,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns (c_t, h_new)."""
        with torch.no_grad():
            features = self.extract_features(obs)
            _, h_new, c_t, _ = self._get_latent(features, h_prev)
        return c_t, h_new

    # ------------------------------------------------------------------
    # predict (action selection, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        obs,
        h_prev: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns (action, h_new)."""
        features = self.extract_features(obs)
        latent, h_new, _, _ = self._get_latent(features, h_prev)
        action_logits = self.action_net(latent)
        if deterministic:
            action = action_logits.argmax(dim=1)
        else:
            action = Categorical(logits=action_logits).sample()
        return action, h_new

    # ------------------------------------------------------------------
    # Update learning rate
    # ------------------------------------------------------------------

    def update_lr(self, lr: float) -> None:
        opts = [self.optimizer, self.optimizer_exclude_concept]
        if self.optimizer_concept_only is not None:
            opts.append(self.optimizer_concept_only)
        if self.optimizer_concept_and_features is not None:
            opts.append(self.optimizer_concept_and_features)
        for opt in opts:
            for pg in opt.param_groups:
                pg["lr"] = lr

    # ------------------------------------------------------------------
    # set_training_mode
    # ------------------------------------------------------------------

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)
