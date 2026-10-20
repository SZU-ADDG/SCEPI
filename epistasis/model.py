from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    projection: int = 64
    hidden: int = 256
    representation: str = "full"
    geometry: bool = False
    embedding_dim: int = 1280
    pair_form: str = "contrast"
    pair_single_input: str = "calibrated"
    pair_mask_input: str = "auto"


def mlp(width, hidden):
    return nn.Sequential(
        nn.Linear(width, hidden), nn.GELU(),
        nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1),
    )


class EndpointEncoder(nn.Module):
    def __init__(self, config, cross):
        super().__init__()
        self.config = config
        self.cross = cross and config.representation in {"full"}
        self.has_wt = config.representation != "identity"
        self.has_delta = config.representation in {"full", "no_cross"}
        self.width = 40
        self.projections = nn.ModuleDict()
        for key, enabled in [("wt", self.has_wt), ("delta", self.has_delta), ("cross", self.cross)]:
            if enabled:
                self.projections[key] = nn.Linear(config.embedding_dim, config.projection, bias=False)
                self.width += config.projection
        if self.has_wt:
            self.width += 1

    def forward(self, data, active=None):
        if active is None:
            active = data["mut_aa"] != data["wt_aa"]
        active_float = active.to(data["wt"].dtype)
        mutant = torch.where(active, data["mut_aa"], data["wt_aa"])
        parts = [
            F.one_hot(data["wt_aa"], num_classes=20).to(data["wt"].dtype),
            F.one_hot(mutant, num_classes=20).to(data["wt"].dtype),
        ]
        if self.has_wt:
            parts += [self.projections["wt"](data["wt"]), (data["score"] * active_float).unsqueeze(-1)]
        if self.has_delta:
            parts.append(self.projections["delta"](data["delta"] * active_float.unsqueeze(-1)))
        if self.cross:
            # 另一端点恢复参考状态时，其引起的环境变化归零。
            other_active = active_float.flip(-1)
            parts.append(self.projections["cross"](data["cross"] * other_active.unsqueeze(-1)))
        return torch.cat(parts, dim=-1)


class SingleEffect(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EndpointEncoder(config, cross=False)
        self.network = mlp(self.encoder.width, config.hidden)

    def forward(self, data):
        active = data["mut_aa"] != data["wt_aa"]
        variant = self.encoder(data, active)
        reference = self.encoder(data, torch.zeros_like(active))
        return (self.network(variant) - self.network(reference)).squeeze(-1)


class PairEffect(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.pair_form in {"contrast", "direct"}
        assert config.pair_single_input in {"calibrated", "predicted"}
        assert config.pair_mask_input in {"auto", "observed", "zero"}
        self.config = config
        self.encoder = EndpointEncoder(config, cross=True)
        endpoint_width = self.encoder.width + 2 + int(config.geometry)
        relation_width = 2 + (17 if config.geometry else 0)
        self.network = mlp(endpoint_width * 2 + relation_width, config.hidden)
        if config.geometry:
            self.register_buffer("rbf_centers", torch.linspace(0, 30, 16))

    def symmetric_function(self, data, effects, observed, active):
        encoded = self.encoder(data, active)
        active_float = active.to(encoded.dtype)
        parts = [
            encoded,
            (effects * active_float).unsqueeze(-1),
            (observed.to(encoded.dtype) * active_float).unsqueeze(-1),
        ]
        if self.config.geometry:
            parts.append(data["confidence"].unsqueeze(-1))
        endpoint = torch.cat(parts, dim=-1)
        distance = data["sequence_gap"]
        relation = [torch.stack([torch.log1p(distance), distance / data["sequence_length"]], dim=-1)]
        if self.config.geometry:
            ca_distance = data["ca_distance"]
            rbfs = torch.exp(-0.5 * ((ca_distance.unsqueeze(-1) - self.rbf_centers) / 2.0) ** 2)
            relation += [rbfs, (ca_distance > 30).to(encoded.dtype).unsqueeze(-1)]
        relation = torch.cat(relation, dim=-1)
        ordered = self.network(torch.cat([endpoint[:, 0], endpoint[:, 1], relation], dim=-1))
        swapped = self.network(torch.cat([endpoint[:, 1], endpoint[:, 0], relation], dim=-1))
        return ((ordered + swapped) * 0.5).squeeze(-1)

    def forward(self, data, effects, observed):
        active = data["mut_aa"] != data["wt_aa"]
        if self.config.pair_form == "direct":
            return self.symmetric_function(data, effects, observed, active)
        first = active.clone()
        first[:, 1] = False
        second = active.clone()
        second[:, 0] = False
        reference = torch.zeros_like(active)
        return (
            self.symmetric_function(data, effects, observed, active)
            - self.symmetric_function(data, effects, observed, first)
            - self.symmetric_function(data, effects, observed, second)
            + self.symmetric_function(data, effects, observed, reference)
        )


class PhenotypeMap(nn.Module):
    def __init__(self, flexible=False, components=20):
        super().__init__()
        self.flexible = flexible
        if flexible:
            self.b_raw = nn.Parameter(torch.full((components,), -2.0))
            self.c_raw = nn.Parameter(torch.full((components,), -1.0))
            self.d = nn.Parameter(torch.linspace(-3, 3, components))
            self.tau_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))

    def normalized(self, value):
        if not self.flexible:
            return value
        b = F.softplus(self.b_raw)
        c = F.softplus(self.c_raw)
        centered = torch.tanh(value.unsqueeze(-1) * c + self.d) - torch.tanh(self.d)
        denominator = 1 + torch.sum(b * c * (1 - torch.tanh(self.d).square()))
        return (value + torch.sum(b * centered, dim=-1)) / denominator

    def forward(self, value):
        if not self.flexible:
            return value
        return F.softplus(self.tau_raw) * self.normalized(value)


class EpistasisModel(nn.Module):
    def __init__(self, single, pair, mapping, omit_pair=False):
        super().__init__()
        self.single = single
        self.pair = pair
        self.mapping = mapping
        self.omit_pair = omit_pair

    def freeze_single(self):
        self.single.requires_grad_(False)

    def calibrated_effects(self, data, observed_values, observed_mask, omega):
        predicted = self.single(data)
        # 有效性由观察标记确定，真实零效应保留为有效标签。
        active = data["mut_aa"] != data["wt_aa"]
        selected = torch.where(observed_mask & active, observed_values, predicted)
        return predicted + omega * (selected - predicted)

    def forward(self, data, observed_values, observed_mask, omega=0.0, remove_interaction=False):
        effects = self.calibrated_effects(data, observed_values, observed_mask, omega)
        if remove_interaction or self.omit_pair:
            interaction = torch.zeros_like(effects[:, 0])
        else:
            pair_config = self.pair.config
            pair_effects = self.single(data) if pair_config.pair_single_input == "predicted" else effects
            # auto 保留既有完整模型和联合输入移除对照的行为。
            zero_mask = pair_config.pair_mask_input == "zero" or (
                pair_config.pair_mask_input == "auto" and pair_config.pair_single_input == "predicted"
            )
            pair_mask = torch.zeros_like(observed_mask) if zero_mask else observed_mask
            interaction = self.pair(data, pair_effects, pair_mask)
        additive = effects.sum(dim=-1)
        predicted = self.mapping(additive + interaction)
        single_predictions = self.mapping(effects)
        epsilon_global = self.mapping(additive) - single_predictions.sum(dim=-1)
        epsilon_specific = predicted - self.mapping(additive)
        epsilon = predicted - single_predictions.sum(dim=-1)
        if (remove_interaction or self.omit_pair) and not self.mapping.flexible:
            epsilon = torch.zeros_like(predicted)
        return {
            "phenotype": predicted, "single": single_predictions,
            "epsilon": epsilon,
            "interaction": interaction, "epsilon_global": epsilon_global,
            "epsilon_specific": epsilon_specific,
        }


class ObservationModel(nn.Module):
    def __init__(self, initial_sigma=1.0):
        super().__init__()
        self.sigma_raw = nn.Parameter(torch.tensor(math.log(math.expm1(initial_sigma))))

    @property
    def sigma(self):
        return F.softplus(self.sigma_raw)

    def nll(self, predicted, lower, upper, point):
        predicted = predicted.double()
        lower = lower.double()
        upper = upper.double()
        sigma = self.sigma.double()
        result = torch.empty_like(predicted)
        result[point] = (
            0.5 * ((lower[point] - predicted[point]) / sigma).square()
            + torch.log(sigma) + 0.5 * math.log(2 * math.pi)
        )
        interval = ~point
        assert (lower[interval] < upper[interval]).all(), "Invalid observation interval"
        left = interval & torch.isneginf(lower) & torch.isfinite(upper)
        right = interval & torch.isfinite(lower) & torch.isposinf(upper)
        unbounded = interval & torch.isneginf(lower) & torch.isposinf(upper)
        finite = interval & torch.isfinite(lower) & torch.isfinite(upper)
        assert (point | left | right | unbounded | finite).all()
        result[left] = -torch.special.log_ndtr((upper[left] - predicted[left]) / sigma)
        result[right] = -torch.special.log_ndtr((predicted[right] - lower[right]) / sigma)
        result[unbounded] = predicted[unbounded] * 0
        lo = (lower[finite] - predicted[finite]) / sigma
        hi = (upper[finite] - predicted[finite]) / sigma
        positive = lo > 0
        log_large = torch.where(positive, torch.special.log_ndtr(-lo), torch.special.log_ndtr(hi))
        log_small = torch.where(positive, torch.special.log_ndtr(-hi), torch.special.log_ndtr(lo))
        log_probability = log_large + torch.log(-torch.expm1(log_small - log_large))
        result[finite] = -log_probability
        assert torch.isfinite(result).all(), "Non-finite observation likelihood"
        return result
