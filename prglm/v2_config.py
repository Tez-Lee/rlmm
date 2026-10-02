"""Topology parameters fixed before any mutable-topology results."""
from dataclasses import dataclass
from .v1_config import V1Config


@dataclass
class V2Config(V1Config):
    edges_per_node: int = 16
    topology_mode: str = 'adaptive'
    mutation_interval: int = 50
    mutation_probability: float = .25
    exploration_epsilon: float = .10
    exploration_policy: str = 'uniform'
    hotness_temperature: float = .5
    hotness_ema_decay: float = .95
    maximum_rewires_per_interval: int = 8
    usefulness_weight: float = 1.
    visit_weight: float = .25
    congestion_weight: float = 1.
    probation_steps: int = 100
    probation: bool = True
    gradient_credit: bool = False

    def validate(self):
        super().validate()
        if self.topology_mode not in ('static','uniform','adaptive'):
            raise ValueError('invalid topology mode')
        if self.exploration_policy not in ('uniform','low_visit'):
            raise ValueError('invalid exploration policy')
        if self.mutation_interval < 1 or self.maximum_rewires_per_interval < 0 or self.probation_steps < 0:
            raise ValueError('invalid mutation interval/count')
        if not all(0 <= x <= 1 for x in (self.mutation_probability,self.exploration_epsilon,self.hotness_ema_decay)):
            raise ValueError('invalid mutation probability/EMA')
        if self.hotness_temperature <= 0:
            raise ValueError('temperature must be positive')
        return self
