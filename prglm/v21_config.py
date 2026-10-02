"""Preregistered v2.1 core corrections; immutable static topology."""
from dataclasses import dataclass
from .v1_config import V1Config


@dataclass
class V21Config(V1Config):
    minimum_gateway_hops: int = 0  # Diagnostic probe only, per walker.

    def validate(self):
        super().validate()
        if self.region_size % self.active_nodes:
            raise ValueError('context pooling requires region_size divisible by active_nodes')
        if not 0 <= self.minimum_gateway_hops < self.max_cycles:
            raise ValueError('probe needs a processing cycle after required gateway hops')
        if self.minimum_gateway_hops and not self.recurrence:
            raise ValueError('gateway probe is defined only for default recurrence')
        return self
