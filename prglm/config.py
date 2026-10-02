from dataclasses import dataclass, asdict


@dataclass
class Config:
    vocab_size: int = 2048
    context: int = 16
    regions: int = 4
    region_size: int = 32
    max_cycles: int = 3
    initial_regions: int = 2
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    edge_type: str = "ternary"
    stochastic_recurrence: bool = True
    fixed_recurrence_probability: float | None = None
    fatigue: bool = True
    fatigue_strength: float = 0.5
    recovery: float = 0.8
    local_router: bool = True
    seed: int = 42

    def validate(self):
        assert self.region_size > 0 and self.regions > 1
        assert self.max_cycles >= 1 and 1 <= self.initial_regions <= self.regions
        assert self.edge_type in {"ternary", "float"}
        assert 0 <= self.recovery <= 1
        if self.fixed_recurrence_probability is not None:
            assert 0 <= self.fixed_recurrence_probability <= 1
        return self

    def dict(self):
        return asdict(self)
