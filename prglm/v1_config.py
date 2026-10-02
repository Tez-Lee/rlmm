from dataclasses import asdict, dataclass


@dataclass
class V1Config:
    vocab_size: int = 256
    context: int = 32
    regions: int = 32
    region_size: int = 1024
    edges_per_node: int = 8
    gateways: int = 4
    active_nodes: int = 16
    initial_regions: int = 2
    max_cycles: int = 4
    d_model: int = 64
    edge_type: str = "ternary"
    recurrence: bool = True
    accumulation: bool = True
    fatigue: bool = True
    fatigue_strength: float = 0.5
    recovery: float = 0.8
    state_decay: float = 0.9
    accumulator_decay: float = 1.0
    seed: int = 42

    def validate(self):
        if not (1 < self.regions <= 256 and self.region_size >= self.active_nodes > 0):
            raise ValueError("invalid region or active-node count")
        if not (1 <= self.edges_per_node <= 16 and self.edges_per_node % 2 == 0):
            raise ValueError("edges_per_node must be even and at most 16")
        if not (1 <= self.gateways < self.regions and 1 <= self.initial_regions <= self.regions):
            raise ValueError("invalid gateways or initial regions")
        if self.edge_type not in ("ternary", "float") or self.max_cycles < 1:
            raise ValueError("invalid edge representation or cycle limit")
        if not 0 <= self.recovery <= 1 or not 0 <= self.accumulator_decay <= 1:
            raise ValueError("invalid decay")
        return self

    def dict(self):
        return asdict(self)
