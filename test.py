from typing import Callable
def get_lambda(
    alpha: int,    
    epochs: list[int] | None,
) -> Callable[[int], float]:
    """Create lambda function for param scheduler."""
    if epochs is None:
        _epochs = []
    else:
        _epochs = epochs

    def scale(epoch: int) -> float:
        """Compute current scale factor using epoch."""
        factor = 1.0
        for e in _epochs:
            if epoch >= e:
                factor *= alpha
        return factor

    return scale
kfac_damping_alpha = 10.0
kfac_damping_decay = 1
damping_lambda=get_lambda(
        kfac_damping_alpha,
        kfac_damping_decay,
    )