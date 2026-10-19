import torch


def finite_nonzero_gradients(named_parameters):
    gradients = [(name, parameter.grad) for name, parameter in named_parameters if parameter.grad is not None]
    assert gradients
    finite = torch.stack([torch.isfinite(gradient).all() for _, gradient in gradients]).cpu().tolist()
    assert all(finite), [name for (name, _), valid in zip(gradients, finite) if not valid]
    nonzero = torch.stack([torch.count_nonzero(gradient).ne(0) for _, gradient in gradients]).cpu().tolist()
    return {name for (name, _), active in zip(gradients, nonzero) if active}
