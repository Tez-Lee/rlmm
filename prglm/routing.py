import torch


def route_step(active,route,origin,has_previous):
    """Move mass through LOCAL, ring NEIGHBOR, recorded-origin RETURN, OUTPUT.

    origin[b, destination, source] is the conditional previous-region distribution.
    """
    local=active*route[...,0]
    neighbor=active*route[...,1]
    returning=active*route[...,2]
    returns=origin*returning[...,None]  # source region, destination region
    next_mass=local+torch.roll(neighbor,1,dims=1)+returns.sum(1)
    neighbor_origins=torch.roll(torch.diag_embed(neighbor),1,dims=1)
    origin_mass=(origin*local[...,None])+neighbor_origins+returns.transpose(1,2)
    next_origin=origin_mass/next_mass.clamp_min(1e-8)[...,None]
    valid_mass=has_previous*local+torch.roll(neighbor,1,dims=1)+returns.sum(1)
    next_has_previous=valid_mass/next_mass.clamp_min(1e-8)
    return next_mass.clamp(0,1),next_origin,next_has_previous
