# Burgers' equation model
model = UNet(;
    nspace = grid.n,
    channels = [8, 8],
    nresidual = 2,
    t_embed_dim = 32,
    y_embed_dim = 32,
    device,
) # 34,792 parameters

# KS equation model
m2 = UNet(;
    nspace = grid_les.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
) # 30,844 parameters
