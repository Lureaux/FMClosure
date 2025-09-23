large2(device) = (;

model = UNet(;
    nspace = grid.n,
    channels = [8, 8, 16, 16],
    nresidual = 4,
    t_embed_dim = 16,
    y_embed_dim = 16,
    device
),

name = "large2",

nsample = 150.,)
    
    
    
#     smalll() = UNet(;
#     nspace = grid.n,
#     channels = [8, 8, 16, 16],
#     nresidual = 4,
#     t_embed_dim = 16,
#     y_embed_dim = 16,
#     device,
# )