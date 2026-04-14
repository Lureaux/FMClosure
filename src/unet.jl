silu(x) = @. x / (1 + exp(-x))

"Conv with periodic padding (`pad` on each side)."
CircularConv(args...; pad, kwargs...) =
    Chain(WrappedFunction(x -> pad_circular(x, pad; dims = 1)), Conv(args...; kwargs...))

"""
Upsample periodic field by a factor of 2.
The grids contain `n + 1` and `2n + 1` points, respectively.
The left and right boundary points overlap periodically, and
so the value of the input field in the right point is not
included in the input `x`.
"""
CircularUpsample() =
    WrappedFunction() do x
        n = size(x, 1)
        x = pad_circular(x, (0, 1); dims = 1) # Add redundant right point
        x = upsample_linear(x; size = 2 * n + 1)
        selectdim(x, 1, 1:(2*n)) # Remove redundant right point
    end

CircularConvTranspose(n, args...; stride = 2, kwargs...) =
    let
        @assert stride == 2 "This implementation only works for stride = 2."
        Chain(
            WrappedFunction(x -> pad_circular(x, (1 - n, n); dims = 1)),
            ConvTranspose((2n + 1,), args...; stride, kwargs...),
            WrappedFunction(x -> selectdim(x, 1, (n+1):(size(x, 1)-3n+1))),
        )
    end

function FourierEncoder(dim, device)
    @assert dim % 2 == 0
    half_dim = div(dim, 2)
    weights = randn(Float32, 1, half_dim)
    @compact(; weights) do t
        freqs = @. 2 * t * weights
        sin_embed = @. sqrt(2.0f0) * sinpi(freqs)
        cos_embed = @. sqrt(2.0f0) * cospi(freqs)
        output = hcat(sin_embed, cos_embed)
        @return output
    end
end

ResidualLayer(nspace, nchan, nt, ny) =
    @compact(;
        block1 = Chain(
            gelu,
            # LayerNorm((nspace, nchan)),
            BatchNorm(nchan),
            CircularConv((3,), nchan => nchan; pad = 1),
        ),
        block2 = Chain(
            gelu,
            # LayerNorm((nspace, nchan)),
            BatchNorm(nchan),
            CircularConv((3,), nchan => nchan; pad = 1),
        ),
        time_adapter = Chain(
            ReshapeLayer((nt,)),
            Dense(nt => nt, gelu),
            Dense(nt => nchan),
            ReshapeLayer((1, nchan)),
        ),
        y_adapter = Chain(
            CircularConv((3,), ny => ny, gelu; pad = 1),
            CircularConv((3,), ny => nchan; pad = 1),
        ),
    ) do (x, t_embed, y_embed)
        res = copy(x)

        # Initial conv block
        x = block1(x)

        # Add time embedding
        t_embed = time_adapter(t_embed)
        x = x .+ t_embed

        # Add y embedding (conditional embedding)
        y_embed = y_adapter(y_embed)
        x = x .+ y_embed

        # Second conv block
        x = block2(x)

        # Add back residual
        x = x .+ res

        @return x
    end

Encoder(nspace, nin, nout, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nspace, nin, nt, ny), nresidual),
        downsample = CircularConv((3,), nin => nout; stride = 2, pad = 1),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        x = downsample(x)
        @return x
    end

Midcoder(nspace, nchannel, nresidual, nt, ny) =
    @compact(;
        res_blocks = fill(ResidualLayer(nspace, nchannel, nt, ny), nresidual),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

Decoder(nspace, nin, nout, nresidual, nt, ny) =
    @compact(;
        upsample = CircularConvTranspose(3, nin => nout),
        res_blocks = fill(ResidualLayer(nspace, nout, nt, ny), nresidual),
    ) do (x, t_embed, y_embed)
        x = upsample(x)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

UNet(; nspace, channels, nresidual, t_embed_dim, y_embed_dim, device) =
    @compact(;
        init_conv = Chain(
            CircularConv((3,), 1 => channels[1]; pad = 1),
            # LayerNorm((nspace, channels[1])),
            BatchNorm(channels[1]),
            gelu,
        ),
        time_embedder = FourierEncoder(t_embed_dim, device),
        y_embedders = map(
            i -> CircularConv((3,), 1 => y_embed_dim, gelu; stride = 2^(i - 1), pad = 1),
            1:length(channels),
        ),
        encoders = map(
            i -> Encoder(
                div(nspace, 2^(i - 1)),
                channels[i],
                channels[i+1],
                nresidual,
                t_embed_dim,
                y_embed_dim,
            ),
            1:(length(channels)-1),
        ),
        decoders = map(
            i -> Decoder(
                div(nspace, 2^(i - 2)),
                channels[i],
                channels[i-1],
                nresidual,
                t_embed_dim,
                y_embed_dim,
            ),
            length(channels):-1:2,
        ),
        midcoder = Midcoder(
            div(nspace, 2^(length(channels) - 1)),
            channels[end],
            nresidual,
            t_embed_dim,
            y_embed_dim,
        ),
        final_conv = CircularConv((3,), channels[1] => 1; pad = 1, use_bias = false),
    ) do (x, t, y)
        # Embed t and y
        t_embed = time_embedder(t)
        # y_embed = y_embedder(y)

        # Initial convolution
        x = init_conv(x)

        residuals = ()
        y_embeds = ()

        # Encoders
        for (encoder, y_embedder) in zip(encoders, y_embedders)
            y_embed = y_embedder(y)
            x = encoder((x, t_embed, y_embed))
            residuals = residuals..., copy(x)
            y_embeds = y_embeds..., y_embed
        end

        # Midcoder
        y_embed = y_embedders[end](y)
        x = midcoder((x, t_embed, y_embed))

        # Decoders
        for decoder in decoders
            y_embeds..., y_embed = y_embeds
            residuals..., res = residuals
            x = x + res
            x = decoder((x, t_embed, y_embed))
        end

        # Final convolution
        x = final_conv(x)

        @return x
    end

function create_dataloader(grid, data, batchsize, rng)
    y, z = data
    y, z = reshape(y, grid.n, 1, :), reshape(z, grid.n, 1, :)
    y, z = (y, z) |> f32
    # z ./= grid.n
    DataLoader((y, z); batchsize, shuffle = true, partial = false, rng)
end

"Create periodic brownian bridge noise."
function brownian_periodic(z, sigma)
    T = eltype(z)
    nx, s... = size(z)
    colons = ntuple(Returns(:), length(s))
    u = randn(T, nx + 1, s...)
    u = cumsum(u; dims=1) 
    u .-= u[1]
    u .*= sigma/sqrt(T(nx))
    x = range(T(0), T(1), nx + 1)
    l = @. x * u[end:end, colons...] + (1-x) * u[1:1, colons...]
    v = u - l
    v = v[1:end-1, colons...]
    vmean = sum(v; dims=1) / nx
    v .-= vmean
    v
end

"Solve flow matching ODE/SDE with pseudo-time stepping and return the final state at pseudo-time 1."
function pseudo_timestepping(model, nsubstep, x, t, y, a, b, adot, bdot, sigma, info_i)
    h = 1.0f0 / nsubstep
    xfull = zeros(length(x), nsubstep+1)
    xfull[:,1] = x
    # print(x)
    for isub = 1:nsubstep # Pseudo-time stepping
        if info_i
            @info isub
        end
        model_u = model(x, t, y)
        score = @. (a(t) .* model_u - adot(t) .* x) ./ (b(t).^2 .* adot(t) - a(t) .* bdot(t) .* b(t))
        x += h * (model_u + score .* sigma(t).^2 /2) + sigma(t) .* sqrt(h) .* randn(size(model_u))
        xfull[:,isub+1] = x
        @. t += h
    end
    x, t
end

"Evaluate the flow matching model with pseudo-time stepping. The initial state is sampled from Gaussian or Brownian noise, and the final state at pseudo-time 1 is returned."
function model_eval(model, y, noise_type, a, b, nsubstep, sigma, sigma_brown, info_i, device)
    adot(t) = ForwardDiff.derivative(a, t)
    bdot(t) = ForwardDiff.derivative(b, t)
    t = fill(0.0f0, 1, 1, size(y, 3)) |> device

    if noise_type == "gaussian"
        x = randn!(similar(y))
    elseif noise_type == "brownian"
        x = brownian_periodic(similar(y), sigma_brown)
    else
        error("Unknown noise type: $noise_type")
    end

    x, t = pseudo_timestepping(model, nsubstep, x, t, y, a, b, adot, bdot, sigma, info_i)

    x
end

"""
Train an flow-matching ODE to predict next state (`z`) from current state (`y`).
The ODE has Gaussian initial contitions `x0` and evolve via `dx = model(x, t, y) dt`
from time 0 to 1.
The target trajectory `x` is a linear interpolation between `x0` and `z`.
"""
function train(; model, rng, nepoch, dataloader, opt, device, a, b, params = nothing)
    ps, st = 
    if isnothing(params)
        Lux.setup(rng, model) |> device
    else
        params |> device
    end
    train_state = Training.TrainState(model, ps, st, opt)
    loss = MSELoss()
    adot(t) = ForwardDiff.derivative(a, t)
    bdot(t) = ForwardDiff.derivative(b, t)
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(dataloader)
        y, z = batch |> device
        nsample = size(z, ndims(z))
        x0 = randn!(similar(z)) # Gaussian initial conditions
        T = eltype(z)
        # x0 = brownian_periodic(z, T(1.0)) |> device # Brownian initial conditions
        t = rand!(similar(z, 1, 1, nsample)) # Pseudo-times
        x = @. a(t) * z + b(t) * x0 # Linear interpolation
        u = @. adot(t) * z + bdot(t) * x0 # Linear conditional vector field
            # @show size(x) size(t) size(y) size(u) size(x0); error()
        # @show typeof(y) typeof(z) typeof(t) typeof(x) typeof(u); error()

        # A 4-Tuple containing:
        # 
        #   - `grads`: Computed Gradients.
        #   - `loss`: Loss from the objective function.
        #   - `stats`: Any computed statistics from the objective function.
        #   - `ts`: Updated Training State

        _, l, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, ((x, t, y), u), train_state)
        ibatch % 1 == 0 && @info "iepoch = $iepoch, ibatch = $ibatch, loss = $l"


    end
    ps_freeze = train_state.parameters
    st_freeze = train_state.states
    ps_freeze, st_freeze
end



"Get grid spacing."
dx(g::Grid) = g.l / g.n


"Call `f(args..., i)` for all grid indices `i`."
apply!(f, g::Grid, args) =
    Threads.@threads for i = 1:g.n
        @inbounds f(args..., i)
    end


# "Korteweg-de Vries equation right hand side."
# @inline function force!(f, u, g::Grid, _, i)
#     h = dx(g)
#     a = (u[i] + u[i-1|>g])^2 / 4
#     b = (u[i+1|>g] + u[i])^2 / 4
#     # b = u[i+1|>g]^2 / 2
#     # a = u[i-1|>g]^2 / 2
#     f[i] = 3 * (b - a) / h - (u[i+2|>g] / 2 - u[i+1|>g] + u[i-1|>g] - u[i-2|>g] / 2) / h^3
# end

"RK4 step adding the physical force function and the learned flow-matching model."
function rk4_con!(u, cache, grid, visc, dt, model, noise_type, a, b, nsubstep, sigma, sigma_brown, device)
    info_i = false
    v, k1, k2, k3, k4 = cache
    apply!(force!, grid, (k1, u, grid, visc))
    k1 += model_eval(model, reshape(u, :, 1, 1), noise_type, a, b, nsubstep, sigma, sigma_brown, info_i, device)
    @. v = u + dt / 2 * k1
    apply!(force!, grid, (k2, v, grid, visc))
    k2 += model_eval(model, reshape(v, :, 1, 1), noise_type, a, b, nsubstep, sigma, sigma_brown, info_i, device)
    @. v = u + dt / 2 * k2
    apply!(force!, grid, (k3, v, grid, visc))
    k3 += model_eval(model, reshape(v, :, 1, 1), noise_type, a, b, nsubstep, sigma, sigma_brown, info_i, device)
    @. v = u + dt * k3
    apply!(force!, grid, (k4, v, grid, visc))
    k4 += model_eval(model, reshape(v, :, 1, 1), noise_type, a, b, nsubstep, sigma, sigma_brown, info_i, device)
    @. u += dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
end

"Given initial condition `u`, simulate `nsample` trajectories using the flow matching model as closure model with `nsubstep * (ntime-1)` time steps with time step 'dt' and return the trajectories. First element is the initial condition."
function sim_data_con(; u, grid, params, nsubstep, ntime, dt, model, noise_type, a, b, nsubstep_pseudo, sigma, sigma_brown, device)
    outputs = zeros(grid.n, ntime)
    adaptive = isnothing(dt)
    cache = similar(u), similar(u), similar(u), similar(u), similar(u)
    outputs[:, 1] = u
    for itime = 2:ntime
        for isubstep = 1:nsubstep
            rk4_con!(u, cache, grid, params, dt, model, noise_type, a, b, nsubstep_pseudo, sigma, sigma_brown, device)
        end
        outputs[:, itime] = u
    end
    outputs
end
