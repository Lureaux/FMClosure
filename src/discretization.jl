"Grid of domain length `l` with `n` points."
struct Grid{T}
    l::T
    n::Int
end

# Call `g(i)` to shift an index `i` periodically 
@inline (g::Grid)(i) = mod1(i, g.n)

"Get grid spacing."
dx(g::Grid) = g.l / g.n

"""
The left point is always at zero.
The (`n + 1`-th) right point is at `l`, but it is not included since it
is periodically redundant.
"""
points(g::Grid) = range(0, g.l, g.n + 1)[1:(end-1)]

"Call `f(args..., i)` for all grid indices `i`."
apply!(f, g::Grid, args) =
    Threads.@threads for i = 1:g.n
        @inbounds f(args..., i)
    end

# "Burgers equation right hand side."
@inline function force!(f, u, g::Grid, (; visc), i)
    h = dx(g)

    g_i = (u[i]^2 + u[i] * u[i+1|>g] + u[i+1|>g]^2) / 6
    g_imin1 = (u[i-1|>g]^2 + u[i-1|>g] * u[i] + u[i]^2) / 6
    conv = (g_i - g_imin1) / h
    diff = visc * (u[i+1|>g] - 2 * u[i] + u[i-1|>g]) / h^2
    f[i] = -conv + diff
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

function closureterm(u, g_dns::Grid, g_les::Grid)
    visc=0.005
    f = zero(u)
    apply!(force!, g_dns, (f, u, g_dns, (; visc)))
    ubar = filter_u(u, g_dns.l, g_dns.n, g_les.n, "gaussian", 0.02)
    fbar1 = filter_u(f, g_dns.l, g_dns.n, g_les.n, "gaussian", 0.02)
    fbar2 = zero(fbar1)
    apply!(force!, g_les, (fbar2, ubar, g_les, (; visc)))
    c = fbar1 - fbar2
    c
end

function forward_euler!(u, f, grid, visc, dt)
    apply!(force!, grid, (f, u, grid, visc))
    @. u += dt * f
end

function rk4!(u, cache, grid, visc, dt)
    v, k1, k2, k3, k4 = cache
    apply!(force!, grid, (k1, u, grid, visc))
    @. v = u + dt / 2 * k1
    apply!(force!, grid, (k2, v, grid, visc))
    @. v = u + dt / 2 * k2
    apply!(force!, grid, (k3, v, grid, visc))
    @. v = u + dt * k3
    apply!(force!, grid, (k4, v, grid, visc))
    @. u += dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
end



# function rk4!(f!, u, cache, dt)
#     v, k1, k2, k3, k4 = cache
#     f!(k1, u)
#     @. v = u + dt / 2 * k1
#     f!(k2, v)
#     @. v = u + dt / 2 * k2
#     f!(k3, v)
#     @. v = u + dt * k3
#     f!(k4, v)
#     @. u += dt * (k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6)
# end

propose_timestep(u, g::Grid, visc) = min(dx(g) / maximum(abs, u), dx(g)^2 / visc)

# function randomfield(g::Grid, kpeak, rng)
#     amp = sqrt(4 / kpeak / 3 / sqrt(π))
#     k = 0:div(g.n, 2)
#     c = @. amp * (k / kpeak)^2 * exp(-(k / kpeak)^2 / 2 + 2π * im * rand(rng))
#     irfft(c * g.n, g.n)
# end

function randomfield(g::Grid, kpeak, total_energy, rng)
    K = div(g.n, 2)
    k = 0:K
    c = @. k^4 * exp(-2(k / kpeak)^2 + 2π * im * rand(rng))
    a = irfft(c, g.n)
    Δx = 2*π / g.n
    u₀ = a * sqrt(2*total_energy / sum(abs2, a) / Δx)
    u₀
end


function create_data(; grid, params, nsample, nsubstep, ntime, dt, rng)
    inputs = zeros(grid.n, ntime, nsample)
    outputs = similar(inputs)
    adaptive = isnothing(dt)
    # f!(du, u) = apply!(force!, grid, (du, u, grid, params))
    for isample = 1:nsample
        @show isample
        u = randomfield(grid, 10.0, 10.0, rng)
        cache = similar(u), similar(u), similar(u), similar(u), similar(u)
        for itime = 1:ntime
            # @show (isample, itime)
            inputs[:, itime, isample] = u
            for isubstep = 1:nsubstep
                # forward_euler!(u, cache, grid, params, dt)
                rk4!(u, cache, grid, params, dt)
                # rk4!(f!, u, cache, dt)
            end
            outputs[:, itime, isample] = u
        end
    end
    @. outputs -= inputs # Let the difference be the target
    inputs, outputs
end



function cutoff(u_hat, n_dns, n_les)
    u_bar_hat = u_hat[1:div(n_les, 2)+1].*n_les./n_dns
    u_bar_hat
end

# Filter kernel
function spectral_cutoff(u, n_dns, n_les)
    u_hat = rfft(u)
    u_bar_hat = cutoff(u_hat, n_dns, n_les)
    u_bar = irfft(u_bar_hat, n_les)
    u_bar
end

function g(k, Δ, filter_type, L)
    if filter_type == "spectral"
        K_cutoff = div(1/Δ, 2)
        g_k = zeros(Complex{Float64}, length(k))
        j = 1
        for i in k
            if i <= K_cutoff
                g_k[j] = 1.0
            end
            j += 1
        end
    elseif filter_type == "gaussian"
        g_k = exp.( -π^2 * (k).^2 * Δ^2 / 6)
    elseif filter_type == "top_hat"
        g_k = zeros(Complex{Float64}, length(k))
        j = 1
        for i in k
            if i == 0
                g_k[j] = 1.0
            else
                g_k[j] = sin(π * (i / L) * Δ) / (π * (i/ L) * Δ)
            end
        j += 1
        end
    end   
    g_k 
end

function filter_u(u, L, n_dns, n_les, filter_type, Δ = 0.02)
    K_dns = div(n_dns, 2)
    k = 0:K_dns
    u_hat = rfft(u)
    u_bar_hat = g(k, Δ, filter_type, L) .* u_hat
    u_bar_dns = irfft(u_bar_hat, n_dns)
    u_bar = spectral_cutoff(u_bar_dns, n_dns, n_les)
    u_bar
end  

function spectrum(u, grid)
    uhat = rfft(u)[2:end]
    n = grid.n
    k = rfftfreq(n) * n
    k = k[2:end]
    s = abs2.(uhat) / 2n^2
    k, s
end

function create_data_dns(; grid_dns, grid_les, params, nsample, nsubstep, ntime, dt, rng)
    n_dns = grid_dns.n
    n_les = grid_les.n
    L = grid_dns.l
    dt_les = nsubstep * dt
    
    inputs = zeros(n_dns, ntime, nsample)
    inputs_les = zeros(n_les, ntime, nsample)
    outputs = zeros(n_les, ntime, nsample)
    outputs_new = zeros(n_les, ntime, nsample)
    adaptive = isnothing(dt)
    for isample = 1:nsample
        @show isample
        u_dns = randomfield(grid_dns, 10.0, 10.0, rng)
        u_les = filter_u(u_dns, L, n_dns, n_les, "gaussian")
        cache_dns = similar(u_dns), similar(u_dns), similar(u_dns), similar(u_dns), similar(u_dns)
        cache_les = similar(u_les), similar(u_les), similar(u_les), similar(u_les), similar(u_les)
        for itime = 1:ntime
            # @show (isample, itime)
            inputs[:, itime, isample] = u_dns
            inputs_les[:, itime, isample] = u_les
            u_dns_bar_new = filter_u(u_dns, L, n_dns, n_les, "gaussian")
            for isubstep = 1:nsubstep
                # forward_euler!(u, cache, grid, params, dt)
                rk4!(u_dns, cache_dns, grid_dns, params, dt) 
                # rk4!(u_les, cache_les, grid_les, params, dt)
                # rk4!(u_dns_bar_new, cache_les, grid_les, params, dt)
            end
            rk4!(u_les, cache_les, grid_les, params, dt_les)
            rk4!(u_dns_bar_new, cache_les, grid_les, params, dt_les)
            u_dns_bar = filter_u(u_dns, L, n_dns, n_les, "gaussian")
            diff = u_dns_bar - u_les
            diff_new = u_dns_bar - u_dns_bar_new
            outputs[:, itime, isample] = diff
            outputs_new[:, itime, isample] = diff_new
        end
    end
    # @. outputs -= inputs # Let the difference be the target
    inputs, outputs, inputs_les, outputs_new
end

function sim_data(; u, grid, params, nsubstep, ntime, dt)
    outputs = zeros(grid.n, ntime)
    adaptive = isnothing(dt)
    cache = similar(u), similar(u), similar(u), similar(u), similar(u)
    outputs[:, 1] = u
    for itime = 2:ntime
        # @show (isample, itime)
        for isubstep = 1:nsubstep
            # forward_euler!(u, cache, grid, params, dt)
            rk4!(u, cache, grid, params, dt)
        end
        outputs[:, itime] = u
    end
    outputs
end


function create_data_con(; grid_dns, grid_les, params, nsample, nsubstep, ntime, dt, rng)
    # First entry of u_bars is initial condition
    n_dns = grid_dns.n
    n_les = grid_les.n
    L = grid_dns.l
    
    u_bars = zeros(n_les, ntime, nsample)
    closures = zeros(n_les, ntime, nsample)
    adaptive = isnothing(dt)

    for isample = 1:nsample
        @show isample
        u = randomfield(grid_dns, 10.0, 10.0, rng)
        cache = similar(u), similar(u), similar(u), similar(u), similar(u) # v, k1, k2, k3, k4
        for itime = 1:ntime
            # @show (isample, itime)
            u_bar = filter_u(u, L, n_dns, n_les, "gaussian", 0.02)
            u_bars[:, itime, isample] = u_bar
            closure = closureterm(u, grid_dns, grid_les)
            closures[:, itime, isample] = closure
            for isubstep = 1:nsubstep
                # forward_euler!(u, cache, grid, params, dt)
                rk4!(u, cache, grid_dns, params, dt) 
            end
        end
    end
    u_bars, closures
end


