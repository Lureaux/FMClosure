# Import packages
# This is just a hack for "go to definition" to work in editor.
if false
    include("src/FMClosure.jl")
    using .FMClosure
end


using CairoMakie
using Lux
using MLUtils
using NNlib
using Optimisers
using LinearAlgebra
using FFTW
# using WGLMakie
using Random
using Zygote
using ForwardDiff
using Statistics
using KernelDensity
using OptimalTransport
using Distances
using Tulip
using FMClosure

outdir = joinpath(@__DIR__, "output") |> mkpath

visc = 0.005

# Define problem
burgers(n, visc) = (; grid = Grid(2π, n), params = (; visc))
kdv(n) = (;
    grid = Grid(30.0, n),
    params = (;), # No params for KdV
)


# Plot solution
let
    (; grid, params) = burgers(1024, visc)
    # (; grid, params) = kdv(256)
    ustart = randomfield(grid, 10.0, 10.0, Xoshiro(0))
    u = copy(ustart)
    # cache = similar(u) # (forward_euler)
    cache = similar(u), similar(u), similar(u), similar(u), similar(u) # (RK4)
    t = 0.0
    tstop = 0.2
    while t < tstop
        # dt = 0.3 * propose_timestep(u, grid, visc)
        dt = 1e-3
        dt = min(dt, tstop - t) # Don't overstep
        # forward_euler!(u, cache, grid, params, dt)
        rk4!(u, cache, grid, params, dt)
        t += dt
    end
    x = points(grid)
    fig = Figure(; size = (400, 340))
    ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "u")
    lines!(ax, x, ustart; label = "Initial")
    lines!(ax, x, u; label = "Final")
    Legend(
        fig[0, 1],
        ax;
        tellwidth = false,
        orientation = :horizontal,
        framevisible = false,
    )
    rowgap!(fig.layout, 5)
    # save("$outdir/solution.pdf", fig; backend = CairoMakie)
    fig
end


# Create dataset
# (; grid, params) = burgers(2048, 2e-3)

(; grid, params) = burgers(256, visc)
data = create_data(;
    grid,
    params,
    nsample = 100,
    ntime = 200,
    nsubstep = 10,
    dt = 1e-3,
    rng = Xoshiro(0),
);
data[2]

data_test = create_data(;
    grid,
    params,
    nsample = 1,
    ntime = 100,
    nsubstep = 10,
    dt = 1e-3,
    rng = Xoshiro(1),
);


# Load DNS data
(; grid, params) = burgers(1024, visc)
grid_dns = grid
(; grid, params) = burgers(128, visc)
grid_les = grid

data_dns_burgers =  create_data_dns(;
    grid_dns,
    grid_les,
    params,
    nsample = 500,
    ntime = 20,
    nsubstep = 10,
    dt = 1e-3,
    rng = Xoshiro(0),
);

data_dns_burgers[3][:,end,1]


data_dns[4]
# [1]: u simulated on fine grid  (DNS (time t))
# [2]: difference between fine and coarse grid (Difference between fDNS and LES)
# [3]: u simulated on coarse grid (LES)
# [4]: difference between u_bar_sim and u_bar
data_dns[2]

itime = 1
fig = Figure()
x = points(grid_dns)
ax = Axis(fig[1, 1])
lines!(ax, x, data_dns[1][:, itime, 1])
lines!(ax, x, data_dns[1][:, itime+1, 1])
fig


using JLD2

data_dns = let
filename = "mydata_dns.jld2"
load(filename, "data_dns")
end

data_dns_kdv = let
filename = "mydata_dns_kdv.jld2"
load(filename, "data_dns_kdv")
end

data_kdv_small = let
filename = "mydata_kdv_small.jld2"
load(filename, "data_kdv_small")
end

data_kdv_small_train = let
filename = "mydata_kdv_small_train.jld2"
load(filename, "data_kdv_small_train")
end

data_kdv_small_test = let
filename = "mydata_kdv_small_test.jld2"
load(filename, "data_kdv_small_test")
end

data_dns[1]

# Show one input-output pair
let
    isample = 2
    itime = 20
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    # lines!(ax, points(grid_dns), data_dns[1][:, itime, isample])
    lines!(ax, points(grid_les),   data_dns[2][:, itime-1, isample] +data_dns[3][:, itime, isample])
    # lines!(ax, points(grid_les), data_dns[3][:, itime, isample])
    # save("$outdir/fdns_end.png", fig; backend = CairoMakie)
    fig
end


# n_rows_dns, n_cols, n_samples = size(data_dns_kdv[1])
# n_rows_les, _ = size(data_dns_kdv[2])
# n_select = 200
# n_select_train = 140


# # Prepare output tensor
# data_small_1 = Array{Float32}(undef, n_rows_dns, n_select, n_samples)
# data_small_2 = Array{Float32}(undef, n_rows_les, n_select, n_samples)
# data_small_3 = Array{Float32}(undef, n_rows_les, n_select, n_samples)
# data_small_4 = Array{Float32}(undef, n_rows_les, n_select, n_samples)

# # # Random selection per sample
# # Random.seed!(123)  # optional for reproducibility

# for i in 1:n_samples
#     shuffled_index = shuffle(1:n_cols)
#     selected_cols = sort(shuffled_index[1:n_select])
#     selected_cols_train = selected_cols[1:n_select_train]
#     selected_cols_test = selected_cols[(n_select_train+1):end]
#     data_small_1[:, :, i] = data_dns_kdv[1][:, selected_cols, i]
#     data_small_2[:, :, i] = data_dns_kdv[2][:, selected_cols, i]
#     data_small_3[:, :, i] = data_dns_kdv[3][:, selected_cols, i]
#     data_small_4[:, :, i] = data_dns_kdv[4][:, selected_cols, i]
# end
# data_kdv_small = (data_small_1, data_small_2, data_small_3, data_small_4)
# data_kdv_small_train = (data_small_1[:, 1:n_select_train, :], data_small_2[:, 1:n_select_train, :], data_small_3[:, 1:n_select_train, :], data_small_4[:, 1:n_select_train, :])
# data_kdv_small_test = (data_small_1[:, (n_select_train+1):end, :], data_small_2[:, (n_select_train+1):end, :], data_small_3[:, (n_select_train+1):end, :], data_small_4[:, (n_select_train+1):end, :])



# Show two successive states   
let
    isample = 1
    itime = 10
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, x, data[1][:, itime, isample])
    lines!(ax, x, data[1][:, itime+1, isample])
    save("$outdir/states.pdf", fig; backend = CairoMakie)
    fig
end

# Show one input-output pair
let
    isample = 2
    itime = 1
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, x, data[1][:, itime, isample])
    lines!(ax, x, data[2][:, itime, isample])
    fig
end

# Define model
device = gpu_device()
model = UNet(;
    nspace = grid.n,
    channels = [8, 8],
    nresidual = 2,
    t_embed_dim = 32,
    y_embed_dim = 32,
    device,
)
# model, name = FMClosure.large2(device)
# model, name = FMClosure.small(device)

const πf = Float32(π)

# Linear
a(t) = t
b(t) = 1 .- t.^2
# adot(t) = ones(size(t))
# bdot(t) = -1*ones(size(t))

# # GVP
# a(t) = sin((πf/2)*t)
# b(t) = cos((πf/2)*t)
# adot(t) = @. (πf/2) * cos((πf/2)*t)
# bdot(t) = @. -(πf/2) * sin((πf/2)*t)

adot(t) = ForwardDiff.derivative(a, t)
bdot(t) = ForwardDiff.derivative(b, t)

data_dns = data_dns_burgers

# Prepare DNS simulated data with full next snapshot as target
data = let
    n_dns = grid_dns.n
    n_les = grid_les.n
    L = grid_dns.l
    
    inputs = zeros(grid_les.n, size(data_dns[1], 2)-1, size(data_dns[1], 3))
    outputs = zeros(grid_les.n, size(data_dns[1], 2)-1, size(data_dns[1], 3))
    for isample = 1:size(data_dns[1], 3)
        for itime = 1:size(data_dns[1], 2)-1
            inputs[:, itime, isample] = filter_u(data_dns[1][:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
            outputs[:, itime, isample] = filter_u(data_dns[1][:, itime+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
        end
    end
    inputs, outputs
end


# Prepare DNS simulated data continuous
data_dns_bar = similar(data_dns[4])
closures = similar(data_dns[4])
for isample = 1:size(data_dns[4], 3)
    for itime = 1:size(data_dns[4], 2)
        data_dns_bar[:, itime, isample] = filter_u(data_dns[1][:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
        closures[:, itime, isample] = closureterm(data_dns[1][:, itime, isample], grid_dns, grid_les)
    end
end
# data = (data_dns_bar, data_dns[4])
data = (data_dns_bar, closures)
grid


# # Train model
ps_freeze, st_freeze = train(;
    model,
    rng = Xoshiro(0),
    nepoch = 20,
    dataloader = create_dataloader(grid, data, 64, Xoshiro(0)),
    opt = AdamW(1.0f-3),  # Set weight decay \lambda maybe to 10^-4
    device,
    a,
    b,
    params = (ps_freeze, st_freeze),
);


# Load/save trained model
using JLD2
ic_type = "brownian"
# filename = "myparameters_linear_batchnorm_cont_brownian.jld2"
filename = "parameters/burgers_discsmall70_brownian.jld2"
# jldsave(filename; ps_freeze, st_freeze)
# ps_freeze, st_freeze = load(filename, "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))



# Define noise schedule

sigma(t) = 0.0f0 * ones(size(t))
# sigma(t) = sqrt.(0.1f0 * (1 .- t))
# sigma(t) = sqrt.(a(t))
# sigma(t) = 1*sin.(π*t)
# sigma(t) = 2*(adot(t) .*a(t) - bdot(t) .* a(t).^2 ./ b(t))

noise_type = "gaussian"
noise_type = "brownian"
sigma_brown = 1.0


# Method 1: Direct prediction of u(t_{n+1})
# Plot one prediction
nt = 20
data_test = create_data(;
    grid = grid_dns,
    params,
    nsample = 1,
    ntime = nt+1,
    nsubstep = 10,
    dt = 1e-3,
    rng = Xoshiro(4),
    );

let
    isample = 1
    itime = 1
    y_data, z_data = data_test
    y = reshape(filter_u(y_data[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02), :, 1, 1) |> f32 |> device
    z = reshape(filter_u(y_data[:, nt+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02), :, 1, 1) |> f32 |> device
    nsubstep = 10

    x = copy(y)
    for i = 1:nt
        x = x + model_eval(unet, x, noise_type , a, b, nsubstep, sigma, sigma_brown, false, device)
    end

    fig = Figure()
    ax = Axis(fig[1, 1])
    input = y[:] |> cpu_device()
    target = z[:] |> cpu_device()
    prediction = x[:] |> cpu_device()
    lines!(ax, points(grid), target; label = "Target")
    lines!(ax, points(grid), prediction; label = "Prediction")
    axislegend(ax)
    # save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    display(fig)

    k, s = spectrum(target, grid_les)
    kbar, sbar = spectrum(prediction, grid_les)
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="k",
        ylabel="S(k)",
        xscale=log10,
        yscale=log10,
        title = "Energy spectrum of u at final time",
    )
    lines!(ax, k, s; label = "Energy true")
    lines!(ax, kbar, sbar; label = "Energy prediction")
    # ylims!(ax, 1e-16, 1e+1)
    axislegend(ax; position = :lb)
    # save("spectrum_cutoff.png", fig)
    # ylims!(ax, 1e-10, 1e+1)
    display(fig)

end

let
    itime = 1
    y_data, z_data = data_test
    z = reshape(filter_u(y_data[:, 20+1, 1], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02), :, 1, 1) |> f32 |> device
    k, s = spectrum(z, grid_les)
    s_fdns_avg = similar(s)
    s_fm_avg = similar(s)
    for isample = 1:100
        y = reshape(filter_u(y_data[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02), :, 1, 1) |> f32 |> device
        z = reshape(filter_u(y_data[:, 20+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02), :, 1, 1) |> f32 |> device
        k, s = spectrum(z, grid_les)
        s_fdns_avg = similar(s)
        s_fm_avg = similar(s)

        nsubstep = 10

        x = copy(y)
        for i = 1:20
            x = model_eval(unet, x, noise_type , a, b, nsubstep, sigma, sigma_brown, false, device)
        end

        input = y[:] |> cpu_device()
        target = z[:] |> cpu_device()
        prediction = x[:] |> cpu_device()

        k, s_fdns = spectrum(target, grid_les)
        k, s_fm = spectrum(prediction, grid_les)
        s_fdns_avg .+= s_fdns
        s_fm_avg .+= s_fm

    end
    s_fdns_avg ./= 100
    s_fm_avg ./= 100
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="k",
        ylabel="S(k)",
        xscale=log10,
        yscale=log10,
        title = "Energy spectrum of u at final time averaged over 100 simulations",
    )
    lines!(ax, k, s_fdns_avg; label = "Energy true")
    lines!(ax, k, s_fm_avg; label = "Energy prediction")
    ylims!(ax, 1e-16, 1e+1)
    axislegend(ax; position = :lb)
    # save("spectrum_cutoff.png", fig)
    # ylims!(ax, 1e-10, 1e+1)
    display(fig)
end






# Plug FM model back into physical time stepping loop
let
    isample = 1
    inputs, _ = data_test
    ntime = 10
    y = reshape(inputs[:, 1, isample], :, 1, 1) |> f32 |> device
    nsubstep = 10
    for itime = 1:ntime # Physical time stepping
        @show itime
         x = model_eval(unet, y, noise_type , a, b, nsubstep, sigma, sigma_brown, false, device)
        @. y += x # x is the physical step
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    input = inputs[:, 1, isample]
    target = inputs[:, ntime+1, isample]
    prediction = y[:] |> cpu_device()
    # lines!(ax, points(grid), input; label = "Input")
    lines!(ax, points(grid), target; label = "Target")
    lines!(ax, points(grid), prediction; label = "Prediction")
    axislegend(ax)
    save("$outdir/prediction.png", fig; backend = CairoMakie)
    fig
end


# Method 2: Prediction of error from DNS/LES simulation \overline{u(t_{n+1})} - RK4(\overline{u(t_n)})   
# Plot one prediction
let
    isample = 1
    itime = 1
    y, z = data
    y = reshape(y[:, itime, isample], :, 1, 1) |> f32 |> device
    z = reshape(z[:, itime, isample], :, 1, 1) |> f32 |> device

    nsubstep = 10

    x = model_eval(unet, y, noise_type, a, b, nsubstep, sigma, sigma_brown, true, device)

    input = y[:] |> cpu_device()
    input_next_mat = sim_data(; u = input, 
        grid = grid_les, 
        params, 
        nsubstep = 1, 
        ntime = 20, 
        dt = 10*1e-3)
    input_next = input_next_mat[:,2]
    target = z[:] |> cpu_device()
    prediction = x[:] |> cpu_device()

    fig = Figure()
    ax = Axis(fig[1, 1])
    # lines!(ax, points(grid), input; label = "Input")
    # lines!(ax, points(grid), target; label = "Target")
    # lines!(ax, points(grid), prediction; label = "Prediction")
    lines!(ax, points(grid), input_next + target; label = "Target")
    lines!(ax, points(grid), input_next + prediction; label = "Prediction")
    axislegend(ax)
    save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    fig
end

# Plug FM model back into physical time stepping loop
let
    isample = 1
    inputs, target_exact = data
 
    ntime = 20
    y = reshape(inputs[:, 1, isample], :, 1, 1) |> f32 |> device

    nsubstep = 10
    for itime = 1:ntime # Physical time stepping
        @show itime
        
        x = model_eval(unet, y, noise_type, a, b, nsubstep, sigma, sigma_brown, false, device)

        y_mat = sim_data(
            # f!; 
            ;
            u = y, 
            grid = grid_les, 
            params, 
            nsubstep = 1000, 
            ntime = 2, 
            dt = 1e-5)
        y = y_mat[:,2]
        y = reshape(y, :, 1, 1)
        @. y += x # x is the physical step
        # @. y += target_exact[:, itime, isample] # x is the physical step
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    input = inputs[:, 1, isample]
    target = inputs[:, ntime+1, isample]
    prediction = y[:] |> cpu_device()
    # lines!(ax, points(grid), input; label = "Input")
    lines!(ax, points(grid), target; label = "Target")
    lines!(ax, points(grid), prediction; label = "Prediction")
    axislegend(ax)
    save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    fig
end

# Method 3: Continuous closure term \overline{F(u(t_n))} - F(\overline{u(t_n)})
# Plot one prediction
nt = 200
Int(200.0)
t0 = time_ns()
data_test = create_data(;
    grid = grid_dns,
    params,
    nsample = 1,
    ntime = nt+1,
    nsubstep = 1,
    dt = 1e-3,
    rng = Xoshiro(1),
    );
t1 = time_ns()
elapsed = (t1 - t0) / 1e9
println("Elapsed time: $elapsed seconds")
data_test[1][:,1,1]'
let
    factor = 10
    nt_fm = Int(nt/factor)
    isample = 1
    itime = 1
    y = filter_u(data_test[1][:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
    y = reshape(y, :, 1, 1) |> f32 |> device
    target = filter_u(data_test[1][:, end, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)

    nsubstep_pseudo = 1
    
    input = y[:] |> cpu_device()
    input_copy = copy(input)
    input_copy_disc = copy(input)
    input_copy_discdiff = copy(input)

    time_fmcon = 0.0
    t0 = time_ns()
    input_next_mat = sim_data_con(; u = input_copy, 
        grid = grid_les,
        params,
        nsubstep = 1, 
        ntime = nt_fm+1, 
        dt = factor*1e-3 ,
        model = unet, noise_type, a, b, nsubstep_pseudo, sigma, sigma_brown, device)
    t1 = time_ns()
    elapsed = (t1 - t0) / 1e9
    println("Elapsed time: $elapsed seconds")
    input_next = input_next_mat[:,end]
    print(norm(input_next - target)/norm(target))
    energy_target = sum(abs2, target)/(2*grid_les.n)
    energy_next = sum(abs2, input_next)/(2*grid_les.n)
    print("Energy target: $energy_target, energy prediction: $energy_next")
    # prediction = x[:] |> cpu_device()

    fig = Figure()
    ax = Axis(fig[1, 1])
    # lines!(ax, points(grid), input; label = "Input")
    lines!(ax, points(grid_les), target; label = "True target")
    lines!(ax, points(grid_les), input_next; label = "Prediction")
    axislegend(ax)
    ylims!(ax, -3, 3)
    # save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    display(fig)

    k, s = spectrum(target, grid_les)
    kbar, sbar = spectrum(input_next, grid_les)
    s = max.(s, 1e-14)
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="k",
        ylabel="S(k)",
        xscale=log10,
        yscale=log10,
        title = "Energy spectrum of u₀ and filtered u₀ at t=0.0 ",
    )
    lines!(ax, k, s; label = "Energy u₀")
    lines!(ax, kbar, sbar; label = "Energy filtered u₀")
    lines!(ax, k, k -> 100k^-2; label = "k^-2")
    ylims!(ax, 1e-16, 1e+1)
    axislegend(ax; position = :lb)
    # save("spectrum_cutoff.png", fig)
    # ylims!(ax, 1e-10, 1e+1)
    display(fig)

    norm_fm = zeros(nt_fm+1)
    for t = 0:nt_fm
        snap_filt = filter_u(data_test[1][:, factor*t+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
        norm_fm[t+1] = norm(input_next_mat[:,t+1] - snap_filt) / norm(snap_filt)
    end
    fig = Figure()
    ax = Axis(fig[1, 1];
        xlabel="t",
        ylabel="Relative error",
        title = "Relative error with filtered DNS",
    )
    lines!(range(0.0, 0.2, step=factor*1e-3), norm_fm; label = "Flow Matching")
    axislegend(ax; position = :lt)
    display(fig)
end



# Average over 100 simulations of FM
# Define discrete model
model = UNet(;
    nspace = grid.n,
    channels = [8, 8],
    nresidual = 2,
    t_embed_dim = 32,
    y_embed_dim = 32,
    device,
)
ps_freeze, st_freeze = load("parameters/burgers_cont_brownian.jld2", "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))

model_disc = UNet(;
    nspace = grid.n,
    channels = [16, 16],
    nresidual = 8,
    t_embed_dim = 16,
    y_embed_dim = 16,
    device,
)
ps_freeze_disc, st_freeze_disc = load("parameters/burgers_discsmall50_brownian.jld2", "ps_freeze", "st_freeze");
unet_disc = (x, t, y) -> first(model_disc((x, t, y), ps_freeze_disc, Lux.testmode(st_freeze_disc)))

model_discdiff = UNet(;
    nspace = grid.n,
    channels = [16, 16],
    nresidual = 8,
    t_embed_dim = 16,
    y_embed_dim = 16,
    device,
)
ps_freeze_discdiff, st_freeze_discdiff = load("parameters/burgers_discdiffsmall50_brownian.jld2", "ps_freeze", "st_freeze");
unet_discdiff = (x, t, y) -> first(model_discdiff((x, t, y), ps_freeze_discdiff, Lux.testmode(st_freeze_discdiff)))


nsample = 100
nt= 200
times = zeros(4)
t0 = time_ns()
test = create_data(;
    grid = grid_les,
    params,
    nsample = nsample,
    ntime = 20+1,
    nsubstep = 1,
    dt = 1e-2,
    rng = Xoshiro(1),
    );
t1 = time_ns()
(t1 - t0) / 1e9 ./ nsample
times[1] += (t1 - t0) / 1e9

factor = 10
nt_fm = Int(nt/factor)
norm_fm_avg = zeros(nt_fm+1)
norm_fmdisc_avg = zeros(nt_fm+1)
norm_fmdiscdiff_avg = zeros(nt_fm+1)
s_fdns_avg = zeros(Int(grid_les.n/2))
s_fm_avg = zeros(Int(grid_les.n/2))
s_fmdisc_avg = zeros(Int(grid_les.n/2))
s_fmdiscdiff_avg = zeros(Int(grid_les.n/2))
s_fdns_avg_avg = zeros(Int(grid_les.n/2))
s_fm_avg_avg = zeros(Int(grid_les.n/2))
s_fmdisc_avg_avg = zeros(Int(grid_les.n/2))
s_fmdiscdiff_avg_avg = zeros(Int(grid_les.n/2))


for isample = 1:nsample
    itime = 1
    y = filter_u(data_test[1][:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
    y = reshape(y, :, 1, 1) |> f32 |> device
    target = filter_u(data_test[1][:, end, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)

    nsubstep_pseudo = 1
    
    input = y[:] |> cpu_device()
    input_copy = copy(input)
    input_copy_disc = copy(input)
    input_copy_discdiff = copy(input)

    time_fmcon = 0.0
    t0 = time_ns()
    input_next_mat = sim_data_con(; u = input_copy, 
        grid = grid_les,
        params,
        nsubstep = 1, 
        ntime = nt_fm+1, 
        dt = factor*1e-3 ,
        model = unet, noise_type, a, b, nsubstep_pseudo, sigma, sigma_brown, device)
    t1 = time_ns()
    times[2] += (t1 - t0) / 1e9

    sol_disc = zeros(grid_les.n, nt_fm+1)
    sol_disc[:, 1] = input_copy_disc
    sol_discdiff = zeros(grid_les.n, nt_fm+1)
    sol_discdiff[:, 1] = input_copy_discdiff
    
    nsubstep = 10
    t0 = time_ns()
    for i = 1:nt_fm
        current = reshape(sol_disc[:, i], :, 1, 1) |> f32 |> device
        sol_disc[:, i+1] = model_eval(unet_disc, current, noise_type , a, b, nsubstep, sigma, sigma_brown, false, device)
    end
    t1 = time_ns()
    times[3] += (t1 - t0) / 1e9
    t0 = time_ns()
    for i = 1:nt_fm
        current = reshape(sol_discdiff[:, i], :, 1, 1) |> f32 |> device
        sol_discdiff[:, i+1] = sol_discdiff[:, i] + model_eval(unet_discdiff, current, noise_type , a, b, nsubstep, sigma, sigma_brown, false, device)
    end
    t1 = time_ns()
    times[4] += (t1 - t0) / 1e9

    k, s = spectrum(target, grid_les)

    norm_fm = zeros(nt_fm+1)
    norm_fmdisc = zeros(nt_fm+1)
    norm_fmdiscdiff = zeros(nt_fm+1)
    for t = 0:nt_fm
        snap_filt = filter_u(data_test[1][:, factor*t+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
        norm_fm[t+1] = norm(input_next_mat[:,t+1] - snap_filt) / norm(snap_filt)
        norm_fmdisc[t+1] = norm(sol_disc[:,t+1] - snap_filt) / norm(snap_filt)
        norm_fmdiscdiff[t+1] = norm(sol_discdiff[:,t+1] - snap_filt) / norm(snap_filt)
        s_fdns = 0
        s_fm = 0
        s_fmdisc = 0
        s_fmdiscdiff = 0
        k, s_fdns_sum = spectrum(snap_filt, grid_les)
        k, s_fm_sum = spectrum(input_next_mat[:,t+1], grid_les)
        k, s_fmdisc_sum = spectrum(sol_disc[:,t+1], grid_les)
        k, s_fmdiscdiff_sum = spectrum(sol_discdiff[:,t+1], grid_les)
        # print(size(s_fdns_sum))
        s_fdns_avg_avg .+= s_fdns_sum
        s_fm_avg_avg .+= s_fm_sum
        s_fmdisc_avg_avg .+= s_fmdisc_sum
        s_fmdiscdiff_avg_avg .+= s_fmdiscdiff_sum
        if t == nt_fm
            k, s_fdns_avg_sum = spectrum(snap_filt, grid_les)
            k, s_fm_avg_sum = spectrum(input_next_mat[:,t+1], grid_les)
            s_fdns_avg .+= s_fdns_avg_sum
            s_fm_avg .+= s_fm_avg_sum
            k, s_fmdisc_avg_sum = spectrum(sol_disc[:,t+1], grid_les)
            s_fmdisc_avg .+= s_fmdisc_avg_sum
            k, s_fmdiscdiff_avg_sum = spectrum(sol_discdiff[:,t+1], grid_les)
            s_fmdiscdiff_avg .+= s_fmdiscdiff_avg_sum
        end
    end
    norm_fm_avg .+= norm_fm 
    norm_fmdisc_avg .+= norm_fmdisc
    norm_fmdiscdiff_avg .+= norm_fmdiscdiff

    # fig = Figure()
    # ax = Axis(fig[1, 1])
    # # input = y[:] |> cpu_device()
    # # target = z[:] |> cpu_device()
    # # prediction = x[:] |> cpu_device()
    # lines!(ax, points(grid), target; label = "Target")
    # lines!(ax, points(grid), input_next_mat[:,end]; label = "Prediction continuous")
    # lines!(ax, points(grid), sol_disc[:,end]; label = "Prediction discrete full")
    # lines!(ax, points(grid), sol_discdiff[:,end]; label = "Prediction discrete difference")
    # axislegend(ax)
    # # save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    # display(fig)

end
norm_fm_avg ./= nsample
norm_fmdisc_avg ./= nsample
norm_fmdiscdiff_avg ./= nsample
s_fdns_avg ./= nsample
s_fm_avg ./= nsample
s_fdns_avg_avg ./= nsample*(nt_fm+1)
s_fm_avg_avg ./= nsample*(nt_fm+1)
s_fmdisc_avg ./= nsample
s_fmdisc_avg_avg ./= nsample*(nt_fm+1)
s_fmdiscdiff_avg ./= nsample
s_fmdiscdiff_avg_avg ./= nsample*(nt_fm+1)

k, _ = spectrum(filter_u(data_test[1][:, end, 1], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02), grid_les)

let
    fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel="k",
            ylabel="S(k)",
            xscale=log10,
            yscale=log10,
            title = "Energy spectrum at final time T",
        )
        lines!(ax, k, s_fdns_avg; label = "Filtered DNS")
        lines!(ax, k, s_fm_avg; label = "FM Continuous")
        lines!(ax, k, s_fmdisc_avg; label = "FM Discrete Full")
        lines!(ax, k, s_fmdiscdiff_avg; label = "FM Discrete Difference")
        # ylims!(ax, 1e-16, 1e+1)
        axislegend(ax; position = :lb)
        # save("Burgers_spectrum_FM_avg.pdf", fig)
        # ylims!(ax, 1e-10, 1e+1)
    display(fig)
end

times./100

let
    fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel="k",
            ylabel="S(k)",
            xscale=log10,
            yscale=log10,
            title = "Energy spectrum averaged over time",
        )
        lines!(ax, k, s_fdns_avg_avg; label = "Filtered DNS")
        lines!(ax, k, s_fm_avg_avg; label = "FM Continuous")
        lines!(ax, k, s_fmdisc_avg_avg; label = "FM Discrete Full")
        lines!(ax, k, s_fmdiscdiff_avg_avg; label = "FM Discrete Difference")
        # ylims!(ax, 1e-16, 1e+1)
        axislegend(ax; position = :lb)
        # save("Burgers_spectrum_FM_avg_avg.pdf", fig)
        # ylims!(ax, 1e-10, 1e+1)
    display(fig)
end

let
    fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel="t",
            ylabel="Relative error",
            title = "Relative error with filtered DNS",
        )
        lines!(range(0.0, 0.2, step=factor*1e-3), norm_fm_avg; label = "FM Continuous", color=Cycled(2))
        lines!(range(0.0, 0.2, step=factor*1e-3), norm_fmdisc_avg; label = "FM Discrete Full", color=Cycled(3))
        lines!(range(0.0, 0.2, step=factor*1e-3), norm_fmdiscdiff_avg; label = "FM Discrete Difference", color=Cycled(4))
        axislegend(ax; position = :lt)
        ylims!(ax, 0, 0.25)
        # save("Burgers_rel_err_FM.pdf", fig)
    display(fig)
end


data_test = create_data(;
    grid = grid_dns,
    params,
    nsample = 1,
    ntime = 100,
    nsubstep = 50,
    dt = 1e-4,
    rng = Xoshiro(1),
    );
k, s = spectrum(data_test[1][:,end,1], grid_dns)
s'
isample = 1
itime = 1
y = filter_u(data_test[1][:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "spectral", 0.000001)
y = reshape(y, :, 1, 1) |> f32 |> device
target = filter_u(data_test[1][:, end, isample], grid_dns.l, grid_dns.n, grid_les.n, "spectral", 0.000001)

nsubstep_pseudo = 10
    
input = y[:] |> cpu_device()
input_copy = copy(input)
input_next_mat = sim_data_con(; u = input_copy, 
    grid = grid_les,
    params,
    nsubstep = 1, 
    ntime = 100, 
    dt = 5e-3,
    model = unet, noise_type, a, b, nsubstep_pseudo, sigma, sigma_brown, device)

data_test[1]






s = 1
d = "toto"
function g(x, s, d)
    println(d)
    x + s
end


f(2)

rk(x -> g(x, s, d), u, dt)


# (Periodic) Brownian motion with mean zero
u = 256 + 1 |> randn |> cumsum |> x -> 0.1x
x = range(0, 1, 256 + 1)
u = randn(256 + 1, s...)
u = cumsum(u; dims=1) 
u .-= u[1]
u .*= 1.0/sqrt(256)
l = @. x * u[end] + (1-x) * u[1]
v = u - l
nx, s... = size(x[1:end-1])
colons = ntuple(Returns(:), length(s))
vmean = sum(v[1:end-1, colons...]; dims=1) / 256
mean(brownian_periodic(similar(v[1:end-1]), 1.0))
fft(v[1:end-1].-vmean)
fft(randomfield(grid, 10.0, Xoshiro(0)))
randomfield(grid, 10.0, Xoshiro(0))
# u .-= u[1]
let fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, x[1:end], u[1:end]; label = "ϵ")
    lines!(ax, x[1:end], l[1:end]; label = "Linear interpolation")
    lines!(ax, x[1:end], v[1:end]; label = "ϵ periodic")
    lines!(ax, x[1:end], v[1:end] .- mean(v[1:end-1]); label = "ϵ periodic with zero mean")
    # lines!(ax, x[1:end-1], v[1:end-1] .- vmean; label = "u - linear - mean")
    axislegend(ax)
    # save("$outdir/brown_noise_per.png", fig; backend = CairoMakie)
    fig
end      

let fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, x[1:end-1], brownian_periodic(x[1:end-1], 1.0); label = "Periodic Brownian Motion")
    axislegend(ax)
    # save("$outdir/brown_noise.png", fig; backend = CairoMakie)
    fig
end


# Square wave and its filtered version
function u_sum(x, N, L)
    ω = 2* π / L  
    sum(1:N) do i
        j = 2i-1
        (2/π) * sin(j * ω * x) / j
    end
end
L = 1.0
N = 1024
Nbar = N
x = range(00., L, N + 1)
y = @. u_sum(x, 5000, L)
y = vec(1.0*(0.0 .< x.-0.25 .< 0.5)).-0.5 

y_bar = filter_u(y, L, N+1, Nbar+1, "spectral", (0.5)^5)
y_bar_spectral = spectral_cutoff(y, N+1, Nbar+1)
y_bar
let fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, x, vec(y); label = "Truth")
    # lines!(ax, range(0.0, L, Nbar + 1), y_bar)
    lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/(2*0.01)); label = "1 nonzero")
    lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/(2*1)); label = "2 nonzero")
    # lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/(2*3)); label = "4 nonzero")
    lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/(2*9)); label = "10 nonzero")
    # lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/(2*39)); label = "40 nonzero")
    # lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/6))
    # lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/8))
    # lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/10))
    # lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/12))
    # lines!(ax, range(0.0, L, Nbar + 1), filter_u(y, L, N+1, Nbar+1, "spectral", 1/30))
    # lines!(ax, range(0.0, L, Nbar + 1), y_bar_spectral; label = "y_bar_spectral")
    axislegend(ax)
    # save("$outdir/sqwave_fourier.png", fig; backend = CairoMakie)
    fig
end
rfft(y)[3]
rfft(y_bar)[9]
rfft(y)[3] - rfft(y_bar)[3]
div(1/(1/(2*30)), 2)

# Energy distribution of closure term with KL divergence
energies = zeros(100)
energies_eles = zeros(100)
n_les = 32
itime = 38
for i = 1:100
    # k, spec = spectrum(data[1][:,end,i], grid_les)
    filtered_u = filter_u(data_dns[1][:,i,itime], grid_dns.l, grid_dns.n, n_les, "spectral", 0.00001)
    energy = sum(abs2, filtered_u)/(2*n_les)
    energies[i] = energy
    # spec_mean = sum(spec)
    # energies[i] = spec_mean
end
for i = 1:100
    u_sim = sim_data(; u = filter_u(data_dns[1][:,i,1], grid_dns.l, grid_dns.n, n_les, "spectral", 0.00001), 
        grid = Grid(30.0, n_les), 
        params, 
        nsubstep = 10, 
        ntime = itime, 
        dt = 1e-3)
    filtered_u = u_sim[:,end]
    energy = sum(abs2, filtered_u)/(2*n_les)
    energies_eles[i] = energy
end

let
    nsubstep_pseudo = 10
    isample = 1
    u_sim = sim_data(; u = filter_u(data_dns[1][:,isample,1], grid_dns.l, grid_dns.n, grid_les.n, "spectral", 0.00001), 
        grid = grid_les, 
        params, 
        nsubstep = 10, 
        ntime = itime, 
        dt = 1e-3)
        # model = unet, noise_type, a, b, nsubstep_pseudo, sigma, sigma_brown, device)
    filtered_u_eles = u_sim[:,end]
    @show sum(abs2, filtered_u_eles)/(2*grid_les.n)
    fig = Figure()
    x = points(Grid(30.0, grid_les.n))
    ax = Axis(fig[1, 1])
    # lines!(ax, points(Grid(30.0, 1024)), data_dns[1][:,isample,38])
    lines!(ax, points(Grid(30.0, grid_les.n)), filter_u(data_dns[1][:,isample,itime], grid_dns.l, grid_dns.n, grid_les.n, "spectral", 0.00001))
    lines!(ax, x, filtered_u_eles)
    # lines!(ax, x, filtered_u_eles)
    fig
end


lines(energies)
ylims!(0, 2.1)
current_figure()
let
    kde_energy = kde(energies)
    points = range(0.0, 1.0, length=512)  
    # lines(points, pdf(kde_energy, points), label="KDE")
    lines(kde_energy.x, kde_energy.density, label="KDE")
    lines!(kde(energies_eles).x, kde(energies_eles).density, label="KDE")
    xlims!(0, 1)
    ylims!(0, 25)
    current_figure()
end

aa, bb = extrema(energies)
(bb-aa)/mean(energies)
k, spec = spectrum(data[1][:,end,41], grid_les)
fig = Figure()
ax = Axis(fig[1, 1];
    xlabel="k",
    ylabel="S(k)",
    xscale=log10,
    yscale=log10,
    title = "",
)
lines!(ax, k, spec)
fig
mean(spec)


function find_KL_range(a, b, tol_per, npoints)
    ka = kde(a)
    kb = kde(b)
    tol_a = tol_per*maximum(ka.density)
    tol_b = tol_per*maximum(kb.density)
    ileft_a = findfirst(>(tol_a), ka.density) 
    ileft_b = findfirst(>(tol_b), kb.density) 
    iright_a = findlast(>(tol_a), ka.density) 
    iright_b = findlast(>(tol_b), kb.density) 

    left = max(ka.x[ileft_a], kb.x[ileft_b])
    right = min(ka.x[iright_a], kb.x[iright_b])
    if left >= right
        error("No overlapping support between the two distributions.")
    end
    points = range(left, right, npoints)
    points
end

function KL(a, b, tol_per = 1e-5, npoints = 2048)
    ka = kde(a)
    kb = kde(b)
    points = find_KL_range(a, b, tol_per, npoints)
    pa = pdf(ka, points)
    pb = pdf(kb, points)
    # tol = 1e-10
    # pa = max.(pa, tol)
    # pb = max.(pb, tol)
    kl = sum(pb .* log.(pb ./ pa)) # * (right - left) / npoint
    kl
end

name = "toto"
file = "results_$(name).txt"

KL(energies, energies)
KL(energies, energies .+ 0.002)
KL(energies .+ 0.002, energies)


points_KL = find_KL_range(energies, energies_eles, 1e-5, 2048)
pdf(kde(energies), points_KL) |> scatter
KL(energies, energies_eles, 1e-1, 10)

pdf(kde(energies), points_KL)
KL(rand(1000), rand(1000), 1e-5, 10)
kde(rand(100))
kde(energies)
(energies-energies_eles)./energies_eles |> scatter


function emd_dist(p, q)
    n_p = length(p)
    n_q = length(q)

    a = fill(1/n_p, n_p)
    b = fill(1/n_q, n_q)

    C = pairwise(SqEuclidean(), p', q'; dims=2)

    emd_dis = emd2(a, b, C, Tulip.Optimizer())
    # γ = emd(a, b, C, Tulip.Optimizer());
    # emd_dis = sum(γ .* C)

    emd_dis
end
emd_dist(energies, energies)
