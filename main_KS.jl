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
pardir = joinpath(@__DIR__, "parameters") |> mkpath

visc = 0.005

# Define problem
burgers(n, visc) = (; grid = Grid(64, n), params = (; visc))
ks(n) = (;
    grid = Grid(64.0, n),
    params = (;), # No params for KdV
)


# Plot solution
let
    (; grid, params) = burgers(64, visc)
    ustart = randomfield(grid, 10.0, 10.0, Xoshiro(0))
    u = copy(ustart)
    # cache = similar(u) # (forward_euler)
    cache = similar(u), similar(u), similar(u), similar(u), similar(u) # (RK4)
    t = 0.0
    tstop = 20.0
    while t < tstop
        # dt = 0.3 * propose_timestep(u, grid, visc)
        dt = 0.005
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



# Load DNS data
(; grid, params) = burgers(256, visc)
grid_dns = grid
(; grid, params) = burgers(32, visc)
grid_les = grid


using JLD2

data_closure_test = let
filename = "KS_sim_closure.jld2"
load(filename, "sol_closure")
end

data_closure_1 = let
filename = "KS_sim_closure1.jld2"
load(filename, "sol_closure")
end

data_closure_2 = let
filename = "KS_sim_closure2.jld2"
load(filename, "sol_closure")
end

data_closure_3 = let
filename = "KS_sim_closure3.jld2"
load(filename, "sol_closure")
end

data_closure_4 = let
filename = "KS_sim_closure4.jld2"
load(filename, "sol_closure")
end

data_closure_5 = let
filename = "KS_sim_closure5.jld2"
load(filename, "sol_closure")
end

data_snap_test = let
filename = "KS_sim_snap.jld2"
load(filename, "sol_snap")
end

data_snap_1 = let
filename = "KS_sim_snap1.jld2"
load(filename, "sol_snap")
end

data_snap_2 = let
filename = "KS_sim_snap2.jld2"
load(filename, "sol_snap")
end

data_snap_3 = let
filename = "KS_sim_snap3.jld2"
load(filename, "sol_snap")
end

data_snap_4 = let
filename = "KS_sim_snap4.jld2"
load(filename, "sol_snap")
end

data_snap_5 = let
filename = "KS_sim_snap5.jld2"
load(filename, "sol_snap")
end

data_dns = zeros(256, 301, 500)
data_dns[:,:,1:100] = data_snap_1
data_dns[:,:,101:200] = data_snap_2
data_dns[:,:,201:300] = data_snap_3
data_dns[:,:,301:400] = data_snap_4
data_dns[:,:,401:500] = data_snap_5




# Define model
device = gpu_device()
model = UNet(;
    nspace = grid.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
)

const πf = Float32(π)

# Noise schedulers
a(t) = t
b(t) = 1 .- t.^2


adot(t) = ForwardDiff.derivative(a, t)
bdot(t) = ForwardDiff.derivative(b, t)



# Prepare DNS simulated data with full next snapshot as target
data = let
    n_dns = grid_dns.n
    n_les = grid_les.n
    L = grid_dns.l
    
    inputs = zeros(grid_les.n, size(data_dns, 2)-1, size(data_dns, 3))
    outputs = zeros(grid_les.n, size(data_dns, 2)-1, size(data_dns, 3))
    for isample = 1:size(data_dns, 3)
        for itime = 1:size(data_dns, 2)-1
            inputs[:, itime, isample] = filter_u(data_dns[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
            outputs[:, itime, isample] = filter_u(data_dns[:, itime+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
        end
    end
    inputs, outputs
end

data_test = let
    n_dns = grid_dns.n
    n_les = grid_les.n
    L = grid_dns.l
    
    inputs = zeros(grid_les.n, size(data_snap_test, 2)-1, size(data_snap_test, 3))
    outputs = zeros(grid_les.n, size(data_snap_test, 2)-1, size(data_snap_test, 3))
    for isample = 1:size(data_snap_test, 3)
        for itime = 1:size(data_snap_test, 2)-1
            inputs[:, itime, isample] = filter_u(data_snap_test[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
            outputs[:, itime, isample] = filter_u(data_snap_test[:, itime+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05) 
        end
    end
    inputs, outputs
end

# Test with [1] filtered DNS snapshots and [2] zeros
data_test = let
    n_dns = grid_dns.n
    n_les = grid_les.n
    L = grid_dns.l
    
    inputs = zeros(grid_les.n, size(data_snap_test, 2), size(data_snap_test, 3))
    outputs = zeros(grid_les.n, size(data_snap_test, 2), size(data_snap_test, 3))
    for isample = 1:size(data_snap_test, 3)
        for itime = 1:size(data_snap_test, 2)
            inputs[:, itime, isample] = filter_u(data_snap_test[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05) 
        end
    end
    inputs, outputs
end



# Prepare DNS simulated data continuous
data_dns_bar = zeros(grid_les.n, 301, 500)
closures = zeros(grid_les.n, 301, 500)
for isample = 1:size(data_dns, 3)
    for itime = 1:size(data_dns, 2)
        data_dns_bar[:, itime, isample] = filter_u(data_dns[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
        closures[:, itime, isample] = closureterm(data_dns[:, itime, isample], grid_dns, grid_les)
    end
end
# data = (data_dns_bar, data_dns[4])
data = (data_dns_bar, closures)

# data_test = let 
#     data_bar = zeros(grid_les.n, 301, 100)
#     closures = zeros(grid_les.n, 301, 100)
#     for isample = 1:size(data_closure_test, 3)
#         for itime = 1:size(data_closure_test, 2)
#             data_bar[:, itime, isample] = filter_u(data_snap_test[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
#             # closures[:, itime, isample] = closureterm(data_closure_test[:, itime, isample], grid_dns, grid_les)
#         end
#     end
#     data_bar, closures
# end



# # Train model
# ps_freeze, st_freeze = train(;
#     model,
#     rng = Xoshiro(0),
#     nepoch = 10,
#     dataloader = create_dataloader(grid, data, 256, Xoshiro(0)),
    # opt = AdamW(1.0f-3),  # Set weight decay \lambda maybe to 10^-4
#     device,
#     a,
#     b,
#     params = (ps_freeze, st_freeze),
# );


# Load/save trained model
using JLD2
# filename = "myparameters_linear_batchnorm_cont_brownian.jld2"
# filename = "myparameters_KS_disc20_m2gaussian.jld2"
filename = "parameters/KS_disc20_m2gaussian.jld2"
# jldsave(filename; ps_freeze, st_freeze)
ps_freeze, st_freeze = load(filename, "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))



# Define noise schedule

# sigma(t) = 0.0f0 * ones(size(t))
sigma(t) = sqrt.(0.1f0 * (1 .- t))


noise_type = "gaussian"
noise_type = "brownian"
sigma_brown = 1.0

data_test[1][:,2,1]
data_test[2][:,1,1]


# Method 1: Direct prediction of u(t_{n+1})
# Plot one prediction
nt = 300

let
    t0 = time_ns()
    isample = 3
    itime = 1
    y_data, z_data = data_test
    y = reshape(y_data[:, itime, isample], :, 1, 1) |> f32 |> device
    z = y + reshape(z_data[:, nt, isample], :, 1, 1) |> f32 |> device
    nsubstep = 10

    x = copy(y)


    for i = 1:nt
        x = x + model_eval(unet, x, noise_type, a, b, nsubstep, sigma, sigma_brown, false, device)
    end
       
    t1 = time_ns()
    elapsed = (t1 - t0) / 1e9
    print(elapsed)

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





testtest = reshape(data_test[1][:, 1, 1], :, 1, 1) |> f32 |> device
testtest_eval = model_eval(unet, testtest, noise_type, a, b, 10, sigma, sigma_brown, false, device);


for i = 1:11
    fig = Figure()
    ax = Axis(fig[1, 1])
    prediction = testtest_eval[:,i] |> cpu_device()
    lines!(ax, points(grid), prediction; label = "Prediction")
    axislegend(ax)
    # save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    display(fig)
end

let
    testtest_eval = model_eval(unet, testtest, noise_type, a, b, 10, sigma, sigma_brown, false, device);
    fig = Figure()
    ax = Axis(fig[1, 1])
    noise_begin = testtest_eval[:,1] |> cpu_device()
    noise_end = testtest_eval[:,11] |> cpu_device()
    lines!(ax, points(grid), noise_begin; label = "X0")
    lines!(ax, points(grid), noise_end; label = "X1")
    lines!(ax, points(grid), data_test[2][:, 1, 1]; label = "Target")
    # ylims!(-2,2)
    axislegend(ax)
    # save("$outdir/prediction.pdf", fig; backend = CairoMakie)
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
data_snap_1
# Method 3: Continuous closure term \overline{F(u(t_n))} - F(\overline{u(t_n)})
# Plot one prediction
nt = 3000
let
    factor = 10
    nt_fm = Int(nt/factor)
    isample = 1
    itime = 1
    y = data_test[1][:, itime, isample]
    y = reshape(y, :, 1, 1) |> f32 |> device
    target = data_test[1][:, end, isample]

    nsubstep_pseudo = 3
    
    input = y[:] |> cpu_device()
    input_copy = copy(input)
    t0 = time_ns()
    input_next_mat = sim_data_con(; u = input_copy, 
        grid = grid_les,
        params,
        nsubstep = 1, 
        ntime = nt_fm+1, 
        dt = factor*0.05 ,
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
    # ylims!(ax, -3, 3)
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
        title = "Energy spectrum at final time",
    )
    lines!(ax, k, s; label = "Target")
    lines!(ax, kbar, sbar; label = "Prediction")
    # lines!(ax, k, k -> 100k^-2; label = "k^-2")
    # ylims!(ax, 1e-16, 1e+1)
    axislegend(ax; position = :lb)
    # save("spectrum_cutoff.png", fig)
    # ylims!(ax, 1e-10, 1e+1)
    display(fig)

    # norm_fm = zeros(nt_fm+1)
    # for t = 0:nt_fm
    #     snap_filt = data_test[1][:, t+1, isample]
    #     norm_fm[t+1] = norm(input_next_mat[:,t+1] - snap_filt) / norm(snap_filt)
    # end
    # fig = Figure()
    # ax = Axis(fig[1, 1];
    #     xlabel="t",
    #     ylabel="Relative error",
    #     title = "Relative error with filtered DNS",
    # )
    # lines!(range(0.0, 150.0, step=factor*0.05), norm_fm; label = "Flow Matching")
    # axislegend(ax; position = :lt)
    # display(fig)
    # # print(norm_fm)
end



# Average over 100 simulations of FM
# Define discrete model
model = UNet(;
    nspace = grid.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
)
ps_freeze, st_freeze = load("parameters/KS_cont25_m2brownian.jld2", "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))

model_disc = UNet(;
    nspace = grid.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
)
ps_freeze_disc, st_freeze_disc = load("parameters/KS_disc20_m2gaussian.jld2", "ps_freeze", "st_freeze");
unet_disc = (x, t, y) -> first(model_disc((x, t, y), ps_freeze_disc, Lux.testmode(st_freeze_disc)))

model_discdiff = UNet(;
    nspace = grid.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
)
ps_freeze_discdiff, st_freeze_discdiff = load("parameters/KS_discdiff20_m2brownian.jld2", "ps_freeze", "st_freeze");
unet_discdiff = (x, t, y) -> first(model_discdiff((x, t, y), ps_freeze_discdiff, Lux.testmode(st_freeze_discdiff)))


nsample = 100
nt= 300

times = zeros(4)
t0 = time_ns()
t1 = time_ns()
(t1 - t0) / 1e9 ./ nsample
times[1] += (t1 - t0) / 1e9

factor = 1
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
    print(isample)
    itime = 1
    y = data_test[1][:, itime, isample]
    y = reshape(y, :, 1, 1) |> f32 |> device
    target = data_test[1][:, end, isample]

    nsubstep_pseudo = 5
    
    input = y[:] |> cpu_device()
    input_copy = copy(input)
    input_copy_disc = copy(input)
    input_copy_discdiff = copy(input)

    time_fmcon = 0.0
    t0 = time_ns()
    # input_next_mat = sim_data_con(; u = input_copy, 
    #     grid = grid_les,
    #     params,
    #     nsubstep = 1, 
    #     ntime = nt_fm+1, 
    #     dt = 0.5 ,
    #     model = unet, noise_type, a, b, nsubstep_pseudo, sigma, sigma_brown, device)
    t1 = time_ns()
    times[2] += (t1 - t0) / 1e9

    input_next_mat = zeros(grid_les.n, nt_fm+1)
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
    # for i = 1:nt_fm
    #     current = reshape(sol_discdiff[:, i], :, 1, 1) |> f32 |> device
    #     sol_discdiff[:, i+1] = sol_discdiff[:, i] + model_eval(unet_discdiff, current, noise_type , a, b, nsubstep, sigma, sigma_brown, false, device)
    # end
    t1 = time_ns()
    times[4] += (t1 - t0) / 1e9

    k, s = spectrum(target, grid_les)

    norm_fm = zeros(nt_fm+1)
    norm_fmdisc = zeros(nt_fm+1)
    norm_fmdiscdiff = zeros(nt_fm+1)
    for t = 0:nt_fm
        snap_filt = data_test[1][:, t+1, isample]
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

    fig = Figure()
    ax = Axis(fig[1, 1])
    # input = y[:] |> cpu_device()
    # target = z[:] |> cpu_device()
    # prediction = x[:] |> cpu_device()
    lines!(ax, points(grid), target; label = "Target")
    # lines!(ax, points(grid), input_next_mat[:,end]; label = "Prediction continuous")
    lines!(ax, points(grid), sol_disc[:,end]; label = "Prediction discrete full")
    # lines!(ax, points(grid), sol_discdiff[:,end]; label = "Prediction discrete difference")
    axislegend(ax)
    # save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    display(fig)

end
times


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

k, _ = spectrum(data_test[1][:, end, 1], grid_les)

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
        # lines!(ax, k, s_fm_avg; label = "FM Continuous")
        lines!(ax, k, s_fmdisc_avg; label = "FM Discrete Full")
        lines!(ax, k, s_fmdiscdiff_avg; label = "FM Discrete Difference")
        # ylims!(ax, 0.001, 1e-1)
        axislegend(ax; position = :lb)
        # save("KS_spectrum_FM_avg.pdf", fig)
        # ylims!(ax, 1e-10, 1e+1)
    display(fig)
end



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
        # ylims!(ax, 0.001, 1e-1)
        axislegend(ax; position = :lb)
        # save("KS_spectrum_FM_avg_avg.pdf", fig)
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
        # lines!(range(0.0, 150.0, step=10*0.05), norm_fm_avg; label = "FM Continuous", color=Cycled(2))
        lines!(range(0.0, 150.0, step=0.5), norm_fmdisc_avg; label = "FM Discrete Full Full", color=Cycled(3))
        lines!(range(0.0, 150.0, step=0.5), norm_fmdiscdiff_avg; label = "FM Discrete Difference", color=Cycled(4))
        axislegend(ax; position = :lt)
        # ylims!(ax, 0, 0.25)
        # save("KS_rel_err_FM.pdf", fig)
    display(fig)
end

times_per_sample = times ./ nsample
times_per_sample




norm_fmdisc_avg,
norm_fmdiscdiff_avg,
s_fdns_avg,
s_fdns_avg_avg,
s_fmdisc_avg,
s_fmdisc_avg_avg,
s_fmdiscdiff_avg,
s_fmdiscdiff_avg_avg = load("sim_results_KSdisc.jld2", "norm_fmdisc_avg",
"norm_fmdiscdiff_avg",
"s_fdns_avg",
"s_fdns_avg_avg",
"s_fmdisc_avg",
"s_fmdisc_avg_avg",
"s_fmdiscdiff_avg",
"s_fmdiscdiff_avg_avg")




norm_fmdiscdiff_avg,
_,
s_fmdisc_avg,
_,
s_fmdiscdiff_avg_avg = load("sim_results_KSdiscGaussian.jld2","norm_fmdisc_avg",
"s_fdns_avg",
"s_fmdisc_avg",
"s_fdns_avg_avg",
"s_fmdisc_avg_avg")


# jldsave("sim_results_KSdiscDiffusion.jld2"; norm_fmdisc_avg,
# s_fdns_avg,
# s_fmdisc_avg,
# s_fdns_avg_avg,
# s_fmdisc_avg_avg)







