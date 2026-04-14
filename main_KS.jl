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
using Random
using Zygote
using ForwardDiff
using Statistics
using KernelDensity
using OptimalTransport
using Distances
using Tulip
using JLD2
using FMClosure

outdir = joinpath(@__DIR__, "output") |> mkpath
pardir = joinpath(@__DIR__, "parameters") |> mkpath

# Define a value for the viscosity, but it is not used for KS
visc = 0.005

# Define problem
get_grid(n, visc) = (; grid = Grid(64, n), params = (; visc))


# Plot an example solution
let
    (; grid, params) = get_grid(64, visc)
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



# Define DNS and LES grids
(; grid, params) = get_grid(256, visc)
grid_dns = grid
(; grid, params) = get_grid(32, visc)
grid_les = grid


# Load simulated data, DNS data constructed with smaller value of dt
# data_snap: Store snapshots after each dt_les=0.5
# data_closure: Used for constructing the closure terms, randomly select 301 snapshots from DNS data. Later in this file, this data is used to construct closure terms. 

# This data_closure_test data is not used
data_closure_test = let
filename = "data_KS/KS_sim_closure.jld2"
load(filename, "sol_closure")
end

data_closure_1 = let
filename = "data_KS/KS_sim_closure1.jld2"
load(filename, "sol_closure")
end

data_closure_2 = let
filename = "data_KS/KS_sim_closure2.jld2"
load(filename, "sol_closure")
end

data_closure_3 = let
filename = "data_KS/KS_sim_closure3.jld2"
load(filename, "sol_closure")
end

data_closure_4 = let
filename = "data_KS/KS_sim_closure4.jld2"
load(filename, "sol_closure")
end

data_closure_5 = let
filename = "data_KS/KS_sim_closure5.jld2"
load(filename, "sol_closure")
end

data_snap_test = let
filename = "data_KS/KS_sim_snap.jld2"
load(filename, "sol_snap")
end

data_snap_1 = let
filename = "data_KS/KS_sim_snap1.jld2"
load(filename, "sol_snap")
end

data_snap_2 = let
filename = "data_KS/KS_sim_snap2.jld2"
load(filename, "sol_snap")
end

data_snap_3 = let
filename = "data_KS/KS_sim_snap3.jld2"
load(filename, "sol_snap")
end

data_snap_4 = let
filename = "data_KS/KS_sim_snap4.jld2"
load(filename, "sol_snap")
end

data_snap_5 = let
filename = "data_KS/KS_sim_snap5.jld2"
load(filename, "sol_snap")
end


data_type = "closure" # "snap" or "closure"
snap_type = "diff" # "full" or "diff"


if data_type == "snap"
    data_dns = zeros(256, 301, 500)
    data_dns[:,:,1:100] = data_snap_1
    data_dns[:,:,101:200] = data_snap_2
    data_dns[:,:,201:300] = data_snap_3
    data_dns[:,:,301:400] = data_snap_4
    data_dns[:,:,401:500] = data_snap_5
elseif data_type == "closure"
    data_dns = zeros(256, 301, 500)
    data_dns[:,:,1:100] = data_closure_1
    data_dns[:,:,101:200] = data_closure_2
    data_dns[:,:,201:300] = data_closure_3
    data_dns[:,:,301:400] = data_closure_4
    data_dns[:,:,401:500] = data_closure_5
end

data_snap_1
data_closure_1

# Create training data    
if data_type == "snap"
    # Prepare DNS simulated data with full next snapshot as target
    # To predict the difference, uncomment the third line in the for loop (and comment second line)
    data = let
        n_dns = grid_dns.n
        n_les = grid_les.n
        L = grid_dns.l
        
        inputs = zeros(grid_les.n, size(data_dns, 2)-1, size(data_dns, 3))
        outputs = zeros(grid_les.n, size(data_dns, 2)-1, size(data_dns, 3))
        for isample = 1:size(data_dns, 3)
            for itime = 1:size(data_dns, 2)-1
                inputs[:, itime, isample] = filter_u(data_dns[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
                if snap_type == "full"
                    outputs[:, itime, isample] = filter_u(data_dns[:, itime+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
                elseif snap_type == "diff"
                    outputs[:, itime, isample] = filter_u(data_dns[:, itime+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05) - inputs[:, itime, isample]
                end
            end
        end
        inputs, outputs
    end
elseif data_type == "closure"
    # Prepare DNS simulated data continuous
    data_dns_bar = zeros(grid_les.n, 301, 500)
    closures = zeros(grid_les.n, 301, 500)
    for isample = 1:size(data_dns, 3)
        for itime = 1:size(data_dns, 2)
            data_dns_bar[:, itime, isample] = filter_u(data_dns[:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.05)
            closures[:, itime, isample] = closureterm(data_dns[:, itime, isample], grid_dns, grid_les)
        end
    end
    data = (data_dns_bar, closures)
end


# Create test data with filtered DNS snapshots
# [1] filtered DNS snapshots and [2] zeros (zeros are only used for correct formatting)
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



# Define model
device = gpu_device()
model = UNet(;
    nspace = grid_les.n,
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

# Noise type "gaussian" or "brownian"
noise_type = "brownian" 
sigma_brown = 1.0

# Define noise schedule
sigma(t) = 0.0f0 * ones(size(t))
# sigma(t) = sqrt.(0.1f0 * (1 .- t))



# Train model, uncomment to train (and comment out loading of trained model below)
if data_type == "snap"
    if snap_type == "full"
        filename = "parameters/KS_disc20_m2brownian.jld2"
        # filename = "parameters/KS_disc20_m2gaussian.jld2"
    elseif snap_type == "diff"
        filename = "parameters/KS_discdiff20_m2brownian.jld2"
    end
elseif data_type == "closure"
    filename = "parameters/KS_cont25_m2brownian.jld2"
end


do_train = false
if do_train
    ps_freeze, st_freeze = train(;
        model,
        rng = Xoshiro(0),
        nepoch = 10,
        dataloader = create_dataloader(grid_les, data, 256, Xoshiro(0)),
        opt = AdamW(1.0f-3),
        device,
        a,
        b,
        # params = (ps_freeze, st_freeze),
    );
    filename = "KS_model.jld2"
    jldsave(filename; ps_freeze, st_freeze)
end

ps_freeze, st_freeze = load(filename, "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))



# Plot one prediction
nt = 300

if data_type == "snap"
    # Method Discrete: Direct prediction of u(t_{n+1}) or u(t_{n+1}) - u(t_n) 
    let
        t0 = time_ns()
        isample = 3
        itime = 1
        y_data, z_data = data_test
        y = reshape(y_data[:, itime, isample], :, 1, 1) |> f32 |> device
        z = reshape(y_data[:, nt+1, isample], :, 1, 1) |> f32 |> device
        nsubstep = 10

        x = copy(y)


        for i = 1:nt
            if snap_type == "full"
                x = model_eval(unet, x, noise_type, a, b, nsubstep, sigma, sigma_brown, false, device)
            elseif snap_type == "diff"
                x = x + model_eval(unet, x, noise_type, a, b, nsubstep, sigma, sigma_brown, false, device)
            end
        end
        
        t1 = time_ns()
        elapsed = (t1 - t0) / 1e9
        print(elapsed)

        fig = Figure()
        ax = Axis(fig[1, 1])
        input = y[:] |> cpu_device()
        target = z[:] |> cpu_device()
        prediction = x[:] |> cpu_device()
        lines!(ax, points(grid_les), target; label = "Target")
        lines!(ax, points(grid_les), prediction; label = "Prediction")
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
        # save("$outdir/spectrum_cutoff.png", fig)
        # ylims!(ax, 1e-10, 1e+1)
        display(fig)
    end


elseif data_type == "closure"
    # Method Continuous: Continuous closure term \overline{F(u(t_n))} - F(\overline{u(t_n)})
    # Plot one prediction

    let
        nt_fm = nt
        isample = 1
        itime = 1
        y = data_test[1][:, itime, isample]
        y = reshape(y, :, 1, 1) |> f32 |> device
        target = data_test[1][:, end, isample]

        nsubstep_pseudo = 5
        
        input = y[:] |> cpu_device()
        input_copy = copy(input)
        t0 = time_ns()
        input_next_mat = sim_data_con(; u = input_copy, 
            grid = grid_les,
            params,
            nsubstep = 1, 
            ntime = nt_fm+1, 
            dt = 0.5 ,
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
        # lines!(ax, points(grid_les), input; label = "Input")
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
        # save("$outdir/spectrum_cutoff.png", fig)
        # ylims!(ax, 1e-10, 1e+1)
        display(fig)

        norm_fm = zeros(nt_fm+1)
        for t = 0:nt_fm
            snap_filt = data_test[1][:, t+1, isample]
            norm_fm[t+1] = norm(input_next_mat[:,t+1] - snap_filt) / norm(snap_filt)
        end
        fig = Figure()
        ax = Axis(fig[1, 1];
            xlabel="t",
            ylabel="Relative error",
            title = "Relative error with filtered DNS",
        )
        lines!(range(0.0, 150.0, step=0.5), norm_fm; label = "Flow Matching")
        axislegend(ax; position = :lt)
        display(fig)
        # print(norm_fm)
    end

end




# Average over multiple simulations for different FM targets
model = UNet(;
    nspace = grid_les.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
)
ps_freeze, st_freeze = load("parameters/KS_cont25_m2brownian.jld2", "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))

model_disc = UNet(;
    nspace = grid_les.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
)
ps_freeze_disc, st_freeze_disc = load("parameters/KS_disc20_m2gaussian.jld2", "ps_freeze", "st_freeze");
unet_disc = (x, t, y) -> first(model_disc((x, t, y), ps_freeze_disc, Lux.testmode(st_freeze_disc)))

model_discdiff = UNet(;
    nspace = grid_les.n,
    channels = [8, 16, 16, 8],
    nresidual = 2,
    t_embed_dim = 8,
    y_embed_dim = 8,
    device,
)
ps_freeze_discdiff, st_freeze_discdiff = load("parameters/KS_discdiff20_m2brownian.jld2", "ps_freeze", "st_freeze");
unet_discdiff = (x, t, y) -> first(model_discdiff((x, t, y), ps_freeze_discdiff, Lux.testmode(st_freeze_discdiff)))


# Maximum of 100 samples, single sample may take a couple of minutes to run
nsample = 1
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
    input_next_mat = sim_data_con(; u = input_copy, 
        grid = grid_les,
        params,
        nsubstep = 1, 
        ntime = nt_fm+1, 
        dt = 0.5 ,
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
    lines!(ax, points(grid_les), target; label = "Target")
    lines!(ax, points(grid_les), input_next_mat[:,end]; label = "Prediction continuous")
    lines!(ax, points(grid_les), sol_disc[:,end]; label = "Prediction discrete full")
    lines!(ax, points(grid_les), sol_discdiff[:,end]; label = "Prediction discrete difference")
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
        lines!(ax, k, s_fm_avg; label = "FM Continuous")
        lines!(ax, k, s_fmdisc_avg; label = "FM Discrete Full")
        lines!(ax, k, s_fmdiscdiff_avg; label = "FM Discrete Difference")
        # ylims!(ax, 0.001, 1e-1)
        axislegend(ax; position = :lb)
        # save("$outdir/KS_spectrum_FM_avg.pdf", fig)
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
        # save("$outdir/KS_spectrum_FM_avg_avg.pdf", fig)
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
        lines!(range(0.0, 150.0, step=10*0.05), norm_fm_avg; label = "FM Continuous", color=Cycled(2))
        lines!(range(0.0, 150.0, step=0.5), norm_fmdisc_avg; label = "FM Discrete Full Full", color=Cycled(3))
        lines!(range(0.0, 150.0, step=0.5), norm_fmdiscdiff_avg; label = "FM Discrete Difference", color=Cycled(4))
        axislegend(ax; position = :lt)
        # ylims!(ax, 0, 0.25)
        # save("$outdir/KS_rel_err_FM.pdf", fig)
    display(fig)
end

times_per_sample = times ./ nsample
times_per_sample









