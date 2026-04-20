# IMPORTANT: In the discretization.jl file, make sure to use the correct right hand side function for the Burgers equation (instead of the KS equation) when simulating the data and doing the physical time stepping with the learned model. Comment/uncomment the appropriate lines.

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
using JLD2
using FMClosure

outdir = joinpath(@__DIR__, "output") |> mkpath

visc = 0.005

# Define problem
get_grid(n, visc) = (; grid = Grid(2π, n), params = (; visc))



# Plot solution
let
    (; grid, params) = get_grid(1024, visc)
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
# (; grid, params) = get_grid(2048, 2e-3)

(; grid, params) = get_grid(256, visc)
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
(; grid, params) = get_grid(1024, visc)
grid_dns = grid
(; grid, params) = get_grid(128, visc)
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

# [1]: u simulated on fine grid  (DNS (time t))
# [2]: difference between fine and coarse grid (Difference between fDNS and LES)
# [3]: u simulated on coarse grid (LES)
# [4]: difference between u_bar_sim and u_bar


data_type = "closure" # "snap" or "closure"
snap_type = "diff" # "full" or "diff"


data_dns = data_dns_burgers

# Prepare DNS simulated data with full next snapshot as target
if data_type == "snap"
    data = let
        n_dns = grid_dns.n
        n_les = grid_les.n
        L = grid_dns.l
        
        inputs = zeros(grid_les.n, size(data_dns[1], 2)-1, size(data_dns[1], 3))
        outputs = zeros(grid_les.n, size(data_dns[1], 2)-1, size(data_dns[1], 3))
        for isample = 1:size(data_dns[1], 3)
            for itime = 1:size(data_dns[1], 2)-1
                inputs[:, itime, isample] = filter_u(data_dns[1][:, itime, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
                if snap_type == "full"
                    outputs[:, itime, isample] = filter_u(data_dns[1][:, itime+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02)
                elseif snap_type == "diff"
                    outputs[:, itime, isample] = filter_u(data_dns[1][:, itime+1, isample], grid_dns.l, grid_dns.n, grid_les.n, "gaussian", 0.02) - inputs[:, itime, isample]
                end
            end
        end
        inputs, outputs
    end

elseif data_type == "closure"
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






# Define filename to save/load trained model parameters
if data_type == "snap"
    if snap_type == "full"
        filename = "parameters/burgers_discsmall50_brownian.jld2"
    elseif snap_type == "diff"
        filename = "parameters/burgers_discdiffsmall50_brownian.jld2"
    end
elseif data_type == "closure"
    filename = "parameters/burgers_cont_brownian.jld2"
end

# Set do_train to true to train the model, and false to load the trained model parameters. Uncomment/comment the params line to use these parameters for training (instead of training from scratch). 
do_train = false
if do_train
    ps_freeze, st_freeze = train(;
        model,
        rng = Xoshiro(0),
        nepoch = 20,
        dataloader = create_dataloader(grid, data, 64, Xoshiro(0)),
        opt = AdamW(1.0f-3),  # Set weight decay \lambda maybe to 10^-4
        device,
        a,
        b,
        # params = (ps_freeze, st_freeze),
    );
    filename = "parameters/burgers_model.jld2"
    jldsave(filename; ps_freeze, st_freeze)
end


# Load trained model
ps_freeze, st_freeze = load(filename, "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))




if data_type == "snap"
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
            if snap_type == "full"
                x = model_eval(unet, x, noise_type, a, b, nsubstep, sigma, sigma_brown, false, device)
            elseif snap_type == "diff"
                x = x + model_eval(unet, x, noise_type, a, b, nsubstep, sigma, sigma_brown, false, device)
            end
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
        # save("outdir/spectrum_cutoff.png", fig)
        # ylims!(ax, 1e-10, 1e+1)
        display(fig)

    end


elseif data_type == "closure"
    # Method Continuous: Continuous closure term \overline{F(u(t_n))} - F(\overline{u(t_n)})
    # Plot one prediction
    nt = 200
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
        # save("outdir/spectrum_cutoff.png", fig)
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

end



# Average over multiple simulations of FM, change the number of nsample to increase/decrease the number of simulations.

# Define models
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
    channels = [8, 8],
    nresidual = 2,
    t_embed_dim = 32,
    y_embed_dim = 32,
    device,
)
ps_freeze_disc, st_freeze_disc = load("parameters/burgers_discsmall50_brownian.jld2", "ps_freeze", "st_freeze");
unet_disc = (x, t, y) -> first(model_disc((x, t, y), ps_freeze_disc, Lux.testmode(st_freeze_disc)))

model_discdiff = UNet(;
    nspace = grid.n,
    channels = [8, 8],
    nresidual = 2,
    t_embed_dim = 32,
    y_embed_dim = 32,
    device,
)
ps_freeze_discdiff, st_freeze_discdiff = load("parameters/burgers_discdiffsmall50_brownian.jld2", "ps_freeze", "st_freeze");
unet_discdiff = (x, t, y) -> first(model_discdiff((x, t, y), ps_freeze_discdiff, Lux.testmode(st_freeze_discdiff)))


nsample = 5
nt = 200
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
        # save("outdir/Burgers_spectrum_FM_avg.pdf", fig)
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
        # save("outdir/Burgers_spectrum_FM_avg_avg.pdf", fig)
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
        # save("outdir/Burgers_rel_err_FM.pdf", fig)
    display(fig)
end
