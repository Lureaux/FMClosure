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
# using WGLMakie
using Random
using Zygote
using ForwardDiff
using FMClosure

outdir = joinpath(@__DIR__, "output") |> mkpath
# CairoMakie.activate!() 


# Define problem
burgers(n, visc) = (; grid = Grid(2π, n), params = (; visc))
kdv(n) = (;
    grid = Grid(30.0, n),
    params = (;), # No params for KdV
)


# Plot solution
let
    # (; grid, params) = burgers(8192, 5e-4)
    (; grid, params) = kdv(256)
    ustart = randomfield(grid, 10.0, Xoshiro(0))
    u = copy(ustart)
    # cache = similar(u) # (forward_euler)
    cache = similar(u), similar(u), similar(u), similar(u), similar(u) # (RK4)
    t = 0.0
    tstop = 0.1
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
    save("$outdir/solution.pdf", fig; backend = CairoMakie)
    fig
end


# Create dataset
# (; grid, params) = burgers(2048, 2e-3)

(; grid, params) = kdv(256)
data = create_data(;
    grid,
    params,
    nsample = 100,
    ntime = 100,
    nsubstep = 10,
    dt = 1e-3,
    rng = Xoshiro(0),
);

data_test = create_data(;
    grid,
    params,
    nsample = 1,
    ntime = 100,
    nsubstep = 10,
    dt = 1e-3,
    rng = Xoshiro(1),
);


# Experiment with datasets
(; grid, params) = kdv(1024)
grid_dns = grid
(; grid, params) = kdv(256)
grid_les = grid

data_dns =  create_data_dns(;
    grid_dns,
    grid_les,
    params,
    nsample = 1,
    ntime = 100,
    nsubstep = 1000,
    dt = 1e-5,
    rng = Xoshiro(0),
);
# [1]: u simulated on fine grid 
# [2]: difference between fine and coarse grid
# [3]: u simulated on coarse grid

using JLD2
filename = "mydata_dns.jld2"
# jldsave(filename; data_dns)
data_dns = load(filename, "data_dns")


data_dns[4]

for isample = 1:100
    @info "isample = $isample"
    u_test = data_dns[1][:,1,isample]
    u_test_bar = filter_u(u_test, grid.l, 1024, 256, "spectral")

    u_sim_test =  sim_data(;
        u = u_test_bar, 
        grid = grid_les, 
        params, 
        nsubstep = 1000, 
        ntime = 100, 
        dt = 1e-5)
    data_dns[3][:,:,isample] = u_sim_test
end

for isample = 1:100
    @info "isample = $isample"
    for itime = 1:99
        data_dns[2][:,itime,isample] = filter_u(data_dns[1][:,itime+1,isample], grid.l, 1024, 256, "spectral") - data_dns[3][:,itime+1,isample]
    end
    u_next_mat = sim_data(;
        u = data_dns[1][:,100,isample], 
        grid = grid_dns, 
        params, 
        nsubstep = 1000, 
        ntime = 2, 
        dt = 1e-5)
    u_next = u_next_mat[:,2]    
    u_next_bar = filter_u(u_next, grid.l, 1024, 256, "spectral")

    v_next_mat = sim_data(;
        u =  data_dns[3][:,100,isample], 
        grid = grid_les, 
        params, 
        nsubstep = 1000, 
        ntime = 2, 
        dt = 1e-5)
    v_next = v_next_mat[:,2]

    data_dns[2][:,100,isample] = u_next_bar - v_next
end



for isample = 1:100
    @info "isample = $isample"
    for itime = 1:99
        u_bar = filter_u(data_dns[1][:,itime,isample], grid.l, 1024, 256, "spectral")
        u_bar_next_mat = sim_data(;
            u = u_bar, 
            grid = grid_les, 
            params, 
            nsubstep = 1000, 
            ntime = 2, 
            dt = 1e-5)
        u_bar_next = u_bar_next_mat[:,2]    
        data_dns[4][:,itime,isample] = filter_u(data_dns[1][:,itime+1,isample], grid.l, 1024, 256, "spectral") - u_bar_next
    end
    u_bar = filter_u(data_dns[1][:,100,isample], grid.l, 1024, 256, "spectral")
    u_bar_next_mat = sim_data(;
        u = u_bar, 
        grid = grid_les, 
        params, 
        nsubstep = 1000, 
        ntime = 2, 
        dt = 1e-5)
    u_bar_next = u_bar_next_mat[:,2]    

    u_next_mat = sim_data(;
        u = data_dns[1][:,100,isample], 
        grid = grid_dns, 
        params, 
        nsubstep = 1000, 
        ntime = 2, 
        dt = 1e-5)
    u_next = u_next_mat[:,2] 
    u_next_bar = filter_u(u_next, grid.l, 1024, 256, "spectral")

    data_dns[4][:,100,isample] = u_next_bar - u_bar_next
end



u_sim_test - data_dns[3][:,:,1]
t_sample = 1
# maximum( (u_sim_test[:,2] + data_dns[2][:,1,1]) - filter_u(data_dns[1][:,2,1], grid.l, 1024, 256, "spectral"))
maximum( (data_dns[3][:,t_sample+1,1] + data_dns[2][:,t_sample,1]) - filter_u(data_dns[1][:,t_sample+1,1], grid.l, 1024, 256, "spectral"))
maximum(data_dns[4])

test = sim_data(;
    u = filter_u(data_dns[1][:,10,1], grid.l, 1024, 256, "spectral"), 
    grid = grid_les, 
    params, 
    nsubstep = 1000, 
    ntime = 2, 
    dt = 1e-5)
minimum(test[:,2]+data_dns[4][:,10,1] - filter_u(data_dns[1][:,11,1], grid.l, 1024, 256, "spectral"))

size(data_dns[1])
let
    # isample = 1
    # itime = 10
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, points(Grid(30.0, 1024)), data_dns[1][:,2,1])
    lines!(ax, points(Grid(30.0, 1024)), data_dns[1][:,1,1] + data_dns[2][:,1,1])
    fig
end
for i = 1:100
    fig = Figure()
    x = points(grid)
    ax = Axis(fig[1, 1])
    lines!(ax, points(Grid(30.0, 1024)), data_dns[1][:,i,1])
    display(fig)
end

filter_u(data_test[1][:,1,1], 2π, 1024, 256, "spectral")
grid

data[1] |> length |> x -> 1.0 * x
data


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
    # lines!(ax, x, data[1][:, itime, isample] + data[2][:, itime, isample])
    # lines!(ax, x, data[1][:, itime+1, isample])
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

# Experiment
cospi(0.25)
cos(pi * 0.25)
similar(data[1][:,1,1])

# Linear
a(t) = t
b(t) = 1 .- t
# adot(t) = ones(size(t))
# bdot(t) = -1*ones(size(t))

# # GVP
# a(t) = sin((πf/2)*t)
# b(t) = cos((πf/2)*t)
# adot(t) = @. (πf/2) * cos((πf/2)*t)
# bdot(t) = @. -(πf/2) * sin((πf/2)*t)

adot(t) = ForwardDiff.derivative(a, t)
bdot(t) = ForwardDiff.derivative(b, t)



data_dns_bar = similar(data_dns[4])
for isample = 1:size(data_dns[4], 3)
    for itime = 1:size(data_dns[4], 2)
        data_dns_bar[:, itime, isample] = filter_u(data_dns[1][:, itime, isample], grid.l, 1024, 256, "spectral")
    end
end
data = (data_dns_bar, data_dns[4])


size(data[2], ndims(data[2]))
rand!(similar(data[2], 1, 1, 100))

# Train model
# ps_freeze, st_freeze = train(;
#     model,
#     rng = Xoshiro(0),
#     nepoch = 5,
#     dataloader = create_dataloader(grid, data, 100, Xoshiro(0)),
#     opt = AdamW(1.0f-3),  # Set weight decay \lambda maybe to 10^-4
#     device,
#     a,
#     b,
#     # params = (ps_freeze, st_freeze),
# );


# Load/save trained model
using JLD2
ic_type = "gaussian"
filename = "myparameters_linear_batchnorm_les.jld2"
# jldsave(filename; ps_freeze, st_freeze)
ps_freeze, st_freeze = load(filename, "ps_freeze", "st_freeze");
unet = (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))


# x = 1:5 |> collect
# y = reshape(20:20:80, 1, :) |> collect
# @. x + y 
# size(x)


# Define noise schedule
sigma(t) = 0.0f0 * ones(size(t))
# sigma(t) = sqrt.(a(t))
# sigma(t) = 1*sin.(π*t)
# sigma(t) = 2*(adot(t) .*a(t) - bdot(t) .* a(t).^2 ./ b(t))


# Plot one prediction
let
    isample = 1
    itime = 1
    dev = gpu_device()
    y, z = data_test
    y = reshape(y[:, itime, isample], :, 1, 1) |> f32 |> device
    z = reshape(z[:, itime, isample], :, 1, 1) |> f32 |> device
    x = randn!(similar(z))
    # x = brownian_periodic(x, 1.0)
    x_init = copy(x)
    # x_brown = copy(x)
    # x_brown[1] = 0.0
    # for i = 2:length(x_brown)
    #     x_brown[i] = x_brown[i-1] + sqrt(2π/length(x_brown)) * randn()
    # end 
    # x = x_brown
    nstep = 10
    t = fill(0.0f0, 1, 1, size(z, 3)) |> dev
    h = 1.0f0 / nstep
    for i = 1:nstep
        @info i
        u = unet(x, t, y)
        # @. x += h * u
        # score = (u - adot(t) / a(t) * x) / (b(t)^2 * adot(t) / a(t) - bdot(t) * b(t))
        score = @. (a(t) .* u - adot(t) .* x) ./ (b(t).^2 .* adot(t) - a(t) .* bdot(t) .* b(t))
        x += h * (u + score .* sigma(t).^2 /2) + sigma(t) .* sqrt(h) .* randn(size(u))
        @. t += h
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    input = y[:] |> cpu_device()
    target = z[:] |> cpu_device()
    prediction = x[:] |> cpu_device()
    # lines!(ax, points(grid), input; label = "Input")
    # lines!(ax, points(grid), target; label = "Target")
    # lines!(ax, points(grid), prediction; label = "Prediction")
    lines!(ax, points(grid), input + target; label = "Target")
    lines!(ax, points(grid), input + prediction; label = "Prediction")
    # lines!(ax, points(grid), x_init[:]; label = "Noise")
    # lines!(ax, points(grid), x_brown[:]; label = "Brownian Noise")
    # lines!(ax, points(grid), x[:]; label = "Prediction")
    axislegend(ax)
    save("$outdir/prediction.pdf", fig; backend = CairoMakie)
    fig
end


u = 256 + 1 |> randn |> cumsum |> x -> 0.1x
x = range(0, 1, 256 + 1)
l = @. x * u[end] + (1-x) * u[1]
v = u - l
let fig = Figure()
    ax = Axis(fig[1, 1])
    lines!(ax, x[1:end-1], u[1:end-1]; label = "u")
    lines!(ax, x[1:end-1], l[1:end-1]; label = "linear")
    lines!(ax, x[1:end-1], v[1:end-1]; label = "u - linear")
    axislegend(ax)
    fig
end

# Plug FM model back into physical time stepping loop
let
    isample = 1
    inputs, _ = data_test
    # ntime = size(y, 2)
    ntime = 50
    y = reshape(inputs[:, 1, isample], :, 1, 1) |> f32 |> device
    x = similar(y) |> device
    nsample = size(y, 3)
    t = fill(0.0f0, 1, 1, nsample) |> device
    nsubstep = 100
    h = 1.0f0 / nsubstep
    for itime = 1:ntime # Physical time stepping
        @show itime
        fill!(t, 0)
        # randn!(x) # Random initial conditions
        x = brownian_periodic(x, 1.0) # Random initial conditions
        for isub = 1:nsubstep # Pseudo-time stepping
            u = unet(x, t, y)
            # @. x += h * u
            score = @. (a(t) .* u - adot(t) .* x) ./ (b(t).^2 .* adot(t) - a(t) .* bdot(t) .* b(t))
            # @. x += h * (u + score * sigma^2 /2) + sigma * sqrt(h) * randn()
            x += h * (u + score .* sigma(t).^2 /2) + sigma(t) .* sqrt(h) .* randn(size(u))
            @. t += h
        end
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

isample = 1
inputs, _ = data_test
# ntime = size(y, 2)
ntime = 10
y = reshape(inputs[:, 1, isample], :, 1, 1) |> f32 |> device
x = similar(y) |> device

let fig = Figure()
    y = reshape(inputs[:, 1, isample], :, 1, 1) |> f32 |> device
    x = brownian_periodic(y, 1.0)
    print(sum(x))
    ax = Axis(fig[1, 1])
    lines!(ax, x[:]; label = "u - linear")
    axislegend(ax)
    fig
end



# Plot one prediction
let
    isample = 1
    itime = 10
    dev = gpu_device()
    y, z = data
    y = reshape(y[:, itime, isample], :, 1, 1) |> f32 |> device
    z = reshape(z[:, itime, isample], :, 1, 1) |> f32 |> device
    x = randn!(similar(z))
    nstep = 10
    t = fill(0.0f0, 1, 1, size(z, 3)) |> dev
    h = 1.0f0 / nstep
    for i = 1:nstep
        @info i
        u = unet(x, t, y)
        # @. x += h * u
        # score = (u - adot(t) / a(t) * x) / (b(t)^2 * adot(t) / a(t) - bdot(t) * b(t))
        score = @. (a(t) .* u - adot(t) .* x) ./ (b(t).^2 .* adot(t) - a(t) .* bdot(t) .* b(t))
        x += h * (u + score .* sigma(t).^2 /2) + sigma(t) .* sqrt(h) .* randn(size(u))
        @. t += h
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    input = y[:] |> cpu_device()
    input_next_mat = sim_data(; u = input, 
        grid = grid_les, 
        params, 
        nsubstep = 1000, 
        ntime = 2, 
        dt = 1e-5)
    input_next = input_next_mat[:,2]
    target = z[:] |> cpu_device()
    prediction = x[:] |> cpu_device()
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
    # ntime = size(y, 2)
    ntime = 5
    y = reshape(inputs[:, 1, isample], :, 1, 1) |> f32 |> device
    x = similar(y) |> device
    nsample = size(y, 3)
    t = fill(0.0f0, 1, 1, nsample) |> device
    nsubstep = 10
    h = 1.0f0 / nsubstep
    for itime = 1:ntime # Physical time stepping
        @show itime
        fill!(t, 0)
        randn!(x) # Random initial conditions
        for isub = 1:nsubstep # Pseudo-time stepping
            u = unet(x, t, y)
            # @. x += h * u
            score = @. (a(t) .* u - adot(t) .* x) ./ (b(t).^2 .* adot(t) - a(t) .* bdot(t) .* b(t))
            # @. x += h * (u + score * sigma^2 /2) + sigma * sqrt(h) * randn()
            x += h * (u + score .* sigma(t).^2 /2) + sigma(t) .* sqrt(h) .* randn(size(u))
            @. t += h
        end
        # function f!(du, u) 
        #     apply!(force!, grid, (du, u, grid, params))
        #     du .+= x
        # end

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
        # @. y += x # x is the physical step
        @. y += target_exact[:, itime, isample] # x is the physical step
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
