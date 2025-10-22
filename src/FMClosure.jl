module FMClosure

using FFTW
using LinearAlgebra
using Lux
using MLUtils
using NNlib
using Random
using ForwardDiff

include("discretization.jl")
include("unet.jl")
include("models.jl")
include("KdV.jl")


export Grid,
    points, force!, forward_euler!, rk4!, propose_timestep, randomfield, create_data
export UNet, create_dataloader, train
export create_data_dns, sim_data
export spectral_cutoff, spectrum
export filter_u, closureterm, pseudo_timestepping, model_eval, sim_data_con
export brownian_periodic


end # module Burgers
