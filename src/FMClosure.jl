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



export Grid, points, force!, forward_euler!, rk4!, randomfield, create_data, create_data_dns, sim_data, filter_u, closureterm, spectral_cutoff, spectrum

export UNet, create_dataloader, train, pseudo_timestepping, model_eval, sim_data_con, brownian_periodic
 


end # module Burgers
