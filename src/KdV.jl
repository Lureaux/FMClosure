# du/dt = - d^3 u /dx^3 - 6 u du/dx

using FFTW
# using OrdinaryDiffEq      #DifferentialEquations
# using Statistics
using LinearAlgebra
using Distributions


# tophat(k, Δ) = sinpi(k * Δ) / π / k / Δ
# gaussian(k, Δ) = exp(-π^2 * Δ^2 * k^2 / 6)
# cutoff(k, Δ) = k < 1 / Δ

#   "Burgers' equation with Jameson scheme"
function f_disc(u, p, t=nothing)
    #p is e.g. of the form (grid = (L = 1, n = 500)) )
    (; grid) = p
    (; L, n) = grid
    h = L / n
    g = 1/6 .* (u.^2 + u .* circshift(u,-1) + circshift(u,-1).^2)
    conv = (g - circshift(g,1)) ./ h
    diff = (circshift(u,-1) - 2 * u + circshift(u,1)) ./ h^2
    fourth = (circshift(u,-2) - 4 *circshift(u,-1) + 6 * u - 4 * circshift(u,1) + circshift(u,2)) ./ h^4
    du = -conv - diff - fourth
    du
end



function f_fourier(u_hat, p, t=nothing)
    # du_hat/dt = -3 * i k (u.^2)_hat + i * k^3 * u_hat
    #p is e.g. of the form (grid = (L = 1, n = 4096), visc =0.0005) )
    (; grid) = p
    (; L, n) = grid
    K = div(n, 2)
    k = 0:K
    Kthird = div(K, 3)
    high_k = 2 * Kthird+1:K

    third = im .* (2π/L)^3 * (k.^3) .* u_hat

    u_hat_tilde = copy(u_hat)
    u_hat_tilde[high_k] .= 0.0
    u = irfft(u_hat_tilde, n)
    u_squared = u.^2
    u_squared_hat = rfft(u_squared)
    conv = -(1/2) * 6 * 2π/L * im * k .* u_squared_hat
    du_hat = conv + third
    du_hat
end

function f_fourier_lin(u_hat, p, t)
    (; grid) = p
    (; L, n) = grid
    K = div(n, 2)
    k = 0:K

    # diff = (2π/L)^2 * (k.^2) .* u_hat

    # fourth = -(2π/L)^4 * (k.^4) .* u_hat

    # du_hat_lin = diff + fourth
    A = Diagonal((2π/L)^2 * (k.^2) -(2π/L)^4 * (k.^4)) 
    du_hat_lin = A * u_hat
    du_hat_lin
end

function f_fourier_nonlin(u_hat, p, t)
    (; grid) = p
    (; L, n) = grid
    K = div(n, 2)
    k = 0:K
    Kthird = div(K, 3)
    high_k = 2 * Kthird+1:K

    u_hat_tilde = copy(u_hat)
    u_hat_tilde[high_k] .= 0.0
    u = irfft(u_hat_tilde, n)
    u_squared = u.^2
    u_squared_hat = rfft(u_squared)
    conv = -(1/2) * 2π/L * im * k .* u_squared_hat

    du_hat_nonlin = conv
    return du_hat_nonlin
end





#   "Define grid and viscosity"
function centeredrange(xₗ, xᵣ; length)
    step = (xᵣ - xₗ) / length
    start = xₗ + step / 2
    stop = xᵣ - step / 2
    return range(start, stop, length)
end

function leftrange(xₗ, xᵣ; length)
    return range(xₗ, xᵣ, length+1)[1:end-1]
end


#   "Remove high frequencies" 
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

function reconstruct(u_hat_les, n_dns, n_les)
    K_les = div(n_les, 2) # n_les/2
    K_dns = div(n_dns, 2) # n_dns/2
    kprime = K_les+1:K_dns # n_les/2 + 1 : n_dns/2
    power = 6
    a = 0
    a_num = 5
    for i=0:a_num-1
       a += abs(u_hat_les[end-i]) / n_les * (K_les-i)^power
    end
    a /= a_num    
    # a = abs(u_hat_les[end]) / n_les * K_les
    u_hat_prime = @. a * n_dns / kprime^power * exp(2 * π * im * rand())
    u_hat = [u_hat_les * n_dns / n_les; 1*u_hat_prime]
    u_hat
end

# function g(k, Δ, filter_type, L)
#     if filter_type == "spectral"
#         K_cutoff = div(1/Δ, 2)
#         g_k = zeros(Complex{Float64}, length(k))
#         j = 1
#         for i in k
#             if i <= K_cutoff
#                 g_k[j] = 1.0
#             end
#             j += 1
#         end
#     elseif filter_type == "gaussian"
#         g_k = exp.( -π^2 * (k/L).^2 * Δ^2 / 6)
#     elseif filter_type == "top_hat"
#         g_k = zeros(Complex{Float64}, length(k))
#         j = 1
#         for i in k
#             if i == 0
#                 g_k[j] = 1.0
#             else
#                 g_k[j] = sin(π * (i / L) * Δ) / (π * (i/ L) * Δ)
#             end
#         j += 1
#         end
#     end   
#     g_k 
# end


# function filter(u, L, n_dns, n_les, filter_type, Δ = 0.01)
#     K_dns = div(n_dns, 2)
#     k = 0:K_dns
#     u_hat = rfft(u)
#     u_bar_hat = g(k, Δ, filter_type, L) .* u_hat
#     u_bar_dns = irfft(u_bar_hat, n_dns)
#     u_bar = spectral_cutoff(u_bar_dns, n_dns, n_les)
#     u_bar
# end   

# function filter_u(u, L, n_dns, n_les, filter_type, Δ = 0.01)
#     K_dns = div(n_dns, 2)
#     k = 0:K_dns
#     u_hat = rfft(u)
#     u_bar_hat = g(k, Δ, filter_type, L) .* u_hat
#     u_bar_dns = irfft(u_bar_hat, n_dns)
#     u_bar = spectral_cutoff(u_bar_dns, n_dns, n_les)
#     u_bar
# end  



function inv_filter_cutoff(u_les, n_dns, n_les)
    u_hat_les = rfft(u_les)
    u_hat = reconstruct(u_hat_les, n_dns, n_les)
    u = irfft(u_hat, n_dns)
    u
end



function inv_filter(u_bar, L, n_dns, n_les, filter_type, Δ=0.01, λ=0.01, inv_type = "MAP")
    u_bar_hat = rfft(u_bar)
    n_u = length(u_bar)
    K_u = length(u_bar_hat) - 1
    k = 0:K_u
    g_k = g(k, Δ, filter_type, L)
    if inv_type == "ML"
        u_hat = @. (1 / g_k) * u_bar_hat
    elseif inv_type == "MAP"
        u_hat = @. (g_k / (g_k^2 + λ)) * u_bar_hat
    end
    
    u_inv = irfft(u_hat, n_u)

    if n_les < n_dns
        u = inv_filter_cutoff(u_inv, n_dns, n_les)
    else
        u = u_inv
    end
    u
end


function f_ideal_fourier(v_hat, p, t=nothing)
    (; grid, n_bar, n_sample, filter_type, Δ, λ) = p
    (; L, n) = grid
    p_solve = (; grid=grid)
    n_bar_hat = length(v_hat)
    v = irfft(v_hat, n_bar)
    dv_hat = zeros(n_bar_hat)
    # noise = maximum(abs.(v))/2500000
    noise = 0.0000
    for i = 1:n_sample
        u = inv_filter(v.+ noise.* rand.(), L, n, n_bar, filter_type, Δ, λ)
        u_hat = rfft(u)
        f_hat_sol = f_fourier(u_hat, p_solve)
        f_sol = irfft(f_hat_sol, n)
        fbar = filter(f_sol, L, n, n_bar, filter_type, Δ)
        fbar_hat = rfft(fbar)
        dv_hat += fbar_hat
    end    
    dv_hat / n_sample
end


function f_ideal_disc(v, p, t=nothing)
    (; grid, n_bar, n_sample, filter_type, Δ, λ) = p
    (; L, n) = grid
    p_solve = (; grid=grid)
    dv = zeros(n_bar)
    noise = maximum(abs.(v))/25
    # noise = 0
    for i = 1:n_sample
        u = inv_filter(v.+ noise .* rand.(), n, n_bar, filter_type, Δ, λ)
        f_sol = f_disc(u, p_solve)
        fbar = filter(f_sol, n, n_bar, filter_type, Δ)
        dv += fbar
    end    
    dv / n_sample
end

function sample_u(u, n_dns, n_les)
    K_dns = div(n_dns, 2)
    K_les = div(n_les, 2)
    k = 0:K_dns
    u_hat = rfft(u)
    u_new_hat = u_hat
    for i = K_les+2:K_dns+1
        a_i = 10*abs(u_hat[i])
        u_new_hat_i = a_i * exp(2 * π * im * rand())
        u_new_hat[i] = u_new_hat_i
    end    
    u_new = irfft(u_new_hat, n_dns)
    u_new
end   


function spectrum(u, grid)
    uhat = rfft(u)[2:end]
    n = grid.n
    k = rfftfreq(n) * n
    k = k[2:end]
    s = abs2.(uhat) / 2n^2
    k, s
end

function spectrum_to_hat(s, n)
    n_hat = length(s) + 1
    u_hat = zeros(Complex{Float64}, n_hat)
    u_hat[1] = 0.0
    for i = 2:n_hat
        u_hat[i] = sqrt(s[i-1]*2*n^2) * exp(2 * pi * im * rand())
    end
    u_hat
end    

function bisection(f::Function, a::Number, b::Number;
                   tol::AbstractFloat=1e-15, maxiter::Integer=100000)
    fa = f(a)
    fa*f(b) <= 0 || error("No real root in [a,b]")
    i = 0
    local c
    while b-a > tol
        i += 1
        i != maxiter || error("Max iteration exceeded")
        c = (a+b)/2
        fc = f(c)
        if fc == 0
            break
        elseif fa*fc > 0
            a = c  # Root is in the right half of [a,b].
            fa = fc
        else
            b = c  # Root is in the left half of [a,b].
        end
    end
    return c
end


function ks_etdrk4(u₀, p, t_range, Δt)
    # solution of Kuramoto-Sivashinsky
    # u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,32*pi]
    # computation is based on v = fft(u), so linear term is diagonal
    # the code below is inspired by Kassam and Trefethen's Matlab code

    # problem specifications
    (; grid) = p
    (; L, n) = grid
    l = L
    N = n
    space_interval = (0.0,l)

    # spatial discretization
    x = space_interval[1] .+ space_interval[2] * (1:N)/N
    u = u₀
    v = rfft(u)

    # precompute ETDRK4 coefficients
    # here we utilize that things are diagonal, which means that we can store
    # the matrices as vectors instead
    # k = [0:(N/2-1); 0; (-N/2+1):-1] * (2 * pi) / l # wave numbers
    K = div(n, 2)
    k = [0:K;] * (2 * pi) / l
    L = k .^ 2 - k .^ 4
    E = exp.(Δt*L)
    E2 = exp.(Δt*L/2) 
    M = 16 # number of points for complex mean
    r = exp.(1im*pi*((1:M).-.5)/M); # roots of unity
    # LR: 2d array with each row representing one eigenvalue, and the different points to be evaluated for it
    LR = Δt*reshape(repeat(L,M), ((K+1),M)) + reshape(repeat(r,(K+1)), (M,(K+1)))'
    Q = Δt*real(mean( (exp.(LR/2).-1)./LR , dims = 2))
    f1 = Δt*real(mean( (-4 .- LR .+ exp.(LR) .* (4 .- 3*LR+LR.^2))./LR.^3 , dims = 2))
    f2 = Δt*real(mean( (2 .+LR+exp.(LR).*(-2 .+ LR))./LR.^3 , dims = 2))
    f3 = Δt*real(mean( (-4 .-3*LR-LR.^2+exp.(LR) .* (4 .-LR))./LR.^3 , dims = 2))

    # main time-stepping loop
    uu = [u]
    tt = [t_range[1]]
    tmax = t_range[2]
    nmax = round(tmax/Δt)
    nplt = 1

    g = -0.5im*k
    for n in 1:nmax
        t = n*Δt
        Nv = g.*rfft(irfft(v, N).^2)
        a = E2.*v + Q.*Nv
        Na = g.*rfft(irfft(a, N).^2)
        b = E2.*v + Q.*Na
        Nb = g.*rfft(irfft(b, N).^2)
        c = E2.*a + Q.*(2*Nb-Nv)
        Nc = g.*rfft(irfft(c, N).^2)
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3
        if mod(n,nplt) == 0
            u = reshape(irfft(v, N), N)
            push!(uu,u)
            push!(tt,t)
        end
    end
    return (tt, uu)
end

function f_etdrk4_coeff(p, Δt)
    # solution of Kuramoto-Sivashinsky
    # u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,32*pi]
    # computation is based on v = fft(u), so linear term is diagonal
    # the code below is inspired by Kassam and Trefethen's Matlab code

    # problem specifications
    (; grid) = p
    (; L, n) = grid
    l = L
    N = n

    # precompute ETDRK4 coefficients
    # here we utilize that things are diagonal, which means that we can store
    # the matrices as vectors instead
    # k = [0:(N/2-1); 0; (-N/2+1):-1] * (2 * pi) / l # wave numbers
    K = div(n, 2)
    k = [0:K;] * (2 * pi) / l
    L = k .^ 2 - k .^ 4
    E = exp.(Δt*L)
    E2 = exp.(Δt*L/2) 
    M = 16 # number of points for complex mean
    r = exp.(1im*pi*((1:M).-.5)/M); # roots of unity
    # LR: 2d array with each row representing one eigenvalue, and the different points to be evaluated for it
    LR = Δt*reshape(repeat(L,M), ((K+1),M)) + reshape(repeat(r,(K+1)), (M,(K+1)))'
    Q = Δt*real(mean( (exp.(LR/2).-1)./LR , dims = 2))
    f1 = Δt*real(mean( (-4 .- LR .+ exp.(LR) .* (4 .- 3*LR+LR.^2))./LR.^3 , dims = 2))
    f2 = Δt*real(mean( (2 .+LR+exp.(LR).*(-2 .+ LR))./LR.^3 , dims = 2))
    f3 = Δt*real(mean( (-4 .-3*LR-LR.^2+exp.(LR) .* (4 .-LR))./LR.^3 , dims = 2))

    g = -0.5im*k

    return (g, E2, Q, f1, f2, f3, E)
end


function f_etdrk4(u_hat, p, Δt, g, E2, Q, f1, f2, f3, E)
    # solution of Kuramoto-Sivashinsky
    # u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,32*pi]
    # computation is based on v = fft(u), so linear term is diagonal
    # the code below is inspired by Kassam and Trefethen's Matlab code

    # problem specifications
    (; grid) = p
    (; L, n) = grid
    l = L
    N = n

    Nu_hat = g.*rfft(irfft(u_hat, N).^2)
    a = E2.*u_hat + Q.*Nu_hat
    Na = g.*rfft(irfft(a, N).^2)
    b = E2.*u_hat + Q.*Na
    Nb = g.*rfft(irfft(b, N).^2)
    c = E2.*a + Q.*(2*Nb-Nu_hat)
    Nc = g.*rfft(irfft(c, N).^2)
    u_hat_new = E.*u_hat + Nu_hat.*f1 + 2*(Na+Nb).*f2 + Nc.*f3
    
    fu_hat = (u_hat_new - u_hat) / Δt
    
    fu_hat
end

function uhat_etdrk4(u_hat, p, Δt, g, E2, Q, f1, f2, f3, E)
    # solution of Kuramoto-Sivashinsky
    # u_t = -u*u_x - u_xx - u_xxxx, periodic BCs on [0,32*pi]
    # computation is based on v = fft(u), so linear term is diagonal
    # the code below is inspired by Kassam and Trefethen's Matlab code

    # problem specifications
    (; grid) = p
    (; L, n) = grid
    l = L
    N = n

    Nu_hat = g.*rfft(irfft(u_hat, N).^2)
    a = E2.*u_hat + Q.*Nu_hat
    Na = g.*rfft(irfft(a, N).^2)
    b = E2.*u_hat + Q.*Na
    Nb = g.*rfft(irfft(b, N).^2)
    c = E2.*a + Q.*(2*Nb-Nu_hat)
    Nc = g.*rfft(irfft(c, N).^2)
    u_hat_new = E.*u_hat + Nu_hat.*f1 + 2*(Na+Nb).*f2 + Nc.*f3
    
    u_hat_new
end

function inv_filter_MC(u_bar, L, n_dns, n_les, filter_type, Δ=0.01, B=100)
    σ_eta_2 = 10^-2
    u_bar_hat = rfft(u_bar)
    u_inv_hat = zeros(Complex{Float64}, div(n_dns,2)+1)
    samples = zeros(Complex{Float64}, div(n_dns,2)+1, B)
    for j = 1:div(n_dns,2)
        q = zeros(B)
        w = zeros(B)
        
        if j > div(n_les,2)
            # for i =1:B
            #     q[i] = exp(-1/(2*σ_eta_2) *  abs2(0.0 - n_les*samples_B[j+1,1,i]*g ))
            #     if q[i] < 1e-16
            #         q[i] = 0.0
            #     end
            # end
        else
            u_hat_j =  u_bar_hat[j+1]
            u_hat_j_a = abs(u_hat_j)
            u_hat_j_θ = angle(u_hat_j)
            g = KdV.g(j, Δ, filter_type, L)
            for i =1:B
                a_new = (n_dns/n_les) * (u_hat_j_a + 0.2 * u_hat_j_a * randn()) / g
                θ_new = u_hat_j_θ + 0.1 * randn()
                u_hat_j_new = a_new * exp(im * θ_new)
                samples[j+1, i] = u_hat_j_new
                q[i] = exp(-1/(2*σ_eta_2) *  abs2(u_hat_j*n_dns - n_les*u_hat_j_new*g ))
                if q[i] < 1e-16
                    q[i] = 0.0
                end
            end
        end
        q_sum = sum(q)

        if q_sum == 0.0
            w = ones(B) / B
        else
            for i =1:B
                w[i] = q[i] / q_sum
            end
        end
        
        # print(q_sum, " ")
        for l = 1:100
            sample_ui = rand(Multinomial(1, w))
            sample = argmax(sample_ui)
            u_inv_hat[j+1] += samples[j+1,sample]
        end
        u_inv_hat[j+1] /= 100.0
    end
    u_inv = irfft(u_inv_hat, n_dns) 

    u_inv
end

function step_rk4(f, u, p, dt)
    k1 = f(u, p)
    k2 = f(u + 1 / 2 * k1 * dt, p)
    k3 = f(u + 1 / 2 * k2 * dt, p)
    k4 = f(u + k3 * dt, p)
    u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
end

function solve_ode(step, f, u_hat, p, nstep, dt; nsave = 10)
    t = 0.0
    usave = zeros(Complex{Float64}, length(u_hat), Int((nstep / nsave) +1))
    j = 1
    usave[:,j] = u_hat
    for i = 1:nstep
        u_hat = step(f, u_hat, p, dt)
        t += dt
        if i % nsave == 0
            j += 1
            usave[:,j] = u_hat
        end
    end
    usave
end

function solve_KdV(solve_type, u, p, t_range, Δt; saveat=nothing, ideal=false, n_sample=10, filter_type="gaussian", Δ=0.01, λ=0.01, inv_type="MAP")
    if saveat == nothing
        saveat = t_range[2]
    end
    (; grid) = p
    (; L, n) = grid
    n_u = length(u)
    if solve_type == "disc"
        if ideal == true
            f = f_ideal_disc
            p_solve = (; grid=grid, n_bar = n_u, n_sample=n_sample, filter_type=filter_type, Δ=Δ, λ=λ)
        else
            f = f_disc
            p_solve = p
        end
        prob = ODEProblem(f, u, t_range, p_solve, saveat=saveat)
        # sol = Array(solve(prob, Tsit5()) )
        sol = Array(solve(prob, Rodas5P(autodiff = AutoFiniteDiff()), dt = Δt, adaptive=false) )
    elseif solve_type == "Fourier"
        if ideal == true
            f = f_ideal_fourier
            p_solve = (; grid=grid, n_bar = n_u, n_sample=n_sample, filter_type=filter_type, Δ=Δ, λ=λ)
        else
            f = f_fourier
            p_solve = p
        end
        u_hat = rfft(u)

        # prob = ODEProblem(f, u_hat, t_range, p_solve, saveat=saveat)
        # sol_hat = Array(solve(prob, RK4(), dt = Δt, adaptive=false) )
        if saveat == nothing
            nsave = 1
        else
            nsave = Int(saveat[2]/ Δt)
        end
        sol_hat = solve_ode(step_rk4, f, u_hat, p, Int(t_range[end]/ Δt), Δt; nsave = nsave)

        # if true
        # prob = ODEProblem(f, u_hat, t_range, p_solve, save_everystep = false, saveat=saveat)
        # sol_hat = Array(solve(prob, RK4(), dt = Δt, adaptive=false) )
        # else
        #     solve_manual()
        # end


        sol = zeros(n_u, size(sol_hat, 2))
        i = 1
        for u_hat in eachcol(sol_hat[:, 1:end])
            sol_i = irfft(u_hat, n_u)
            sol[:,i] = sol_i
            i += 1
        end
    elseif solve_type == "Fourier_ETDRK4"
        
        nstep = Int( (t_range[2] - t_range[1]) / Δt )
        g, E2, Q, f1, f2, f3, E = f_etdrk4_coeff(p, Δt)
        sol = zeros(n_u, nstep+1)
        sol[:, 1] = u
        if ideal == true
            noise = 0.0
            for i=1:nstep
                uᵢ = sol[:, i]
                uᵢ_hat = rfft(uᵢ)
                du_bar_hat_sum = zeros(div(n_u, 2)+1)
                u_hat_new_sum = zeros(div(n_u, 2)+1)
                for j = 1:n_sample
                    u_dns = inv_filter(uᵢ .+ noise .* rand.(), L, n, n_u, filter_type, Δ, λ, inv_type)
                    # u_dns = inv_filter_MC(u₀_bar, L, n_dns, n_les, filter_type, Δ, 100)
                    u_dns_hat = rfft(u_dns)

                    u_dns_hat_new = uhat_etdrk4(u_dns_hat, p, Δt, g, E2, Q, f1, f2, f3, E)
                    u_dns_new = irfft(u_dns_hat_new, n)
                    u_bar_new = filter(u_dns_new, L, n, n_u, filter_type, Δ)
                    u_hat_new_sum += rfft(u_bar_new)

                end
                u_hat_new = u_hat_new_sum / n_sample
                #     f_hat_sol = f_etdrk4(u_dns_hat, p, Δt, g, E2, Q, f1, f2, f3, E)
                #     f_sol = irfft(f_hat_sol, n)
                #     fbar = filter(f_sol, L, n, n_u, filter_type, Δ)
                #     fbar_hat = rfft(fbar)
                #     du_bar_hat_sum += fbar_hat
                # end
                # du_bar_hat = du_bar_hat_sum / n_sample
                # u_hat_new = uᵢ_hat + Δt * du_bar_hat
                u_new = irfft(u_hat_new, n_u)
                sol[:, i+1] = u_new
            end

        else
            for i=1:nstep
                uᵢ = sol[:, i]
                uᵢ_hat = rfft(uᵢ)
                u_hat_new = uhat_etdrk4(uᵢ_hat, p, Δt, g, E2, Q, f1, f2, f3, E)
                # du_hat = f_etdrk4(uᵢ_hat, p, Δt, g, E2, Q, f1, f2, f3, E)
                # u_hat_new = uᵢ_hat + Δt * du_hat
                u_new = irfft(u_hat_new, n)
                sol[:, i+1] = u_new
            end
        end


    end
    
    sol
end

#   "Create initial condition"
function randomfield(n, kpeak, amp, warmup=false; solve_type=nothing, p=nothing, Δt=0.01)
    K = div(n, 2)
    k = 0:K
    c = @. amp * k^4 * exp(-2(k / kpeak)^2 + 2π * im * rand())
    u₀ = irfft(c, n)
    if warmup
        u₀ = solve_KS(solve_type, u₀, p, (0,5Δt), Δt)[:,end]
    end
    u₀
end


function randominitialstate(T, N, K)
    u₀ = zeros(N)
    u_hat_plus = randn(Complex{T}, K)
    u_hat_min = randn(Complex{T}, K)
    for l=1:N
        e_plus = 0
        e_min = 0
        for k=1:K
            e_plus += u_hat_plus[k] * exp(im * 2π * k * l ./ N)
            e_min += u_hat_min[k] * exp(-im * 2π * k * l ./ N)
        end
        e_real = real(e_plus + e_min)
        u₀[l] = e_real
    end
    u₀_max = maximum(abs.(u₀))
    u₀ .= (u₀ ./ u₀_max) .* 2
end
# u₀ = randominitialstate(Float32, grid.n, 50)
