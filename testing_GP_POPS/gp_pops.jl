using LinearAlgebra, Distributions, Random, Plots

# ----------------------------
# 1. Define a toy ground-truth function
# ----------------------------
# (replace the simple f_true and small dataset with a much harder, nonstationary
# function and many more training points)
function f_true(x)
    # multi-scale base + amplitude modulation
    base = sin(3x) + 0.5*cos(9x)
    modulation = 0.5*sin(2x) * (1 + 0.8*sin(25x))    # slowly varying amplitude with fast wiggles

    # localized high-frequency burst (sharp, non-smooth feature)
    burst = 0.9 * cos(60x) * exp(-40*(x - 1.0)^2)

    # very sharp, narrow spikes (near-discontinuous behavior)
    spikes = 0.8*exp(-200*(x - 0.5)^2) + 0.6*exp(-300*(x - 2.0)^2)

    # a step/shift
    step = x > π ? 0.7 : 0.0

    return base + modulation + burst + spikes + step
end

Random.seed!(0)

# Much larger training set (adjust Ntrain if memory/compute is an issue)
Ntrain = 2000          # "way way more data"
X = collect(range(0, 2π; length=Ntrain))
y = f_true.(X) .+ 0.05 .* randn(Ntrain)

# Denser test grid
Xtest = collect(range(0, 2π; length=2000))
ytrue = f_true.(Xtest)

# ----------------------------
# 2. Define an RBF kernel (too smooth on purpose)
# ----------------------------
function kernel(x1::AbstractVector, x2::AbstractVector; ℓ=1.0, σf=1.0)
    n1, n2 = length(x1), length(x2)
    K = zeros(Float64, n1, n2)
    for i in 1:n1, j in 1:n2
        K[i,j] = σf^2 * exp(-0.5 * ((x1[i]-x2[j])/ℓ)^2)
    end
    return K
end

ℓ = 1.0      # too large => oversmooths data
σf = 1.0
σn = 0.05

# ----------------------------
# 3. Fit standard GP posterior mean
# ----------------------------
K = kernel(X, X; ℓ=ℓ, σf=σf) + σn^2 * I(length(X))
L = cholesky(K)
α = L \ (L' \ y)  # (K + σn^2 I)^(-1) y

# Predictive mean on grid
Kxx = kernel(X, X; ℓ=ℓ, σf=σf)                # kernel matrix without noise
K = Kxx + σn^2 * I(length(X))                # covariance used for inference (with noise)
L = cholesky(K)
α = L \ (L' \ y)                             # (K + σn^2 I)^(-1) y

# Predictive mean on grid
Kx = kernel(Xtest, X; ℓ=ℓ, σf=σf)
μ = Kx * α

# mean at training points (avoid searching Xtest)
μ_train = Kxx * α

# ----------------------------
# 4. POPS function-space corrections
# ----------------------------
Kdiag = diag(Kxx)                            # k(x_i,x_i) (no noise)
μ_pops = zeros(length(X), length(X))

for i in 1:length(X)
    # Compute the residual at that training point (using μ_train)
    δ = y[i] - μ_train[i]

    # Minimal-norm kernel correction coefficient
    α_i = δ / Kdiag[i]

    # Apply correction across the test grid
    μ_pops[:, i] = μ .+ α_i .* kernel(X, [X[i]]; ℓ=ℓ, σf=σf)[:,1]
end

μ_min = minimum(μ_pops, dims=2)
μ_max = maximum(μ_pops, dims=2)

# ----------------------------
# 5. Plot results
# ----------------------------
plt = plot(Xtest, ytrue, lw=2, label="True f(x)", color=:black)
plot!(plt, Xtest, μ, lw=2, label="GP mean (misspecified)", color=:blue)
plot!(plt, Xtest, μ_min, lw=1, ls=:dash, color=:red, label="POPS envelope")
plot!(plt, Xtest, μ_max, lw=1, ls=:dash, color=:red, label="")
scatter!(plt, X, y; markersize=0.6, markerstrokewidth=0, marker=:dot, color=:black, alpha=0.6, label="Data")
xlabel!("x"); ylabel!("y")
title!("GP misspecification and POPS function-space corrections")
savefig(plt,"plot.png")