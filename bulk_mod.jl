# vol_en     = [[17.813109023568057, -163.06663280425], [18.322054995670012, -163.110046112], 
#              [18.831000967771949, -163.1415028745], [19.339946939873919, -163.162464704125],
#              [19.848892911975845, -163.174242557], [20.3578388840778, -163.177966801875], 
#              [20.866784856179741, -163.17463056225], [21.375730828281693, -163.165132595375], 
#              [21.884676800383637, -163.150242437875], [22.393622772485578, -163.130671150875], 
#              [22.902568744587512, -163.107009425875], [23.411514716689464, -163.079814823625]]
# gap_volumes    = [x[1] for x in vol_en]; gap_energies = [x[2] for x in vol_en];
# pointwise_corrections = corrections(Ap, Y, P; leverage_percentile=0.5, lambda=1/size(Ap,1))
# hypercube_eigs, hypercube_bounds = hypercube(pointwise_corrections; percentile_clipping=5.)
# committee, dΘ = sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params)
# co_ps_vec = [committee[:,i] for i = 1:size(committee,2)]
# ACEpotentials.Models.set_committee!(model, co_ps_vec)
using Pkg; Pkg.add("AtomsCalculators")
using ACEpotentials, DelimitedFiles, ExtXYZ, AtomsCalculators, LinearAlgebra
Pkg.add("Unitful")
using Unitful
suffix = "_17_4"
model, _ = ACEpotentials.load_model("./big_ACE/model$(suffix).json")
A = readdlm("./big_ACE/A$(suffix).csv", ',')
Y = readdlm("./big_ACE/Y$(suffix).csv", ',')
W = readdlm("./big_ACE/W$(suffix).csv", ',')
P = readdlm("./big_ACE/P$(suffix).csv", ',')

Y = Y[:,1]
W = W[:,1]
Ap= Diagonal(W) * A / P
Y = W .* Y
lin_params = P \ (Ap \ Y)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
test = ExtXYZ.load("high_entropy_pops/manual_df_test_Al.xyz")
test_E = []
test_E_predictions = []
test_co_E_predictions = []
ev_val = counter = 0  
co_E_range = []
for (i, at) in enumerate(test)
    try
        energy = at[:dft_energy]
  
        push!(test_E, energy)
        E = ustrip(AtomsCalculators.potential_energy(at, model))
	push!(test_E, E)
    catch
    end
    
end
using Statistics
itr = test_E
RMSE_ = sqrt.(sum(abs2.(itr .- mean(itr))) / (length(itr) - 1))
writedlm("./big_ACE/test_errors_$(RMSE_)_$(suffix).csv", test_E, ',')
# ev_val /= counter
# ev_val *= 100
Pkg.add("CairoMakie")
using CairoMakie

# Create figure and axis
fig = CairoMakie.Figure()
ax = CairoMakie.Axis(fig[1, 1],
    xlabel = "Test errors / eV",
    ylabel = "Uncertainty estimates / eV"
)

# Data
xdata = abs.(test_E)
ydata = co_E_range ./ 2

# Scatter plot
CairoMakie.scatter!(ax, xdata, ydata)

# Add identity line
max_val = maximum([maximum(xdata), maximum(ydata)])
CairoMakie.lines!(ax, [0, max_val], [0, max_val],
    linestyle = :dash,
    label = "Identity"
)

# Add EV label (invisible dummy line)
CairoMakie.lines!(ax, [NaN], [NaN],
    label = "EV = $(ev_val)%",
    color = :white
)

# Axis limits and legend
padding = 0.05 * max_val
CairoMakie.xlims!(ax, 0, max_val + padding)
CairoMakie.ylims!(ax, 0, max_val + padding)
CairoMakie.axislegend(ax, position = :rb)
clipping = "5_5"
# Save to file
CairoMakie.save("./high_entropy_pops/test_error_corner_plot_$(clipping).png", fig)

# Display figure
fig

function volume_energy_curve_(;element, lattice_consts_, model, num_in_committee)
    volumes = zeros(length(lattice_consts_))
    energies = zeros(length(lattice_consts_))
    co_energies = zeros(length(lattice_consts_), num_in_committee)
    for i = 1:length(lattice_consts_)
        a = lattice_consts_[i]u"Å"
        bulk_system = bulk(element, cubic = true; a = a)
        vec_1, vec_2, vec_3 = bulk_system.cell.cell_vectors
        volume = dot(vec_1, cross(vec_2, vec_3))
        E, co_E = @committee AtomsCalculators.potential_energy(bulk_system, model)
        energies[i] = ustrip(E)
        co_energies[i, :] = ustrip(co_E)
        volumes[i] = ustrip(volume)
    end
    return volumes, energies, co_energies
end
lattice_consts = LinRange(400,430,100) ./ 100

volumes, energies, co_energies = volume_energy_curve_(element = :Al, lattice_consts_ = lattice_consts, model = model, num_in_committee = length(co_ps_vec));
volumes_per_atom     = volumes     ./ 8
energies_per_atom    = energies    ./ 8
co_energies_per_atom = co_energies ./ 8;
function OLS(x::Vector, y::Vector; degree::Int=2)
    n = length(x)
    X = zeros(n, degree + 1)
    for i in 0:degree
        X[:, i+1] .= x .^ i
    end
    coeffs = X \ y
    return coeffs, X
end
function QoI(coeffs)
    c0, c1, c2 = coeffs
    c = c0
    b = c1
    a = c2
    discriminant = b^2 - 4*a*c

    if discriminant < 0
        return NaN, NaN, NaN
    end

    v0 = -b / (2 * a)
    e0 = a*v0^2 + b*v0 + c
    B  = - b
    
    B_quantity = B * u"eV/Å^3"           
    B_GPa = uconvert(u"GPa", B_quantity) |> ustrip
    return v0, e0, B_GPa
end
coeffs, X = OLS(volumes_per_atom, energies_per_atom; degree=2)
ace_V0, ace_e0, ace_B     = QoI(coeffs)
bulk_committee = []
ace_v0_committee=[]
for i = 1:length(co_ps_vec)
    coeffs_i, X_i = OLS(volumes_per_atom, co_energies_per_atom[:,i]; degree=2)
    ace_V0_i, ace_e0_i, ace_B_i     = QoI(coeffs_i)
    push!(bulk_committee, ace_B_i)
    push!(ace_v0_committee, ace_V0_i)
end 
writedlm("high_entropy_pops/ACE_bulk_committee_about_$(ace_B)_with_clipping.csv", bulk_committee, ',')
f2 = CairoMakie.Figure()
ax2 = CairoMakie.Axis(f2[1, 1], xticks=([1],["B"]), xlabel="B", ylabel="Energy per atom / eV/Å³", title="Bulk modulus")
for i=1:length(co_ps_vec)
    if (i == 1)
        CairoMakie.scatter!(ax2, 1, bulk_committee[i], label="committee", color = :orange, alpha = 1)
    else
        CairoMakie.scatter!(ax2, 1, bulk_committee[i], color = :orange)
    end
end
CairoMakie.scatter!(ax2, 1, ace_B, label="ACE", color = :blue, alpha = 1.0)
CairoMakie.axislegend(ax2)
CairoMakie.save("./high_entropy_pops/Al_B_with_clipping.png", f2)
f2
