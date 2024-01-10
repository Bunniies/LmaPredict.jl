
using Revise
using LmaPredict, PyPlot

#=========== PLOT VARIABLES SETTINGS =============#
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] =  true
rcParams["mathtext.fontset"]  = "cm"
rcParams["font.size"] =13
rcParams["axes.labelsize"] =24
rcParams["axes.titlesize"] = 18
plt.rc("text", usetex=false) # true if latex installation available 

path_plot="/Users/alessandroconigli/Desktop/postdoc-mainz/projects/MLtoLMA"
path_config = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/"

fname = readdir(path_config)
idx_cut = findall(x->x<=2000, parse.(Int64, fname))
fname = fname[idx_cut]
idx = sortperm( parse.(Int64, fname))
fname = fname[idx]

cnfgarr = Vector{LMAConfig}(undef, 0)
for f in fname
    push!(cnfgarr, get_LMAConfig(joinpath(path_config, f), "g5-g5", em="PA", bc=false))
end

NCNFG = length(cnfgarr)
TVALS = length(cnfgarr[1].data["rr"]["0"]) 

## input manipulation 

TSRC_rr = collect(keys(getfield(cnfgarr[1], :data)["rr"] ))
TSRC_re = collect(keys(getfield(cnfgarr[1], :data)["re"] ))
TSRC_ee = collect(keys(getfield(cnfgarr[1], :data)["ee"] ))

if TSRC_rr != TSRC_re
    @warn "rr and re have different source positions"
end
rr_all_data = Array{Float64}(undef, TVALS, NCNFG, length(TSRC_rr))
re_all_data = Array{Float64}(undef, TVALS, NCNFG, length(TSRC_re))
ee_all_data = Array{Float64}(undef, TVALS, NCNFG, length(TSRC_ee))

for t in 1:TVALS    
    for n in 1:NCNFG
        for (k, ts) in enumerate(TSRC_ee)
            ee_all_data[t, n, k] = getfield(cnfgarr[n], :data)["ee"][ts][t]
        end
        for (k,ts) in enumerate(TSRC_rr)
            rr_all_data[t, n, k] = getfield(cnfgarr[n], :data)["rr"][ts][t]
            re_all_data[t, n, k] = getfield(cnfgarr[n], :data)["re"][ts][t]
        end
    end
end

obs_rr = corr_obs_rr(rr_all_data, "A654"); uwerr.(obs_rr)
obs_re = corr_obs_rr(re_all_data, "A654"); uwerr.(obs_re)
obs_ee = corr_obs_rr(ee_all_data, "A654"); uwerr.(obs_ee)

## plotting correlators
xx = collect(1:length(obs_ee))
errorbar(xx, value.(obs_rr), err.(obs_rr), fmt="d", label=L"$C^{rr}$", mfc="none")
errorbar(xx, value.(obs_re), err.(obs_re), fmt="d", label=L"$C^{re}$", mfc="none")
errorbar(xx, value.(obs_ee), err.(obs_ee), fmt="d", label=L"$C^{ee}$", mfc="none")
legend()
display(gcf())
close("all")