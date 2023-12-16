using Revise, LmaPredict

## Check individual reading routines for rr, ee and re contribution
# path to dat file for config 1
pp_rr = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/1/A654r000n1.mseigPArr_ts0.dat"
pp_re = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/1/A654r000n1.mseigPAre_ts0.dat"
pp_ee = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/1/A654r000n1.mseigPAee.dat"

# read rest rest G5G5 contribution 
dat_rr = read_rest_rest(pp_rr, "g5-g5")

# read rest eigen G5G5 contribution without bias correction
data_re = read_rest_eigen(pp_re, "g5-g5", bc=false)

# read rest eigen G5G5 contribution with bias correction
data_re, data_re_bc = read_rest_eigen(pp_re, "g5-g5", bc=true)

# read eigen eigen G5G5 contribution
data_ee = read_eigen_eigen(pp_ee, "g5-g5")


## Reading for a given config the rr, ee and re contributions 
# for a given tsource and gamma structure. Data stored in CnfgEigData

path_config = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/1"

lmacnfg = get_LMAConfig(path_config, "g5-g5", em="PA", bc=true)

## Exploring data
lmacnfg.data["rr"]
lmacnfg.data["ee"]
lmacnfg.data["re"]

##
using PyPlot, LaTeXStrings
TSRC = "0"

rr = lmacnfg.data["rr"][TSRC]
ee = lmacnfg.data["ee"][TSRC]
re = lmacnfg.data["re"][TSRC]
tot = rr + ee + re

xx = collect(1:length(rr))
scatter(xx, rr, label="r-r")
scatter(xx, ee, label="e-e")
scatter(xx, re, label="r-e")
scatter(xx, tot, label="tot")
xlabel(L"$x_0/a$")
legend()
display(gcf())
PyPlot.close()
#

