using Revise, LmaPredict

# path to dat file for config 1
pp_rr = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/1/A654r000n1.mseigPArr_ts0.dat"
pp_re = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/1/A654r000n1.mseigPAre_ts0.dat"
pp_ee = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/1/A654r000n1.mseigPAee.dat"
# read rest rest G5G5 contribution 
dat_rr = read_rest_rest(pp_rr, "g5-g5")

# read rest eigen G5G5 contribution without bias correction
data_rw = read_rest_eigen(pp_re, "g5-g5", bc=false)

# read rest eigen G5G5 contribution with bias correction
data_rw, data_rw_bc = read_rest_eigen(pp_re, "g5-g5", bc=true)

# read eigen eigen G5G5 contribution
data_ = read_eigen_eigen(pp_ee, "g5-g5")

