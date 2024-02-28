module LmaPredict

include("DataIO/DataIO.jl")
include("PredictionAnalysis/PredictionAnalysis.jl")

using .DataIO, .PredictionAnalysis
export LMAConfig
export read_eigen_vals, read_eigen_eigen, read_rest_rest, read_rest_eigen, get_LMAConfig, analyse_predictions

end # module
