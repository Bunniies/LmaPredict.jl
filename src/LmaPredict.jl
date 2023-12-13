module LmaPredict

include("DataIO/DataIO.jl")

using .DataIO
export LMAConfig
export read_eigen_eigen, read_rest_rest, read_rest_eigen, get_LMAConfig

end # module
