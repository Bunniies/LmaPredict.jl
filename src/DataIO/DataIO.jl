module DataIO

    using DelimitedFiles, Statistics, OrderedCollections

    mutable struct LMAConfig
        ncnfg::Int64
        gamma::String
        eigmodes::Int64
        data::Dict{Any, Any}
        LMAConfig(ncnfg, gamma, eigmodes, data) = new(ncnfg, gamma, eigmodes, data)
    end
    function Base.show(io::IO, a::LMAConfig)
        println(io, "LMAConfig")
        println(io, " - Ncnfg :        ", a.ncnfg)
        println(io, " - Gamma :        ", a.gamma)
        println(io, " - Eigen modes :  ", a.eigmodes)
    end
    export LMAConfig
    
    include("DataConst.jl")

    include("DataReader.jl")
    export read_eigen_vals, read_eigen_eigen, read_rest_rest, read_rest_eigen, get_LMAConfig


end