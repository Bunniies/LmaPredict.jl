module DataIO

    using DelimitedFiles, Statistics, OrderedCollections

    mutable struct CnfgEigData
        id::String
        gamma::String
        eigmodes::Int64
        tsrc::Int64
        data::OrderedDict{String, Vector{Float64}}
        CnfgEigData(id, gamma, eigmodes, tsrc, data) = new(id, gamma, eigmodes, tsrc, data)
    end
    export CnfgEigData
    
    include("DataConst.jl")

    include("DataReader.jl")
    export read_eigen_eigen, read_rest_rest, read_rest_eigen


end