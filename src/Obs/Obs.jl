module Obs

    using ADerrors, Statistics

    mutable struct Corr
        obs::Vector{uwreal}
        id::String
        gamma::String
        Corr(a::Vector{uwreal}, id::String, gamma::String) = new(a, id, gamma) 

    end
    function Base.show(io::IO, corr::Corr)
        println(io, "Correlator")
        println(io, " - Ensemble ID: ", corr.id)
        println(io, " - Gamma:       ", corr.gamma)
    end
    export Corr

    include("ObsLma.jl")
    export corr_obs_rr

end