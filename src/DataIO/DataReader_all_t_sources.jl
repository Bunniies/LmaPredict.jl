@doc raw"""
    read_contrib_all_sources(path::String, g::String)

Given a path and a gamma structre g as defined in DataConst.GAMMA, this function reads the eigen-eigen, rest-rest or rest-eigen contributions of LMA for a single configuration and  for all the source positions listed in the dat file.
These are files with mseig**xx.dat extension, where xx=ee,rr,re  

It returns an OrderedDict of Vector{Float64}  where the keys are the source positions while the values are Vector{Float64} containing the eigen-eigen contribution of the corresponding key with length equals the timeslices T of the lattice.
"""
function read_contrib_all_sources(path::String, g::String)
    if !(g in GAMMA)
        error("The gamma structure $(g) is not supported. Please update the GAMMA database in DataConst.jl")
    end

    f = readdlm(path)
    # tvals = parse(Int64, split(filter(x-> typeof(x)<:AbstractString && occursin("#T=", x), f)[1], "=")[end])
    dlm_tsrc = findall(x-> typeof(x)<:AbstractString && occursin("#tsrc", x), f)
    dlm_g    = findall(x-> typeof(x)<:AbstractString && occursin("#"*g, x), f)

    idx_time = findall(x-> typeof(x)<:AbstractString && occursin("#", x), f[dlm_g[1]:dlm_g[2]])
    tvals = length(f[idx_time[1].I[1]+1 : idx_time[2].I[1]-1, 1])
    # println("tvals= ", tvals)

    if length(dlm_tsrc) != length(dlm_g)
        error("Found $(length(dlm_tsrc)) sources delimiters and $(length(dlm_g)) gammas delimiters in $(path) \n Please check that the required gamma structure has been computed for each source position listed in the file.")
    end

    datadict = OrderedDict{String, Vector{Float64}}()

    for k in eachindex(dlm_tsrc)

        tsrc = split(f[dlm_tsrc[k]],"=")[end]
        idx = dlm_g[k].I[1]
        datadict[tsrc] = Float64.(f[idx+1:idx+tvals, 2])
    end

    return datadict

end

function get_LMAConfig_all_sources(path::String, g::String; em::String="VV", bc::Bool=false)

    modes = Dict("PA"=>32, "VV"=> 64)
    f = readdir(path)

    p_ee = filter(x->occursin(string("mseig", em, "ee"), x), f)
    p_re = filter(x->occursin(string("mseig", em, "re"), x), f)
    p_rr = filter(x->occursin(string("mseig", em, "rr"), x), f)
   
    if length(p_ee) == 0  || length(p_re)  == 0 || length(p_rr)  == 0 
        error("No dat files found for at least one of  \n - mseig$(em)ee.dat \n - mseig$(em)re_ts \nCheck your files in $(path)")
    end

    res_dict = Dict()
    res_dict["ee"] = read_contrib_all_sources(joinpath(path, p_ee[1]), g)
    res_dict["re"] = read_contrib_all_sources(joinpath(path, p_re[1]), g)
    res_dict["rr"] = read_contrib_all_sources(joinpath(path, p_rr[1]), g)

    if bc
        error("option to read bias correction data not yet implemented")
    end

    return LMAConfig(parse(Int64, basename(path)), g, modes[em], res_dict)
end