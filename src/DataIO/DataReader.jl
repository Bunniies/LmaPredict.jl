@doc raw"""
    read_eigen_vals(path::String, g::String)

Given a path and a gamma structre g as defined in DataConst.GAMMA, this function reads the eigenvalues for a single configuration listed in the dat file.  

It returns a Vector{Float64} with the eigenvalues.
"""
function read_eigen_vals(path::String, g::String)
    
    if !(g in GAMMA)
        error("The gamma structure $(g) is not supported. Please update the GAMMA database in DataConst.jl")
    end

    f = readdlm(path)
    eigvals = filter(entry -> typeof(entry) <: Float64, f[:,1])

    return eigvals
end

@doc raw"""
    read_eigen_eigen(path::String, g::String)

Given a path and a gamma structre g as defined in DataConst.GAMMA, this function reads the eigen-eigen contribution of LMA for a single configuration and  for all the source positions listed in the dat file.
These are files with mseig**ee.dat extension.  

It returns an OrderedDict of Vector{Float64}  where the keys are the source positions while the values are Vector{Float64} containing the eigen-eigen contribution of the corresponding key with length equals the timeslices T of the lattice.
"""
function read_eigen_eigen(path::String, g::String)
    
    if !(g in GAMMA)
        error("The gamma structure $(g) is not supported. Please update the GAMMA database in DataConst.jl")
    end

    f = readdlm(path)
    tvals = parse(Int64, split(filter(x-> typeof(x)<:AbstractString && occursin("#T=", x), f)[1], "=")[end])

    dlm_tsrc = findall(x-> typeof(x)<:AbstractString && occursin("#tsrc", x), f)
    dlm_g    = findall(x-> typeof(x)<:AbstractString && occursin("#"*g, x), f)

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

@doc raw"""
    read_rest_rest(path::String, g::String)

Given a path and a gamma structre g as defined in DataConst.GAMMA, this function reads the rest-rest contribution of LMA for a single configuration and a given source position.
These are files with mseig**rr_ts*.dat extension.  

It returns a Vector{Float64} with length equals the timeslices T of the lattice.
"""
function read_rest_rest(path::String, g::String)

    if !(g in GAMMA)
        error("The gamma structure $(g) is not supported. Please update the GAMMA database in DataConst.jl")
    end

    f = readdlm(path, skipstart=1)

    dlm_idx = findall(x-> typeof(x)<:AbstractString && occursin("#", x), f)
    dlm_g = filter(x-> f[x]=="#"*g , dlm_idx)
    
    if length(dlm_g) != 1
        error("In $(path) a number n=$(length(dlm_g)) of gamma structure delimiters where found. \n If n=0, the requested gamma structure was not found. If n>1 multiple instances occurred. \n Please check the structure of the file.")
    end
    
    tvals = Int64((size(f)[1] - size(dlm_idx)[1]) / size(dlm_idx)[1])
    data = Array{Float64}(undef, tvals)

    idx = dlm_g[1].I[1]
    data = f[idx+1:idx+tvals, 2]
    return data
end

@doc raw"""
    read_rest_eigen(path::String, g::String; bc::Bool=false)

Given a path and a gamma structre g as defined in DataConst.GAMMA, this function reads the rest-eigen contribution of LMA for a single configuration and a given source position.
These are files with mseig**re_ts*.dat extension.  

The flag bc allows to read the bias correction stored. 
If bc=false, it returns a Vector{Float64} with length equals the timeslices T of the lattice.  

If bc=true, it returns a Tuple(Vector{Float64}, Vector{Float64}), each with length equals the timeslices T of the lattice, where the first entry is the rest-eigen part while the second entry is the bias correction.
    
"""
function read_rest_eigen(path::String, g::String; bc::Bool=false)
    if !(g in GAMMA)
        error("The gamma structure $(g) is not supported. Please update the GAMMA database in DataConst.jl")
    end

    f = readdlm(path, skipstart=1)
    dlm_idx = findall(x-> typeof(x)<:AbstractString && occursin("#", x), f)
    dlm_g = filter(x-> f[x]=="#"*g , dlm_idx)
    dlm_bc = filter(x->f[x]=="#eigen-rest+rest-eigen biascorrection", dlm_idx)

    tvals = length(f[dlm_idx[1]:dlm_idx[2]]) -2
    data = Array{Float64}(undef, tvals)

    idx_data = dlm_g[1].I[1]
    data = f[idx_data+1:idx_data+tvals, 2]
    if bc 
        data_bc = similar(data)
        idx_data_bc = dlm_g[2].I[1]
        if idx_data < dlm_g[1].I[1]
            error("The delim index of the bias correction for gamma $(g) is not below the bias correction delim $(dlm_bc)")
        end
        data_bc =  f[idx_data_bc+1:idx_data_bc+tvals, 2]
        return data, data_bc
    else
        return data
    end
end

@doc raw"""
    get_LMAConfig(path::String, g::String; em::String="PA", bc::Bool=false)

This function reads the eigenvalues, rr, re and ee  LMA contributions with gamma structure 'g' for all available source positions saved in 'path' with the standard nomenclature.

The following flags are available:  

    - bc::Bool=false  : is set to true re bias correction are read.

    - em::String="PA" : if em="PA" reads files with 32 eigenmodes. If em="VV" reads files with 64 eigenmodes.

It returns a LMAConfig object with the following attributes:  

    - ncnfg : number of the configuration read

    - gamma : gamma structure read

    - eigmodes : number of eigenmodes 

    - data : dictionary with keys "eigvals", "rr", "re" and "ee" containing the eigenvalues and the corresponding contributions for each source position detected.

Examples:
```@example
p = path/to/config_repo/#
lmacnfg = get_LMAConfig(p, "g5-g5", em="PA", bc=true)

lmacnfg.ncnfc    # number of the processed config
lmacnfg.gammma   # gamma structured selected
lmacnfg.eigmodes # number of eigenmodes used for LMA
lmacnfg.data     # Dict containing "eigvals", "rr", "re" and "ee" contributions

lmacnfg.data["eigvals"] # Array containing the eigenvalues detected in p
lmacnfg.data["rr"] # Dict containing "rr" contributions for each source position detected in p
lmacnfg.data["re"] # Dict containing "re" contributions for each source position detected in p
lmacnfg.data["ee"] # Dict containing "ee" contributions for each source position detected in p

lmacnfg.data["rr"]["0"] #  "rr" contribution at source position 0 
lmacnfg.data["re"]["0"] #  "re" contribution at source position 0 
lmacnfg.data["ee"]["0"] #  "ee" contribution at source position 0 

```   
"""
function get_LMAConfig(path::String, g::String; em::String="PA", bc::Bool=false)

    modes = Dict("PA"=>32, "VV"=> 64)

    f = readdir(path)

    p_eigvals = filter(x->occursin(string("eigvals"), x), f)
    p_ee = filter(x->occursin(string("mseig", em, "ee"), x), f)
    p_re = filter(x->occursin(string("mseig", em, "re_ts"), x), f)
    p_rr = filter(x->occursin(string("mseig", em, "rr_ts"), x), f)

    if length(p_eigvals) == 0 || length(p_ee) == 0  || length(p_re)  == 0 || length(p_rr)  == 0 
        error("No dat files found for at least one of \n - eigvals.dat \n - mseig$(em)ee.dat \n - mseig$(em)re_ts \nCheck your files in $(path)")
    end

    res_dict = Dict()

    # eigenvalues
    eigvals = read_eigen_vals(joinpath(path, p_eigvals[1]), g)
    res_dict["eigvals"] = eigvals

    # ee
    dict_ee = read_eigen_eigen(joinpath(path, p_ee[1]), g)
    res_dict["ee"] = dict_ee
    res_dict["rr"] = OrderedDict{String, Vector{Float64}}()
    
    # rr
    tsrc = vcat([map(eachmatch(r"[0-9]+"*".dat", p_rr[k])) do m string(getindex(split(m.match, "."), 1)) end for k in eachindex(p_rr)]...)
    sorted_idx = sortperm(parse.(Int64, tsrc))
    for k in sorted_idx
        res_dict["rr"][tsrc[k]] = read_rest_rest(joinpath(path, p_rr[k]), g) 
    end

    # re
    tsrc = vcat([map(eachmatch(r"[0-9]+"*".dat", p_re[k])) do m string(getindex(split(m.match, "."), 1)) end for k in eachindex(p_re)]...)
    sorted_idx = sortperm(parse.(Int64, tsrc))
    res_dict["re"] = OrderedDict{String, Vector{Float64}}()
    if !bc 
        for k in sorted_idx
            res_dict["re"][tsrc[k]] = read_rest_eigen(joinpath(path, p_re[k]), g)
        end
    else
        res_dict["re_bc"] = OrderedDict{String, Vector{Float64}}()
        for k in sorted_idx
            res_dict["re"][tsrc[k]], res_dict["re_bc"][tsrc[k]] = read_rest_eigen(joinpath(path, p_re[k]), g, bc=true)
        end
    end

    return LMAConfig(parse(Int64,basename(path)), g, modes[em], res_dict)
end