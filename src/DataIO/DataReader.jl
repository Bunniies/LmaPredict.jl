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

function read_LMA_per_config(path::String, g::String; em::String="PA")

    modes = Dict("PA"=>32, "VV"=> 64)
    

end