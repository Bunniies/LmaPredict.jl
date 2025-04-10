@doc raw"""
read_all_per_config(path::String, g::String; bc::Bool=false)

Given a path to a given config data file and a gamma structure g, this function return a dictionary containing the rr, re and ee contributions.

If the flag bc is set tu true, the bias correction data for rr and re are included into the dictionary. 
"""
function read_all_per_config(path::String, g::String; bc::Bool=false)
    if !(g in GAMMA)
        error("The gamma structure $(g) is not supported. Please update the GAMMA database in DataConst.jl")
    end
    
    f = readdlm(path)
    fjoint = [join(f[k,:], " ") for k in 1:size(f,1)]
    tvals = parse(Int64, split(filter(x-> typeof(x)<:AbstractString && occursin("#T=", x), f)[1], "=")[end])

    dlm_contrib = Dict()
    dlm_contrib["ee"] =  findall(x->typeof(x)<:AbstractString && occursin("#eigen-eigen correlators", x), fjoint)
    dlm_contrib["re"] =  findall(x->typeof(x)<:AbstractString && occursin("#eigen-rest+rest-eigen correlators", x), fjoint)
    dlm_contrib["rr"] =  findall(x->typeof(x)<:AbstractString && occursin("#rest-rest correlators", x), fjoint)

    dlm_contrib["rr_bc"] =  findall(x->typeof(x)<:AbstractString && occursin("#rest-rest bias", x), fjoint)
    dlm_contrib["re_bc"] =  findall(x->typeof(x)<:AbstractString && occursin("#eigen-rest+rest-eigen bias", x), fjoint)

    dlm_idx = findall(x-> typeof(x)<:AbstractString && occursin("#", x), f)
    dlm_g = filter(x-> f[x]=="#"*g , dlm_idx)

    data = Dict()
    for key in ["ee", "re", "rr"]
        _, idx_ref_g = findmin( abs.(getindex.(dlm_g, 1) .- dlm_contrib[key]))
        idx_start = dlm_g[idx_ref_g].I[1]
        data[key] = f[idx_start+1: idx_start+tvals, 2]
    end

    if bc
        for key in ["re_bc", "rr_bc"]
            _, idx_ref_g = findmin( abs.(getindex.(dlm_g, 1) .- dlm_contrib[key]))
            idx_start = dlm_g[idx_ref_g].I[1]
            data[key] = f[idx_start+1: idx_start+tvals, 2]
        end
    end
    return data
end


function read_all_data(path::String, g::String; em::String="PA", bc::Bool=false)

    f = filter(x->occursin(em, x), readdir(path))
    # println(f)

    cnfg_idx = getindex.(split.(getindex.(split.(f, "n"), 2), "."), 1)
    cnfg_idx = sortperm(parse.(Int64, cnfg_idx))

    res = []
    for item in f[cnfg_idx]
        data = read_all_per_config(joinpath(path, item), g, bc=bc)
        push!(res, data)
    end
    return res
end