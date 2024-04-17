function corr_obs(data::Array{Float64,3}, id::String; idm::Union{Nothing, Vector{Int64}}=nothing, nms::Union{Int64,Nothing}=nothing)


    tvals, ncnfg, nsrcs = size(data)
    idm = isnothing(idm) ? collect(1:ncnfg) : idm
    nms = isnothing(nms) ? ncnfg : nms

    println("tvals: ", tvals)
    println("ncnfg: ", ncnfg)
    println("nsrcs: ", nsrcs)

    data_ave_src = dropdims(mean(data, dims=3),dims=3)

    println(size(data_ave_src))

    obs = Vector{uwreal}(undef, tvals)
    
    for t in 1:tvals
        obs[t] = uwreal(data_ave_src[t,:], id, idm, nms)
    end
    
    return obs
end