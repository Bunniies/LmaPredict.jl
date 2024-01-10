function corr_obs_rr(data::Array{Float64,3}, id::String)

    tvals, ncnfg, nsrcs = size(data)
    println("tvals: ", tvals)
    println("ncnfg: ", ncnfg)
    println("nsrcs: ", nsrcs)

    data_ave_src = dropdims(mean(data, dims=3),dims=3)

    println(size(data_ave_src))

    obs = Vector{uwreal}(undef, tvals)
    
    for t in 1:tvals
        obs[t] = uwreal(data_ave_src[t,:], id)
    end
    
    return obs
end