using LmaPredict, PyPlot
using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Statistics
using Random
using MLUtils

#=========== PLOT VARIABLES SETTINGS =============#
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] =  true
rcParams["mathtext.fontset"]  = "cm"
rcParams["font.size"] =13
rcParams["axes.labelsize"] =24
rcParams["axes.titlesize"] = 18
plt.rc("text", usetex=false) # true if latex installation available 

path_plot="/Users/alessandroconigli/Desktop/postdoc-mainz/projects/MLtoLMA"
path_config = "/Users/alessandroconigli/Lattice/data/HVP/LMA/A654/"

fname = readdir(path_config)
idx_cut = findall(x->x<=2000, parse.(Int64, fname))
fname = fname[idx_cut]
idx = sortperm( parse.(Int64, fname))
fname = fname[idx]

cnfgarr = Vector{LMAConfig}(undef, 0)
for f in fname
    push!(cnfgarr, get_LMAConfig(joinpath(path_config, f), "g5-g5", em="PA", bc=false))
end

NCNFG = length(cnfgarr)
TVALS = length(cnfgarr[1].data["rr"]["0"]) 

## input manipulation for mlp

TSRC="12"

rr_data = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
ee_data = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
re_data = Array{Float64}(undef, Int64(NCNFG/2), TVALS)

rr_data_test = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
ee_data_test = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
re_data_test = Array{Float64}(undef, Int64(NCNFG/2), TVALS)


for (k, dd) in enumerate(getfield.(cnfgarr, :data)[1:Int64(NCNFG/2)])
    rr_data[k,:] = getindex(getindex(dd, "rr"), TSRC)[1:end]
    ee_data[k,:] = getindex(getindex(dd, "ee"), TSRC)[1:end]
    re_data[k,:] = getindex(getindex(dd, "re"), TSRC)[1:end]
end
for (k, dd) in enumerate(getfield.(cnfgarr, :data)[Int64(NCNFG/2):NCNFG-1])
    rr_data_test[k,:] = getindex(getindex(dd, "rr"), TSRC)[1:end]
    ee_data_test[k,:] = getindex(getindex(dd, "ee"), TSRC)[1:end]
    re_data_test[k,:] = getindex(getindex(dd, "re"), TSRC)[1:end]
end


## define the model

function loss(y, ypred)
    lss = Flux.mse(ypred, y)
    return lss
end

dcgan_init(shape...) = randn(Float32, shape...) * 0.02f0

function MLP_net()
    net = Chain(
        Dense(1=>32, leakyrelu),
        #Dense(10 => 20),
        Dense(32 => 64, leakyrelu),
        Dense(64 => 32, leakyrelu),
#        Dense(64 => 64, elu),
        Dense(32 =>1)
    )
    return net
end

function train_mlp!(net, x, y, hp)

    opt = Flux.setup(Adam(hp.lr), net)
    
    x = reshape(x, 1, 1, 1, 1000)
    y = reshape(y, 1, 1, 1, 1000)

    xbatch = [x[:,:,:, k] for k in partition(1:size(x,4), hp.batch_size)]
    ybatch = [y[:,:,:, k] for k in partition(1:size(y,4), hp.batch_size)]

    store_lss = []
    train_steps = 0 
    for ep in 1:hp.epochs
        for k in eachindex(xbatch[1:end])

            lss, grads = Flux.withgradient(net) do net
                loss(ybatch[k], net(xbatch[k]))
            end
            update!(opt, net, grads[1])
            push!(store_lss, lss)

            if train_steps % hp.verbose_freq == 0 
                @info "Epoch $(ep): train step: $(train_steps), loss: $(lss)"
            end
            train_steps +=1
        end
    end
    plot(store_lss, label="loss")
    legend()
    display(gcf())
    close("all")
    return nothing
end

aa = train_mlp!(net_tot[1], hcat(rr_data[:,1]...), hcat(re_data[:,1]...), hp )

## training
Base.@kwdef struct HyperParams
    batch_size::Int = 200
    epochs::Int = 1000
    verbose_freq::Int = 100
    lr::Float32 = 0.001
end

hp = HyperParams()
net_tot = [MLP_net() for t in 1:TVALS]

for t in 1:TVALS
    train_mlp!(net_tot[t], normalise(rr_data[:,t]), normalise(re_data[:,t]), hp )
end

## predictions

minimum(vcat(net_tot[1](reshape(rr_data_test[:,1], 1,1,1, 1000))...))
re_data_test[:,1]

##
fig = figure(figsize=(8,8))
for k in 1:9
    TT = k+10
    ii = 330+k
    # xxtest = rr_data_test[:,TT]
    # ypred = vcat(net_tot[TT](reshape(xxtest, 1, 1, 1, 1000 ))...)
    # ytrue = re_data_test[:,TT]

    xxtest = normalise(rr_data_test[:,TT])
    ypred = normalise(vcat(net_tot[TT](reshape(xxtest, 1, 1, 1, 1000 ))...))
    ytrue = normalise(re_data_test[:,TT])
    

    subplot(ii)
    scatter(xxtest, ytrue, label="true")
    scatter(xxtest, ypred, label="pred")
    legend()
end
tight_layout()
display(gcf())
close("all")
