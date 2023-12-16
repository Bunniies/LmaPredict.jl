using LmaPredict, PyPlot, Flux

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
idx_cut = findall(x->x<2000 && x != 265, parse.(Int64, fname))
fname = fname[idx_cut]
idx = sortperm( parse.(Int64, fname))
fname = fname[idx]

cnfgarr = Vector{LMAConfig}(undef, 0)
for f in fname
    push!(cnfgarr, get_LMAConfig(joinpath(path_config, f), "g5-g5", em="PA", bc=false))
end

TSRC="12"
NCNFG = length(cnfgarr)
TVALS = length(cnfgarr[1].data["rr"][TSRC]) -1

rr_data = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
ee_data = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
re_data = Array{Float64}(undef, Int64(NCNFG/2), TVALS)

rr_data_test = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
ee_data_test = Array{Float64}(undef, Int64(NCNFG/2), TVALS)
re_data_test = Array{Float64}(undef, Int64(NCNFG/2), TVALS)


for (k, dd) in enumerate(getfield.(cnfgarr, :data)[1:Int64(NCNFG/2)])
    rr_data[k,:] = getindex(getindex(dd, "rr"), TSRC)[2:end]
    ee_data[k,:] = getindex(getindex(dd, "ee"), TSRC)[2:end]
    re_data[k,:] = getindex(getindex(dd, "re"), TSRC)[2:end]
end
for (k, dd) in enumerate(getfield.(cnfgarr, :data)[Int64(NCNFG/2):NCNFG-1])
    rr_data_test[k,:] = getindex(getindex(dd, "rr"), TSRC)[2:end]
    ee_data_test[k,:] = getindex(getindex(dd, "ee"), TSRC)[2:end]
    re_data_test[k,:] = getindex(getindex(dd, "re"), TSRC)[2:end]
end

##
TSRC_rr = unique(length.((getindex.(getfield.(cnfgarr, :data), "rr") )))
TSRC_ee = unique(length.((getindex.(getfield.(cnfgarr, :data), "ee") )))
TSRC_re = unique(length.((getindex.(getfield.(cnfgarr, :data), "re") )))

rr_data
ee_data
re_data

## inspect data
fig = figure(figsize=(10,10))


for k in 1:9
    ii = 330 + k
    tt = 10 + 2*k
    # println(ii)
    subplot(ii)
    scatter(rr_data[:,tt], re_data[:,tt], color="blue", label=L"$t/a=$"*" $(tt)")
    # scatter(ee_data[:,tt], re_data[:,tt], color="orange", label=L"$t/a=$"*" $(tt)" )
    ax=gca()
    setp(ax.get_xticklabels(),visible=false) # Disable x tick labels
    setp(ax.get_yticklabels(),visible=false) # Disable y tick labels
    legend()
end
PyPlot.tight_layout()
display(gcf())
savefig(joinpath(path_plot, "rr_vs_re.pdf"))
# savefig(joinpath(path_plot, "ee_vs_re.pdf"))
close("all")
##

scatter(rr_data[:,10], re_data[:,10])
#scatter(ee_data[:,40], re_data[:,40] )
xlabel(L"$C^{ee} + C^{rr}$")
ylabel(L"$C^{re}$")
display(gcf())
PyPlot.close()

##

# xx =[ collect(1:size(rr_data,1)) collect(1:size(rr_data,2))]
xx =hcat(fill(collect(1:size(rr_data,2)), size(rr_data,1))...)
scatter(xx, transpose(rr_data), label="r-r")
scatter(xx .+ 0.05, transpose(re_data), label="r-e")
scatter(xx .+ 0.1, transpose(ee_data), label="e-e")
# scatter(xx, ee, label="e-e")
# scatter(xx, re, label="r-e")
# scatter(xx, tot, label="tot")
#ylim(-0.02, 0.05)
#xlim(10, 40)
xlabel(L"$x_0/a$")
legend()
display(gcf())
PyPlot.close("all")

## Flux models
function loss(y, ypred)
    lss = Flux.mse(ypred, y)
    return lss
end


function train(net, eta, x, y)
    opt = Flux.setup(Adam(eta), net)
    grads = Flux.gradient(m -> Flux.mse(m(x), y), net)
    Flux.update!(opt, net, grads[1])
end

function train1(net, eta, x, y)
    opt = Flux.setup(Adam(eta), net)
    
    lss, grads = Flux.withgradient(net) do net
        loss(y, net(x))
    end
    Flux.update!(opt, net, grads[1])
end

models = []
for t in 1:TVALS
    push!(models, Dense(1=>1))
end

## Training Attempt
EPOCHS = 10000
ETA = 0.001


loss(models[1], hcat(rr_data[:,1]...), hcat(re_data[:,1]...))


@time begin 
    for t in 1:TVALS
        for ep in 1:EPOCHS
            # train(models[t], ETA, loss, hcat(rr_data[:,t]...), hcat(re_data[:,t]...))
            train1(models[t], ETA, hcat(rr_data[:,t]...), hcat(re_data[:,t]...))

            # for k in NCNFG
                # train(models[t], ETA, loss, hcat(rr_data[k,t]...), hcat(re_data[k,t]...))
            # end
        end
    end
end
##
fig = figure(figsize=(9,9))

for k in 1:9
    TT = 10+4*k
    ii = 330 + k
    ypred = models[TT](hcat(rr_data_test[:,TT]  ...))
    ytrue = hcat(re_data_test[:,TT]...)

    subplot(ii)
    ax=gca()
    scatter(rr_data_test[:,TT] , vcat(ytrue...), label="true")
    scatter(rr_data_test[:,TT] , vcat(ypred...), label="prediction", alpha=0.5)
    setp(ax.get_xticklabels(),visible=false) # Disable x tick labels
    setp(ax.get_yticklabels(),visible=false) # Disable y tick labels
   legend() 
end
PyPlot.tight_layout()
display(gcf())
savefig(joinpath(path_plot, "predict_rr_vs_re.pdf"))
# savefig(joinpath(path_plot, "ee_vs_re.pdf"))
close("all")

##

x = hcat(rr_data[:,30]...)
y = hcat(abs.(re_data[:,30])...)
scatter(x,y)
display(gcf())
close("all")

model = Dense(1,1)

function loss(model, x, y)
    ypred = model(x)
    lss = Flux.mse(ypred, y)
    return lss
end

function train(model, eta)
    dLdm, _, _ = gradient(loss, model, x, y)
    
    model.weight .-= eta* dLdm.weight
    model.bias .-=  eta* dLdm.bias
end

loss(model, x, y)

lss = []

for i in 1:20000
    train(model, 0.0001)
    push!(lss, loss(model, x, y))
end

plot(collect(1:length(lss)), lss)
display(gcf())
close("all")
loss(model, x, y)

scatter(vcat(x...), vcat(model(x)...), lw=1, color="red")
scatter(x, y)
plot()
display(gcf())
close("all")
