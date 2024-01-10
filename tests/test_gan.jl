using LmaPredict, PyPlot
using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using Statistics
using Random

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
idx_cut = findall(x->x<=1000, parse.(Int64, fname))
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

## input manipulation for gan 

TSRC_rr = collect(keys(getfield(cnfgarr[1], :data)["rr"] ))
TSRC_re = collect(keys(getfield(cnfgarr[1], :data)["re"] ))

rr_re_images = Vector{Array{Float64}}(undef, TVALS)

for t in 1:TVALS    

    rr_re_images[t] = Array{Float64}(undef, 2, 1, length(TSRC_rr), NCNFG)
    for n in 1:NCNFG
        for (k,ts) in enumerate(TSRC_rr)
            rr_aux = getfield(cnfgarr[n], :data)["rr"][ts][t]
            re_aux = getfield(cnfgarr[n], :data)["re"][ts][t]
            
            rr_re_images[t][:,:,k, n] = [rr_aux, re_aux] 
        end
    end
end

rr_re_images |> size # vector for each TVALS
rr_re_images[1] |> size 

##
function discriminator()
    dscr = Chain(
        Dense(2 => 32, leakyrelu),
        Dropout(0.3),
        Dense(32 => 16, leakyrelu),
        Dropout(0.3),
        Dense(16 => 1, sigmoid)
    )
    return dscr
end

dscr = discriminator()
dscr(rr_re_images[1])

function generator(latent_dim::Int)
    gen = Chain(
        Dense(latent_dim => 16, leakyrelu),
        Dense(16 => 8, leakyrelu),
        Dense(8 => 2)
    )
    return gen
end

function generate_noise(hparams)
    noise = randn(hparams.latent_dim, hparams.batch_size)
    return noise
end

generate_noise(HyperParams())

function generate_fake_sample(gen, hparams)
    noise = generate_noise(hparams)
    return gen(noise)
end
gen = generator(HyperParams().latent_dim)

generate_fake_sample(gen, HyperParams())

function discriminator_loss(real_sample, fake_sample)
    real_loss = logitbinarycrossentropy(real_sample, 1)
    fake_loss = logitbinarycrossentropy(fake_sample, 0)
    return real_loss + fake_loss
end

generator_loss(fake_sample) = logitbinarycrossentropy(fake_sample, 1)

function train_discriminator!(gen, dscr, real_sample, opt_discr, hparams)
    fake_sample = generate_fake_sample(gen, hparams) 
    loss, grad = Flux.withgradient(dscr) do dscr
        discriminator_loss(dscr(real_sample), dscr(fake_sample))
    end
    update!(opt_discr, dscr, grad[1])
    return loss
end

function train_generator!(gen, dscr, opt_gen, hparams)
    noise = generate_noise(hparams)
    loss, grad = Flux.withgradient(gen) do gen
        generator_loss(dscr(gen(noise)))
    end
    update!(opt_gen, gen, grad[1])
    return loss
end

function train(data::Array{Float64,4}; kws...)
    hparams = HyperParams()

    data_batches = [data[:,:,:,k] for k in partition(1:size(data,4), hparams.batch_size)]

    dscr = discriminator()
    gen  = generator(hparams.latent_dim) 

    # optimizers
    opt_dscr = Flux.setup(Adam(hparams.lr_dscr), dscr)
    opt_gen = Flux.setup(Adam(hparams.lr_gen), gen)

    store_loss_dscr = []
    store_loss_gen  = []

    # training
    train_steps = 0
    for ep in 1:hparams.epochs
        for d_batch in data_batches
            loss_dscr = train_discriminator!(gen, dscr, d_batch, opt_dscr, hparams)
            loss_gen  = train_generator!(gen, dscr, opt_gen, hparams) 

            push!(store_loss_dscr, loss_dscr)
            push!(store_loss_gen, loss_gen)
            
            if train_steps % hparams.verbose_freq == 0 
                @info "Epoch $(ep): train step: $(train_steps), dscr loss: $(loss_dscr), gen loss: $(loss_gen)"
            end
            train_steps +=1
        end
    end
    plot(store_loss_dscr, label="dscr loss")
    plot(store_loss_gen, label="gen loss")
    legend()
    display(gcf())
    close("all")
    
    return gen, dscr
end

## Test Training
Base.@kwdef struct HyperParams
    batch_size::Int = 100
    latent_dim::Int = 100
    epochs::Int = 1000
    verbose_freq::Int = 100
    output_x::Int = 6
    output_y::Int = 6
    lr_dscr::Float32 = 0.0002
    lr_gen::Float32 = 0.0002
end

gen, dscr = train(rr_re_images[1][:,:,:,:])

##
# test dscr with real data
all(isapprox.(dscr(rr_re_images[1][:,:,:,:]), 1, atol=0.01)) # dscr correctly classify input data as real data

fake_sample = generate_fake_sample(gen, HyperParams(batch_size=1000))

dscr(fake_sample)
#
scatter(rr_re_images[1][1, :, :, :], rr_re_images[1][2, :, :, :], label="true")
scatter(fake_sample[1,:], fake_sample[2,:], label="prediction")
legend()
display(gcf())
close("all")