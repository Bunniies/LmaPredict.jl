using LmaPredict, Flux, Statistics, Random, StatsBase, JLD2, Dates

include("utils/writeCSV.jl")
include("utils/gridSearch.jl")

curr_date = now()

output_directory = "/home/users/kphth/lgeyer/LmaPredict/gridSearch/ee+ev/"

combination_matrix_name = output_directory * "models_$curr_date.csv"
results_matrix_name = output_directory * "results_$curr_date.csv"
path_config = "/home/users/kphth/lgeyer/Daten Simon/dat"

fname = readdir(path_config)[2:5001]
idx = sortperm( parse.(Int64, fname))
fname = fname[idx]
em_n = "VV"

println("Reading data...")
cnfgarr = Vector{LMAConfig}(undef, 0)
for f in fname
    push!(cnfgarr, get_LMAConfig(joinpath(path_config, f), "g5-g5", em=em_n, bc=false))
end
println("done.")

println("Processing data for CV...")
n_times = 5
k_fold = 8
NCNFG = length(cnfgarr)
train_size = 500
isolated_testset_size = 1000
test_size = NCNFG - isolated_testset_size - train_size

TSRC = "24"
TVALS = length(cnfgarr[1].data["rr"][TSRC]) - 1

rr_data_train = [[Array{Float32}(undef, TVALS, train_size) for j in 1:k_fold] for i in 1:n_times]
ee_data_train = [[Array{Float32}(undef, TVALS, train_size) for j in 1:k_fold] for i in 1:n_times]
re_data_train = [[Array{Float32}(undef, TVALS, train_size) for j in 1:k_fold] for i in 1:n_times]

rr_data_test = [[Array{Float32}(undef, TVALS, test_size) for j in 1:k_fold] for i in 1:n_times]
ee_data_test = [[Array{Float32}(undef, TVALS, test_size) for j in 1:k_fold] for i in 1:n_times]
re_data_test = [[Array{Float32}(undef, TVALS, test_size) for j in 1:k_fold] for i in 1:n_times]

rr_data_isolatedtest = Array{Float32}(undef, TVALS, isolated_testset_size)
ee_data_isolatedtest = Array{Float32}(undef, TVALS, isolated_testset_size)
re_data_isolatedtest = Array{Float32}(undef, TVALS, isolated_testset_size)

for j in 1:n_times
    remaining_indexes_CV = [index for index in 1:NCNFG-isolated_testset_size]
    k_folds_indices = [Array{Int32}(undef, train_size) for k in 1:k_fold]

    for k in 1:k_fold
        Random.seed!(j+k)
        sampled_indices = sort!(sample(remaining_indexes_CV, train_size, replace=false))
        k_folds_indices[k] = sampled_indices
        remaining_indexes_CV = filter!(i -> i ∉ sampled_indices, remaining_indexes_CV)
    end
    
    for i in 1:k_fold
        indexes_train = k_folds_indices[i]
        indexes_test = deleteat!([indx for indx in 1:NCNFG-isolated_testset_size], indexes_train)
        
        training_set = getindex(getfield.(cnfgarr, :data), indexes_train)
        test_set = getindex(getfield.(cnfgarr, :data), indexes_test)
        isolated_testset = getfield.(cnfgarr, :data)[NCNFG-isolated_testset_size+1:end]
    
        for (k, dd) in enumerate(training_set)
            
            rr_data_train[j][i][:,k] = getindex(getindex(dd, "rr"), TSRC)[2:end]
    
            re_data_train[j][i][:,k] = getindex(getindex(dd, "re"), TSRC)[2:end]
        
            ee_all_TSRC = Matrix{Float64}(undef, TVALS, TVALS)
            for ee_TSRC in 0:TVALS-1
                ee_all_TSRC[:,ee_TSRC+1] = getindex(getindex(dd, "ee"), "$ee_TSRC")[2:end]
            end
    
            ee_data_train[j][i][:,k] = mean(ee_all_TSRC, dims=2)
        end
        
        for (k, dd) in enumerate(test_set)
    
            rr_data_test[j][i][:,k] = getindex(getindex(dd, "rr"), TSRC)[2:end]
            
            re_data_test[j][i][:,k] = getindex(getindex(dd, "re"), TSRC)[2:end]
        
            ee_all_TSRC = Matrix{Float64}(undef, TVALS, TVALS)
            for ee_TSRC in 0:TVALS-1
                ee_all_TSRC[:,ee_TSRC+1] = getindex(getindex(dd, "ee"), "$ee_TSRC")[2:end]
            end
    
            ee_data_test[j][i][:,k] = mean(ee_all_TSRC, dims=2)
        end

        for (k, dd) in enumerate(isolated_testset)
    
            rr_data_isolatedtest[:,k] = getindex(getindex(dd, "rr"), TSRC)[2:end]
            
            re_data_isolatedtest[:,k] = getindex(getindex(dd, "re"), TSRC)[2:end]
        
            ee_all_TSRC = Matrix{Float64}(undef, TVALS, TVALS)
            for ee_TSRC in 0:TVALS-1
                ee_all_TSRC[:,ee_TSRC+1] = getindex(getindex(dd, "ee"), "$ee_TSRC")[2:end]
            end
    
            ee_data_isolatedtest[:,k] = mean(ee_all_TSRC, dims=2)
        end
    end
end
println("done.")

###########################

println("Standardizing data...")
input_length = 2*TVALS
output_length = TVALS

input_shape_train = [[vcat(ee_data_train[n][k], rr_data_train[n][k]) for k in 1:k_fold] for n in 1:n_times]
output_shape_train = [[re_data_train[n][k] for k in 1:k_fold] for n in 1:n_times]

input_shape_test = [[vcat(ee_data_test[n][k], rr_data_test[n][k]) for k in 1:k_fold] for n in 1:n_times]
output_shape_test = [[re_data_test[n][k] for k in 1:k_fold] for n in 1:n_times]

input_shape_validation = [[vcat(ee_data_isolatedtest, rr_data_isolatedtest) for k in 1:k_fold] for n in 1:n_times]
output_shape_validation = re_data_isolatedtest

input_data_train_standardized = [[similar(input_shape_train[1][1]) for k in 1:k_fold] for n in 1:n_times]
input_data_test_standardized = [[similar(input_shape_test[1][1]) for k in 1:k_fold] for n  in 1:n_times] 
input_data_validation_standardized = [[similar(input_shape_validation[1][1]) for k in 1:k_fold] for n  in 1:n_times] 

for n in 1:n_times
    for k in 1:k_fold
        mean_input_train = mean(input_shape_train[n][k], dims=ndims(input_shape_train[n][k]))
        std_input_train = std(input_shape_train[n][k], dims=ndims(input_shape_train[n][k]))
        
        input_data_train_standardized[n][k] = (input_shape_train[n][k] .- mean_input_train) ./ std_input_train
        input_data_test_standardized[n][k] = (input_shape_test[n][k] .- mean_input_train) ./ std_input_train
        input_data_validation_standardized[n][k] = (input_shape_validation[n][k] .- mean_input_train) ./ std_input_train
    end
end

output_data_train_standardized = [[similar(output_shape_train[1][1]) for k in 1:k_fold] for n in 1:n_times] 

for n in 1:n_times
    for k in 1:k_fold
        mean_output_train = mean(output_shape_train[n][k], dims=ndims(output_shape_train[n][k]))
        std_output_train = std(output_shape_train[n][k], dims=ndims(output_shape_train[n][k]))
        
        output_data_train_standardized[n][k] = (output_shape_train[n][k] .- mean_output_train) ./ std_output_train
    end
end


println("done.")

###########################
println("Defining data for grid-search...")

activation_functions = [NNlib.tanh, NNlib.celu, NNlib.elu, NNlib.tanhshrink]

models = []

for activation_function in activation_functions
    push!(models,
        [
    Chain(
    Dense(input_length => output_length, identity),
    ),
    Chain(
    Dense(input_length => 1, activation_function),
    Dense(1 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 50, activation_function),
    Dense(50 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 100, activation_function),
    Dense(100 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 400, activation_function),
    Dropout(0.8),
    Dense(400 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 600, activation_function),
    Dropout(0.8),
    Dense(600 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 800, activation_function),
    Dropout(0.8),
    Dense(800 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 1000, activation_function),
    Dropout(0.8),
    Dense(1000 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 1200, activation_function),
    Dropout(0.8),
    Dense(1200 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 50, activation_function),
    Dense(50 => 1, activation_function),
    Dense(1 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 50, activation_function),
    Dense(50 => 50, activation_function),
    Dense(50 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 100, activation_function),
    Dense(100 => 50, activation_function),
    Dense(50 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 100, activation_function),
    Dense(100 => 100, activation_function),
    Dense(100 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 400, activation_function),
    Dropout(0.8),
    Dense(400 => 200, activation_function),
    Dropout(0.8),
    Dense(200 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 600, activation_function),
    Dropout(0.8),
    Dense(600 => 400, activation_function),
    Dropout(0.8),
    Dense(400 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 800, activation_function),
    Dropout(0.8),
    Dense(800 => 600, activation_function),
    Dropout(0.8),
    Dense(600 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 1000, activation_function),
    Dropout(0.8),
    Dense(1000 => 800, activation_function),
    Dropout(0.8),
    Dense(800 => output_length, identity)
    ),
    Chain(
    Dense(input_length => 50, activation_function),
    Dense(50 => 50, activation_function),
    Dense(50 => 50, activation_function),
    Dense(50 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 100, activation_function),
    Dense(100 => 100, activation_function),
    Dense(100 => 100, activation_function),
    Dense(100 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 400, activation_function),
    Dropout(0.8),
    Dense(400 => 400, activation_function),
    Dropout(0.8),
    Dense(400 => 400, activation_function),
    Dropout(0.8),
    Dense(400 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 600, activation_function),
    Dropout(0.8),
    Dense(600 => 600, activation_function),
    Dropout(0.8),
    Dense(600 => 600, activation_function),
    Dropout(0.8),
    Dense(600 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 800, activation_function),
    Dropout(0.8),
    Dense(800 => 800, activation_function),
    Dropout(0.8),
    Dense(800 => 800, activation_function),
    Dropout(0.8),
    Dense(800 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 1000, activation_function),
    Dropout(0.8),
    Dense(1000 => 1000, activation_function),
    Dropout(0.8),
    Dense(1000 => 1000, activation_function),
    Dropout(0.8),
    Dense(1000 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 400, activation_function),
    Dropout(0.8),
    Dense(400 => 200, activation_function),
    Dropout(0.8),
    Dense(200 => 200, activation_function),
    Dropout(0.8),
    Dense(200 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 600, activation_function),
    Dropout(0.8),
    Dense(600 => 400, activation_function),
    Dropout(0.8),
    Dense(400 => 400, activation_function),
    Dropout(0.8),
    Dense(400 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 800, activation_function),
    Dropout(0.8),
    Dense(800 => 600, activation_function),
    Dropout(0.8),
    Dense(600 => 600, activation_function),
    Dropout(0.8),
    Dense(600 => output_length, identity),
    ),
    Chain(
    Dense(input_length => 1000, activation_function),
    Dropout(0.8),
    Dense(1000 => 800, activation_function),
    Dropout(0.8),
    Dense(800 => 800, activation_function),
    Dropout(0.8),
    Dense(800 => output_length, identity),
    ),
     ])
end

models = vcat(models...)

learning_rates = [1e-3]


optimizers = []
push!(optimizers,[Flux.AdaGrad(), Flux.AdaDelta(), Flux.AMSGrad(), Flux.NAdam()])
for learning_rate in learning_rates
    push!(optimizers,
        [
            Flux.Adam(learning_rate),
            Flux.RAdam(learning_rate),
            Flux.AdaMax(learning_rate),
            Flux.AdamW(learning_rate),
            Flux.OAdam(learning_rate)
            ])
end
optimizers = vcat(optimizers...)

function loss_mse(flux_model, x, y)
    batch_size = size(x)[2]
    ŷ = flux_model(x)
    
    return Flux.mse(ŷ, y, agg=sum)
end

function loss_mae(flux_model, x, y)
    batch_size = size(x)[2]
    ŷ = flux_model(x)
    
    return Flux.mae(ŷ, y, agg=sum)
end

function loss_mse_sum(flux_model, x, y)
    batch_size = size(x)[2]
    ŷ = flux_model(x)
    
    return Flux.mse(ŷ, y, agg=sum) + sum(abs.((sum.([ŷ[:,i] for i in 1:batch_size]) - sum.([y[:,i] for i in 1:batch_size]))))
end

loss_functions = [loss_mse, loss_mae, loss_mse_sum]

epochs = [150]

batch_sizes = [64]

variables = ["Model", "Optimizer", "Loss function", "Epoch", "Batch size"]

n_combinations = length(models) * length(loss_functions) * length(epochs) * length(batch_sizes) * length(optimizers)

println("done.")

###########################

n_combinations = length(models) * length(loss_functions) * length(epochs) * length(batch_sizes) * length(optimizers)

###########################

combinations_matrix = writeCombinationsMatrix(models, optimizers, loss_functions, epochs, batch_sizes, combination_matrix_name)

###########################

gridSearch(
    
    n_combinations,
    combinations_matrix,
    output_shape_train,
    output_shape_test,
    input_data_train_standardized,
    output_data_train_standardized,
    input_data_test_standardized,
    input_data_validation_standardized,
    output_shape_validation,
    results_matrix_name
    )