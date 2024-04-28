using LmaPredict, Flux, Statistics, Random, StatsBase, JLD2, Dates

include("utils/gridSearch_varyingTrainingData.jl")
include("utils/writeCSV.jl")

curr_date = now()

output_directory = "/home/users/kphth/lgeyer/LmaPredict/gridSearch/ee+ev/"

previous_results_matrix_name = output_directory * "results_2024-04-13.csv"

combination_matrix_name = output_directory * "models_$curr_date.csv"
results_matrix_name = output_directory * "results_$curr_date.csv"
path_config = "/home/users/kphth/lgeyer/Daten Simon/dat"

fname = readdir(path_config)[2:5001]
idx = sortperm( parse.(Int64, fname))
fname = fname[idx]
em_n = "VV"

###########################

println("Reading data...")
cnfgarr = Vector{LMAConfig}(undef, 0)
for f in fname
    push!(cnfgarr, get_LMAConfig(joinpath(path_config, f), "g5-g5", em=em_n, bc=false))
end
println("done.")

###########################

println("Processing data for CV...")

TSRC = "24"
TVALS = length(cnfgarr[1].data["rr"][TSRC]) - 1
NCNFG = length(cnfgarr)

n_times = 5
n_configs_k_folds = [(500, 5), (500, 4), (500, 2)]
isolated_testset_size = NCNFG - maximum([x[1] for x in n_configs_k_folds])
training_setups = length(n_configs_k_folds)

percentages_bc = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

rr_data_train = []
ee_data_train = []
re_data_train = []

rr_data_test = []
ee_data_test = []
re_data_test = []

rr_data_isolatedtest = Array{Float64}(undef, TVALS, isolated_testset_size)
ee_data_isolatedtest = Array{Float64}(undef, TVALS, isolated_testset_size)
re_data_isolatedtest = Array{Float64}(undef, TVALS, isolated_testset_size)

for (n_configs, k_fold) in n_configs_k_folds
    train_size = Int(n_configs / k_fold)
    test_size = n_configs - train_size

    Random.seed!(10)
    available_config_indices = sort(sample([x for x in 1:NCNFG], n_configs, replace=false))
    available_configs = getindex(getfield.(cnfgarr, :data), available_config_indices)

    if (n_configs, k_fold) == n_configs_k_folds[end]
        indexes_isolated_testset = setdiff([indx for indx in 1:NCNFG], deepcopy(available_config_indices))
        isolated_testset = getindex(getfield.(cnfgarr, :data), indexes_isolated_testset)
        
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

    rr_data_train_curr_setup = [[Array{Float64}(undef, TVALS, train_size) for j in 1:k_fold] for i in 1:n_times]
    ee_data_train_curr_setup = [[Array{Float64}(undef, TVALS, train_size) for j in 1:k_fold] for i in 1:n_times]
    re_data_train_curr_setup = [[Array{Float64}(undef, TVALS, train_size) for j in 1:k_fold] for i in 1:n_times]

    rr_data_test_curr_setup = [[Array{Float64}(undef, TVALS, test_size) for j in 1:k_fold] for i in 1:n_times]
    ee_data_test_curr_setup = [[Array{Float64}(undef, TVALS, test_size) for j in 1:k_fold] for i in 1:n_times]
    re_data_test_curr_setup = [[Array{Float64}(undef, TVALS, test_size) for j in 1:k_fold] for i in 1:n_times]

        
    for j in 1:n_times
        remaining_indexes_CV = deepcopy(available_config_indices)
        k_folds_indices = [Array{Int32}(undef, train_size) for k in 1:k_fold]

        for k in 1:k_fold
            Random.seed!(j+k)
            sampled_indices = sort!(sample(remaining_indexes_CV, train_size, replace=false))
            k_folds_indices[k] = sampled_indices
            remaining_indexes_CV = filter!(i -> i ∉ sampled_indices, remaining_indexes_CV)
        end
    
        for i in 1:k_fold
            indexes_train = k_folds_indices[i]
            indexes_test = setdiff(deepcopy(available_config_indices), indexes_train)
            
            training_set = getindex(getfield.(cnfgarr, :data), indexes_train)
            test_set = getindex(getfield.(cnfgarr, :data), indexes_test)
            
        
            for (k, dd) in enumerate(training_set)
                
                rr_data_train_curr_setup[j][i][:,k] = getindex(getindex(dd, "rr"), TSRC)[2:end]
        
                re_data_train_curr_setup[j][i][:,k] = getindex(getindex(dd, "re"), TSRC)[2:end]
            
                ee_all_TSRC = Matrix{Float64}(undef, TVALS, TVALS)
                for ee_TSRC in 0:TVALS-1
                    ee_all_TSRC[:,ee_TSRC+1] = getindex(getindex(dd, "ee"), "$ee_TSRC")[2:end]
                end
        
                ee_data_train_curr_setup[j][i][:,k] = mean(ee_all_TSRC, dims=2)
            end
            
            for (k, dd) in enumerate(test_set)
        
                rr_data_test_curr_setup[j][i][:,k] = getindex(getindex(dd, "rr"), TSRC)[2:end]
                
                re_data_test_curr_setup[j][i][:,k] = getindex(getindex(dd, "re"), TSRC)[2:end]
            
                ee_all_TSRC = Matrix{Float64}(undef, TVALS, TVALS)
                for ee_TSRC in 0:TVALS-1
                    ee_all_TSRC[:,ee_TSRC+1] = getindex(getindex(dd, "ee"), "$ee_TSRC")[2:end]
                end
        
                ee_data_test_curr_setup[j][i][:,k] = mean(ee_all_TSRC, dims=2)
            end
        end
    end

    push!(rr_data_train, rr_data_train_curr_setup)
    push!(ee_data_train, ee_data_train_curr_setup)
    push!(re_data_train, re_data_train_curr_setup)
    
    push!(rr_data_test, rr_data_test_curr_setup)
    push!(ee_data_test, ee_data_test_curr_setup)
    push!(re_data_test, re_data_test_curr_setup)
end
println("done.")

###########################

println("Standardizing data...")
input_length = 2*TVALS
output_length = TVALS

input_shape_train = []
output_shape_train = []

input_shape_test = []
output_shape_test = []

input_shape_validation = []

for (i, (_, k_fold)) in enumerate(n_configs_k_folds)
    push!(input_shape_train, [[vcat(ee_data_train[i][n][k], rr_data_train[i][n][k]) for k in 1:k_fold] for n in 1:n_times])
    push!(output_shape_train, [[re_data_train[i][n][k] for k in 1:k_fold] for n in 1:n_times])

    push!(input_shape_test, [[vcat(ee_data_test[i][n][k], rr_data_test[i][n][k]) for k in 1:k_fold] for n in 1:n_times])
    push!(output_shape_test, [[re_data_test[i][n][k] for k in 1:k_fold] for n in 1:n_times])

    push!(input_shape_validation, [[vcat(ee_data_isolatedtest, rr_data_isolatedtest) for k in 1:k_fold] for n in 1:n_times])
end

output_shape_validation = re_data_isolatedtest

input_data_train_standardized = []
input_data_test_standardized = []
input_data_validation_standardized = []

for (i, (_, k_fold)) in enumerate(n_configs_k_folds)
    input_data_train_standardized_curr_setup = [[similar(input_shape_train[i][1][1]) for k in 1:k_fold] for n in 1:n_times]
    input_data_test_standardized_curr_setup = [[similar(input_shape_test[i][1][1]) for k in 1:k_fold] for n in 1:n_times] 
    input_data_validation_standardized_curr_setup = [[similar(input_shape_validation[i][1][1]) for k in 1:k_fold] for n in 1:n_times] 
    
    for n in 1:n_times
        for k in 1:k_fold
            mean_input_train = mean(input_shape_train[i][n][k], dims=ndims(input_shape_train[i][n][k]))
            std_input_train = std(input_shape_train[i][n][k], dims=ndims(input_shape_train[i][n][k]))
            
            input_data_train_standardized_curr_setup[n][k] = (input_shape_train[i][n][k] .- mean_input_train) ./ std_input_train
            input_data_test_standardized_curr_setup[n][k] = (input_shape_test[i][n][k] .- mean_input_train) ./ std_input_train
            input_data_validation_standardized_curr_setup[n][k] = (input_shape_validation[i][n][k] .- mean_input_train) ./ std_input_train
        end
    end

    push!(input_data_train_standardized, input_data_train_standardized_curr_setup)
    push!(input_data_test_standardized, input_data_test_standardized_curr_setup)
    push!(input_data_validation_standardized, input_data_validation_standardized_curr_setup)
end

output_data_train_standardized = []

for (i, (_, k_fold)) in enumerate(n_configs_k_folds)
    output_data_train_standardized_curr_setup = [[similar(output_shape_train[i][1][1]) for k in 1:k_fold] for n in 1:n_times] 
    
    for n in 1:n_times
        for k in 1:k_fold
            mean_output_train = mean(output_shape_train[i][n][k], dims=ndims(output_shape_train[i][n][k]))
            std_output_train = std(output_shape_train[i][n][k], dims=ndims(output_shape_train[i][n][k]))
            
            output_data_train_standardized_curr_setup[n][k] = (output_shape_train[i][n][k] .- mean_output_train) ./ std_output_train
        end
    end
    
    push!(output_data_train_standardized, output_data_train_standardized_curr_setup)
end


println("done.")

###########################
println("Reading results from previous grid search")

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

models = vcat(models...) |> f64

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
batch_size = Int(minimum([x[1]/x[2] for x in n_configs_k_folds]))
batch_sizes = [batch_size]
variables = ["Model", "Optimizer", "Loss function", "Epoch", "Batch size"]

combinations_matrix = writeCombinationsMatrixPreviousBest(models, optimizers, loss_functions, epochs, batch_sizes, combination_matrix_name, previous_results_matrix_name)
n_combinations = size(combinations_matrix)[1]

###########################

gridSearch_varyingTrainingData(
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