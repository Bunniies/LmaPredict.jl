include("BC.jl")

function gridSearch_varyingTrainingData(
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

    scores_table_beforeBC_header = [
    "Index",
    "Min(max. R-Score)",
    "<max. R-Score>",
    "Std(max. R-Score)",
    "<(epoch of max R-Score>",
    "Std(epoch of max R-Score)",
    "Min(max. R-Score (Val. set))",
    "<max. R-Score (Val. set)>",
    "Std(max. R-Score (Val. set))",
    "<epoch of max R-Score (Val. set)>",
    "Std(epoch of max R-Score (Val. set))",
]

    scores_table_afterBC_header = vcat(
        [
        "Index",
        "<epoch of min <Chi^2>>",
        "Std(<epoch of min Chi^2>)",
        vcat(hcat([hcat(["<Chi^2($p%)>"]...) for p in Int.(percentages_bc * 100)]...)...),
        vcat(hcat([hcat(["<Chi_configs^2($p%)>"]...) for p in Int.(percentages_bc * 100)]...)...)
        ]
        ...)
    
    scores_mat_before_BC = Matrix{Float64}(undef, n_combinations, length(scores_table_beforeBC_header))
    scores_mat_after_BC = Matrix{Float64}(undef, n_combinations, length(scores_table_afterBC_header))

    println("Performing grid-search...")
    for (j, (n_configs, k_fold)) in enumerate(n_configs_k_folds)
        println("$n_configs" * " configs available and doing CV with " * "$k_fold" * "-fold:")
        
        train_size = Int(n_configs / k_fold)
        test_size = n_configs - train_size
    
        outputn = n_configs
        outputk = k_fold
    
        results_matrix_name_beforeBC = output_directory * "results_beforeBC_" * "$n_configs" * "_" * "$k_fold" * "_$curr_date.csv"
        results_matrix_name_afterBC = output_directory * "results_afterBC_" * "$n_configs" * "_" * "$k_fold" * "_$curr_date.csv"
        
        for i in 1:n_combinations
            model = combinations_matrix[i,1]
            optimizer = combinations_matrix[i,2]
            loss_function = combinations_matrix[i,3]
            epochs = combinations_matrix[i,4]
        
            initial_params = deepcopy(Flux.params(model))
        
            max_R = Array{Float32}(undef, k_fold*n_times)
            epoch_of_max_R = Array{Float32}(undef, k_fold*n_times)
            max_R_val = Array{Float32}(undef, k_fold*n_times)
            epoch_of_max_R_val = Array{Float32}(undef, k_fold*n_times)
            
            epoch_of_min_χ² = Array{Float32}(undef, k_fold*n_times)
            min_χ²_values_model = Array{Float32}(undef, k_fold*n_times, length(percentages_bc))
            min_χ²_values_computedconfigs = Array{Float32}(undef, k_fold*n_times, length(percentages_bc))
            
            index = 1
            for n in 1:n_times
                for k in 1:k_fold
                    mean_train = repeat(mean.([output_shape_train[j][n][k][i,:] for i in 1:output_length]), 1, test_size)
                    mean_train_val = repeat(mean.([output_shape_train[j][n][k][i,:] for i in 1:output_length]), 1, isolated_testset_size)
                    
                    mean_output_train = mean(output_shape_train[j][n][k], dims=ndims(output_shape_train[j][n][k]))
                    std_output_train = std(output_shape_train[j][n][k], dims=ndims(output_shape_train[j][n][k]))
                    
                    Random.seed!(20)
                    loader = Flux.DataLoader(
                                        (input_data_train_standardized[j][n][k], output_data_train_standardized[j][n][k]),
                                        batchsize=batch_size,
                                        shuffle=true)
                
                    optim = Flux.setup(optimizer, model)
                
                    R_scores = zeros(epochs)
                    R_scores_val = zeros(epochs)
                    χ²_scores_model = zeros(epochs, length(percentages_bc))
                    χ²_scores_computedconfigs = zeros(epochs, length(percentages_bc))
                    
                    function training()
                        for e in 1:epochs
                            for (x,y) in loader
                                grads = gradient(m -> loss_function(m, x, y), model)
                                Flux.update!(optim, model, grads[1])
                            end
                            out_of_sample_predictions = (model(input_data_test_standardized[j][n][k]) .* std_output_train) .+ mean_output_train
                            validation_set_predictions = (model(input_data_validation_standardized[j][n][k]) .* std_output_train) .+ mean_output_train
                            
                            R_scores_val[e] = 1 - (Flux.mse(validation_set_predictions, output_shape_validation, agg=sum) / Flux.mse(mean_train_val, output_shape_validation, agg=sum))
                            R_scores[e] = 1 - (Flux.mse(out_of_sample_predictions, output_shape_test[j][n][k], agg=sum) / Flux.mse(mean_train, output_shape_test[j][n][k], agg=sum))
    
                            (χ²_scores_model[e,:], χ²_scores_computedconfigs[e,:]) = applyBC(out_of_sample_predictions, model, j, n, k, train_size, test_size)
                        end
                    end
                
                    training()
        
                    max_R_val[index] = maximum(R_scores_val)
                    epoch_of_max_R_val[index] = argmax(R_scores_val)
                    max_R[index] = maximum(R_scores)
                    epoch_of_max_R[index] = argmax(R_scores)
    
                    avrg_χ²_scores_model = mean(χ²_scores_model, dims=2)
    
                    epoch_of_min_χ²[index] = argmin(avrg_χ²_scores_model)[1]
                    min_χ²_values_model[index,:] = χ²_scores_model[argmin(avrg_χ²_scores_model)[1], :]
                    min_χ²_values_computedconfigs[index,:] = χ²_scores_computedconfigs[argmin(avrg_χ²_scores_model)[1], :]
                    
                    Flux.loadparams!(model,initial_params)
                    
                    index += 1
                end
            end
            
            scores_mat_before_BC[i,:] = [
                i,
                minimum(max_R),
                mean(max_R),
                std(max_R),
                mean(epoch_of_max_R),
                std(epoch_of_max_R),
                minimum(max_R_val),
                mean(max_R_val),
                std(max_R_val),
                mean(epoch_of_max_R_val),
                std(epoch_of_max_R_val)
            ]
    
            scores_mat_after_BC[i,:] = vcat([
                i,
                mean(epoch_of_min_χ²),
                std(epoch_of_min_χ²),
                vcat(mean(min_χ²_values_model, dims=1)...),
                vcat(mean(min_χ²_values_computedconfigs, dims=1)...)
            ]...)    
        
            println("$i/$n_combinations done...")
            CSV.write(results_matrix_name_beforeBC, Tables.table(scores_mat_before_BC), header=scores_table_beforeBC_header, delim=";")
            CSV.write(results_matrix_name_afterBC, Tables.table(scores_mat_after_BC), header=scores_table_afterBC_header, delim=";")
        end
    end
    println("done.")
end