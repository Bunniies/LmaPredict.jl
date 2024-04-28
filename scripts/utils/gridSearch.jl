function gridSearch(
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

    scores_table_header = [
    "Index",
    "Min(max. R-Score)",
    "Avrg(max. R-Score)",
    "Std(max. R-Score)",
    "Avrg(epoch of max R-Score)",
    "Std(epoch of max R-Score)",
    "Min(max. R-Score (Val. set))",
    "Avrg(max. R-Score (Val. set))",
    "Std(max. R-Score (Val. set))",
    "Avrg(epoch of max R-Score (Val. set))",
    "Std(epoch of max R-Score (Val. set))",
    ]
    
    scores_mat = Matrix{Float64}(undef, n_combinations, length(scores_table_header))

    println("Performing grid-search...")
    for i in 2063:n_combinations
        model = combinations_matrix[i,1]
        optimizer = combinations_matrix[i,2]
        loss_function = combinations_matrix[i,3]
        epochs = combinations_matrix[i,4]
        batch_size = combinations_matrix[i,5]

        initial_params = deepcopy(Flux.params(model))

        max_R = Array{Float32}(undef, k_fold*n_times)
        epoch_of_max_R = Array{Int32}(undef, k_fold*n_times)
        max_R_val = Array{Float32}(undef, k_fold*n_times)
        epoch_of_max_R_val = Array{Int32}(undef, k_fold*n_times)
        
        index = 1
        for n in 1:n_times
            for k in 1:k_fold
                Flux.loadparams!(model,initial_params)
                
                mean_train = repeat(mean.([output_shape_train[n][k][i,:] for i in 1:output_length]), 1, test_size)
                mean_train_val = repeat(mean.([output_shape_train[n][k][i,:] for i in 1:output_length]), 1, isolated_testset_size)
                mean_output_train = mean(output_shape_train[n][k], dims=ndims(output_shape_train[n][k])) 
                std_output_train = std(output_shape_train[n][k], dims=ndims(output_shape_train[n][k])) 
                
                Random.seed!(20)
                loader = Flux.DataLoader(
                                    (input_data_train_standardized[n][k], output_data_train_standardized[n][k]),
                                    batchsize=batch_size,
                                    shuffle=true) 
            
                optim = Flux.setup(optimizer, model)
            
                R_scores = zeros(epochs)
                R_scores_val = zeros(epochs)
                function training()
                    for e in 1:epochs
                        for (x,y) in loader
                            grads = gradient(m -> loss_function(m, x, y), model)
                            Flux.update!(optim, model, grads[1])
                        end
                        out_of_sample_predictions = (model(input_data_test_standardized[n][k]) .* std_output_train) .+ mean_output_train 
                        validation_set_predictions = (model(input_data_validation_standardized[n][k]) .* std_output_train) .+ mean_output_train
                        
                        R_scores_val[e] = 1 - (Flux.mse(validation_set_predictions, output_shape_validation, agg=sum) / Flux.mse(mean_train_val, output_shape_validation, agg=sum))
                        R_scores[e] = 1 - (Flux.mse(out_of_sample_predictions, output_shape_test[n][k], agg=sum) / Flux.mse(mean_train, output_shape_test[n][k], agg=sum))
                    end
                end
            
                training()

                max_R_val[index] = maximum(R_scores_val)
                epoch_of_max_R_val[index] = argmax(R_scores_val)
                max_R[index] = maximum(R_scores)
                epoch_of_max_R[index] = argmax(R_scores)
                
                index += 1
            end
        end
        
        scores_mat[i,:] = [
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

        println("$i/$n_combinations done...")
        CSV.write(results_matrix_name, Tables.table(scores_mat), header=scores_table_header, delim=";")
    end

    println("done.")
end