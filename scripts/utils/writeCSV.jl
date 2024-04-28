using CSV, Tables

function writeCombinationsMatrix(models, optimiers, loss_functions, epochs, batch_sizes, combinations_matrix_name)
    n_combinations = length(models) * length(loss_functions) * length(epochs) * length(batch_sizes) * length(optimizers)
    combinations_matrix = Matrix{Any}(undef, n_combinations, length(variables))

    i = 1
    for model in models
        for optimizer in optimizers
            for loss_function in loss_functions
                for epoch in epochs
                    for batch_size in batch_sizes
                        combinations_matrix[i,:] = [model, optimizer, loss_function, epoch, batch_size]
                        i +=1
                    end
                end
            end
        end
    end
    println("Writing .csv for combinations to '$combination_matrix_name'...")
    CSV.write(combination_matrix_name, Tables.table(combinations_matrix), header=variables, delim=";")
    println("done.")

    return combinations_matrix
end

function writeCombinationsMatrixPreviousBest(models, optimiers, loss_functions, epochs, batch_sizes, combination_matrix_name, previous_results_matrix_name)
    n_combinations = length(models) * length(loss_functions) * length(epochs) * length(batch_sizes) * length(optimizers)
    combinations_matrix = Matrix{Any}(undef, n_combinations, length(variables))

    i = 1
    for model in models
        for optimizer in optimizers
            for loss_function in loss_functions
                for epoch in epochs
                    for batch_size in batch_sizes
                        combinations_matrix[i,:] = [model, optimizer, loss_function, epoch, batch_size]
                        i +=1
                    end
                end
            end
        end
    end

    scores_mat_previous = CSV.File(open(previous_results_matrix_name); header=1, types=Float64) |> CSV.Tables.matrix

    ##### Evaluating previous scores #####
    combinations_matrix_prev_best = [combinations_matrix[i,:] for i in  Int.(scores_mat_previous[(scores_mat_previous[:,3] .> -2),:][:,1])]
    ##### / Evaluating previous scores #####

    combinations_matrix = permutedims(hcat(combinations_matrix_prev_best...))

    println("Writing .csv for combinations to '$combination_matrix_name'...")
    CSV.write(combination_matrix_name, Tables.table(combinations_matrix), header=variables, delim=";")
    println("done.")

    return combinations_matrix
end