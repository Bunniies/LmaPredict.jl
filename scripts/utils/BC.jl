function applyBC(out_of_sample_predictions, j, n, k, train_size, test_size)
    
    χ²_model = Array{Float32}(undef, length(percentages_bc))
    χ²_configs = Array{Float32}(undef, length(percentages_bc))
    for (p, percentage) in enumerate(percentages_bc)

        if percentage > 0 
            Δc = 1 / percentage
            configs = Int.(ceil.([x for x in 1:Δc:test_size]))
        else
            configs = []
        end
        n_configs = length(configs)
        
        uncorr_target_configs = stack(deleteat!([output_shape_test[j][n][k][:,i] for i in 1:test_size],configs), dims=2)
            
        mean_target = mean(uncorr_target_configs, dims=2)
        σ_mean_target = std(uncorr_target_configs, dims=2) ./ sqrt(test_size - n_configs - 1)
    
        mean_target_train = mean(output_shape_train[j][n][k], dims=2)
        σ_mean_target_train = std(output_shape_train[j][n][k], dims=2) ./ sqrt(train_size - 1)
                
        mean_predicted = mean(out_of_sample_predictions, dims=2) 
        σ_predicted = std(out_of_sample_predictions, dims=2) ./ sqrt(test_size - 1)
            
        if percentage > 0
            bias_correction = mean(hcat([[out_of_sample_predictions[:,i] - output_shape_test[j][n][k][:,i] for i in configs][i] for i in 1:length(configs)]...), dims=2)
            σ_bc = std(hcat([[out_of_sample_predictions[:,i] - output_shape_test[j][n][k][:,i] for i in configs][i] for i in 1:length(configs)]...), dims=2) ./ sqrt(n_configs - 1)
                
            mean_target_train = mean(hcat(output_shape_train[j][n][k], hcat([output_shape_test[j][n][k][:,i] for i in configs]...)), dims=2)
            σ_mean_target_train = std(hcat(output_shape_train[j][n][k], hcat([output_shape_test[j][n][k][:,i] for i in configs]...)), dims=2) ./ sqrt(train_size + n_configs - 1)
        else
            bias_correction = zeros(output_length) 
            σ_bc = zeros(output_length)
        end

        mean_predicted = mean_predicted - bias_correction
        σ_pred_bc = σ_predicted + σ_bc
            
        χ²_model[p] = sum(((mean_predicted - mean_target) ./ sqrt.(σ_pred_bc.^2 + σ_mean_target.^2)).^2)
        χ²_configs[p] = sum(((mean_target_train - mean_target) ./ sqrt.(σ_mean_target_train.^2 + σ_mean_target.^2)).^2)
    end

    return (χ²_model, χ²_configs)
end