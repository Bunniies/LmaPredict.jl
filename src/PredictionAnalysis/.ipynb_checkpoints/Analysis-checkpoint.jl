function plot_samples(predictions::Matrix, targets::Matrix, output_dir::String)
    if(size(predictions) != size(targets))
        error("Target and prediction size is not identical!")
    end

    num_predictions = size(predictions)[2]
    num_sample_configs = 9
    sample_configs = sort!(sample([i for i in 1:num_predictions], num_sample_configs, replace = false))

    l = @layout [a b c; d e f; g h i]

    p = plot(layout = l, size=(1200,1000), dpi=1000)

    scatter!(p, targets[:,sample_configs], label="Actual", legend=:top, markerstrokewidth = 0)
    scatter!(p, predictions[:,sample_configs], label="Prediction", legend=:top, markerstrokewidth = 0)

    rm(output_dir * "/samples.png", force=true)
    savefig(p, output_dir * "/samples.png")
end 

function plot_training_loss(losses::Vector{Any}, output_dir::String)
    loss_fraction = 0.2
    title = "Training loss in last 20% of training phase"

    plot(losses[end-Int(ceil(length(losses)*loss_fraction)):end], xlabel=("Training"), ylabel="Loss", legend=:false, xticks=false, yticks=false, title=title)

    rm(output_dir * "/training_loss.png", force=true)
    savefig(output_dir * "/training_loss.png")
end

function apply_bias_correction(predictions::Matrix, targets::Matrix, model, output_dir::String)
    num_predictions = size(predictions)[2]
    TVALS = size(predictions)[1]

    percentages_bc = [0.0, 0.01, 0.02, 0.05, 0.1, 0.12]
    n_configs_bc = Int.(4500 .* percentages_bc)

    table_header = [
        "Number of configs used for bc",
        "μ_diff (max)",
        "μ_diff (min)",
        "μ_diff (average)",
    ]

    table = Matrix{Float64}(undef, length(percentages_bc), length(table_header))

    l = @layout [a b c; d e f]

    means_target = Matrix{Float64}(undef, TVALS, length(percentages_bc))
    stds_target = Matrix{Float64}(undef, TVALS, length(percentages_bc))
    
    means_pred = Matrix{Float64}(undef, TVALS, length(percentages_bc))
    stds_pred = Matrix{Float64}(undef, TVALS, length(percentages_bc))
    
    means_diff = Matrix{Float64}(undef, TVALS, length(percentages_bc))
    
    for (i,n) in enumerate(n_configs_bc)
        Random.seed!(10)
        configs = sort!(sample([i for i in 1:num_predictions], n, replace = false))
        
        uncorr_target_configs = stack(deleteat!([targets[:,i] for i in 1:num_predictions], configs), dims=2)

        mean_target = mean.([uncorr_target_configs[i,:] for i in 1:TVALS])
        σ_mean_target = std.([uncorr_target_configs[i,:] for i in 1:TVALS]) ./ sqrt(size(uncorr_target_configs)[2] - 1)
        
        mean_predicted = mean.([predictions[i,:] for i in 1:TVALS])
        σ_predicted = std.([predictions[i,:] for i in 1:TVALS]) ./ sqrt(num_predictions - 1)

        if n > 0
            bias_correction = mean(hcat([[predictions[:,i] - targets[:,i] for i in configs][i] for i in 1:length(configs)]...), dims=2)
            σ_bc = std(hcat([[predictions[:,i] - targets[:,i] for i in configs][i] for i in 1:length(configs)]...), dims=2) ./ sqrt(n - 1)
        else
            bias_correction = zeros(TVALS)
            σ_bc = zeros(TVALS)
        end

        mean_predicted_bc = mean_predicted - bias_correction

        σ_pred_bc = σ_predicted + σ_bc

        mean_diff = mean_target .- mean_predicted_bc
        
        means_target[:,i] = mean_target
        stds_target[:,i] = σ_mean_target
        
        means_pred[:,i] = mean_predicted_bc
        stds_pred[:,i] = σ_pred_bc

        means_diff[:,i] = mean_diff

        max_mean_diff = maximum(abs.(mean_diff))
        min_mean_diff = minimum(abs.(mean_diff))
        average_mean_diff = mean(abs.(mean_diff))

        table[i,1] = n
        table[i,2] = max_mean_diff
        table[i,3] = min_mean_diff
        table[i,4] = average_mean_diff
    end

    open(output_dir * "/analysis.txt", "a") do file
        pretty_table(file, table, header=table_header)
        println(file)
        println(file, "Model: ", model)
    end

    p = scatter(
        means_diff[7:41,:],
        layout = l,
        size=(1400,1000),
        dpi = 1000,
        legend=:false,
        thickness_scaling = 1.1,
        title=reshape(["bc: $n" for n in n_configs_bc],1,length(n_configs_bc)),
        marker=:+,
        markersize = 2,
        markerstrokewidth = 0.3
    )
    savefig(p,output_dir * "/mean_diff.png")

    p = scatter(
        layout = l,
        size=(1400,1000),
        dpi = 1000,
        thickness_scaling = 1.1,
        title=reshape(["bc: $n" for n in n_configs_bc],1,length(n_configs_bc)))
    
    scatter!(p,
        means_target[7:41,:],
        label="actual",
        legend=:bottom,
        linecolor=:blue,
        marker=:xcross,
        markersize = 2,
        markerstrokewidth = 0.3
    )
    scatter!(p,
        means_pred[7:41,:],
        label="predicted",
        legend=:bottom,
        linecolor=:red,
        marker =:+,
        markersize = 2,
        markerstrokewidth = 0.3
    )
    savefig(p,output_dir * "/mean.png")

    p = scatter(
        layout = l,
        size=(1400,1000),
        dpi = 1000,
        thickness_scaling = 1.1,
        title=reshape(["bc: $n" for n in n_configs_bc],1,length(n_configs_bc)))
    
    scatter!(p,
        means_target[7:41,:],
        yerr=stds_target[7:41,:],
        label="actual",
        legend=:bottom,
        linecolor=:blue,
        marker=:xcross,
        markersize = 2,
        markerstrokewidth = 0.3
    )
    scatter!(p,
        means_pred[7:41,:],
        yerr=stds_pred[7:41,:],
        label="predicted",
        legend=:bottom,
        linecolor=:red,
        marker =:+,
        markersize = 2,
        markerstrokewidth = 0.3
    )
    savefig(p,output_dir * "/mean_errorbar.png")
end


function analyse_predictions(
                            predictions::Matrix,
                            targets::Matrix,
                            TSRC::String,
                            EIGVALS::Int64,
                            model,
                            optimizer,
                            loss_function,
                            loss_discription::String,
                            epochs::Int64,
                            batch_size::Int64,
                            losses::Vector{Any},
                            output_dir::String
                            )

    mkpath(output_dir)

    plot_samples(predictions, targets, output_dir)
    plot_training_loss(losses, output_dir)

    num_predictions = size(predictions)[2]

    minimum_training = minimum(losses)
    maximum_training = maximum(losses)
    average_training = mean(losses)
    
    out_of_sample_error = [loss_function(predictions[:,i], targets[:,i]) for i in 1:num_predictions]

    minimum_test = minimum(out_of_sample_error)
    maximum_test = maximum(out_of_sample_error)
    average_test = mean(out_of_sample_error)

    rm(output_dir * "/analysis.txt", force=true)
    open(output_dir * "/analysis.txt", "a") do file
        println(file, "Time source position: ", TSRC)
        println(file, "Number of used eigenvalues: ", EIGVALS, "\n")

        println(file, "Optimizer: ", optimizer)
        println(file, "Loss function: ", loss_discription)
        println(file, "Epochs: ", epochs)
        println(file, "Batch size: ", batch_size, "\n")
        
        println(file, "Minimum training error (∑⁴⁷ ", loss_discription, "): ", minimum_training)
        println(file, "Maximum training error (∑⁴⁷ ", loss_discription, "): ", maximum_training)
        println(file, "Average training error (∑⁴⁷ ", loss_discription, "): ", average_training, "\n")
        
        println(file, "Minimum test error (∑⁴⁷ ", loss_discription, "): ",  minimum_test)
        println(file, "Maximum test error (∑⁴⁷ ", loss_discription, "): ", maximum_test)
        println(file, "Average test error (∑⁴⁷ ", loss_discription, "): ", average_test, "\n")
    end

    apply_bias_correction(predictions, targets, model, output_dir)
end