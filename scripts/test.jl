using Flux, CUDA, BenchmarkTools

#model_GPU = Chain(Dense(2*47 => 100, tanh),Dropout(0.8),Dense(100 => 20, tanh),Dropout(0.8),Dense(20 => 47, identity)) |> gpu
#model_CPU = Chain(Dense(2*47 => 100, tanh),Dropout(0.8),Dense(100 => 20, tanh),Dropout(0.8),Dense(20 => 47, identity))

model_CPU = Chain(Dense(94 => 1000, tanh), Dropout(0.8), Dense(1000 => 47))
model_GPU = Chain(Dense(94 => 1000, tanh), Dropout(0.8), Dense(1000 => 47)) |> gpu

optimizer_GPU = Flux.Adam(0.001) |> gpu
optimizer_CPU = Flux.Adam(0.001)

loss_function_CPU = function loss_mse(flux_model, x, y)
    batch_size = size(x)[2]
    ŷ = flux_model(x)
    
    return Flux.mse(ŷ, y, agg=sum)
end 
loss_function_GPU = loss_function_CPU |> gpu

batch_size = 100
epochs = 150
training_size = batch_size

batch_size_GPU = batch_size |> gpu
batch_size_CPU = batch_size

epochs_GPU = epochs |> gpu
epochs_CPU = epochs

input_GPU = rand(2*47,training_size) |> gpu
input_CPU = rand(2*47,training_size) |> f32

target_GPU = rand(47,training_size) |> gpu
target_CPU = rand(47,training_size) |> f32

loader_GPU = Flux.DataLoader((input_GPU, target_GPU), batchsize=batch_size_GPU, shuffle=true)
loader_CPU = Flux.DataLoader((input_CPU, target_CPU), batchsize=batch_size_CPU, shuffle=true)

optim_GPU = Flux.setup(optimizer_GPU, model_GPU)
optim_CPU = Flux.setup(optimizer_CPU, model_CPU)

function train_GPU(model, loader, epochs, optim, loss_function)
    for e in 1:epochs
        for (x,y) in loader
            grads = Flux.gradient(model) do m
                loss_function(model, x, y)
            end
            
            Flux.update!(optim, model, grads[1])
        end
    end
end

function train_CPU(model, loader, epochs, optim, loss_function)
    for e in 1:epochs
        for (x,y) in loader
            grads = Flux.gradient(model) do m
                loss_function(model, x, y)
            end
            
            Flux.update!(optim, model, grads[1])
        end
    end
end

@btime train_GPU(model_GPU, loader_GPU, epochs_GPU, optim_GPU, loss_function_GPU)
@btime train_CPU(model_CPU, loader_CPU, epochs_CPU, optim_CPU, loss_function_CPU)