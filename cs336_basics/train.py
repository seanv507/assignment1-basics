import wandb
import torch
from cs336_basics.batching import get_batch
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint

wandb.login()

# specify model
# specify hyper parameters
# learn from scratch or continue
# Ability to configure and control the various model and optimizer hyperparameters.
# • Memory-eﬀicient loading of training and validation large datasets with np.memmap.
# • Serializing checkpoints to a user-provided path.
# • Periodically logging training and validation performance (e.g., to console and/or an external
# service like Weights and Biases).a

def train():
    n_epochs = 100
    n_batches = 
    batch_size = 100
    context_length = 128
    logging_every = 100
    checkpoint_file = 
    checkpoint_every = 1000
    model
    optim
    
    with wandb.init(project=project, config=config) as run:
    for i_epoch in range(n_epochs):
        for i_batch in range(n_batches):
            X, y = get_batch(data, batch_size, context_length)

            pred = model(X)
            loss = cross_entropy_loss(y, pred)
            loss.backward()
        if i_epoch % logging_every ==0:
            run.log()
        if i_epoch % checkpoint_every ==0:
            save_checkpoint(model, optim, i_epoch, checkpoint_file)
    
    
