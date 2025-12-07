# Fine-tuning the model to classify spam
import pdb
import time
import torch
import torch.nn as nn


from ch06.utils import calc_loss_batch, calc_accuracy_loader, evaluate_model






def train_classifier_simple(train_loader, val_loader, model, optimizer, device, num_epochs, eval_freq, eval_iter):

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Initialize lists to track losses and examples seen
    example_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        # Sets model to training mode
        model.train()
        for input_batch, target_batch in enumerate(train_loader):
            # Resets loss gradients from the previous batch iteration
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # Calculates loss gradients
            loss.backward()

            # Updates model weights using loss gradients
            optimizer.step()

            # New: tracks examples instead of tokens
            example_seen += input_batch.shape[0]
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Step {global_step:06d}):"
                      f"Train loss: {train_loss:.3f}"
                      f"Val loss: {val_loss:.3f}")
                
        # Calculates accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)