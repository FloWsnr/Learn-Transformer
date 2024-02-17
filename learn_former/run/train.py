"""Train the model."""

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import CrossEntropyLoss

from learn_former.model.transformer import Transformer
from learn_former.data.dataset import get_dataloaders
from learn_former.data.tokenizer import CustomTokenizer


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    criterion: CrossEntropyLoss,
    tokenizer: CustomTokenizer,
):
    for epoch in range(10):
        print(f"Epoch {epoch}")

        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            # input: [batch, tokens]
            # target: [batch, tokens]
            input = batch["de"]
            target = batch["en"]

            # Shift target to the right and input to the left
            # With this, the model will predict the next token given the previous ones
            input = input[:, 1:]
            target = target[:, :-1]

            # Encode input and target, both are padded to the same length
            # Mask show which tokens are padding tokens
            input, input_padding_mask = tokenizer.encode_batch(input)
            target, target_padding_mask = tokenizer.encode_batch(target)

            # Forward pass
            output = model(input, target, input_padding_mask, target_padding_mask)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        print(f"Epoch loss: {epoch_loss}")

        # Validation
        with torch.no_grad():
            for batch in val_loader:
                input, target = batch
                output = model(input, target)
                loss = criterion(output, target)

            print(f"Validation loss: {loss.item()}")

        scheduler.step()


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: CrossEntropyLoss,
    tokenizer: CustomTokenizer,
):
    with torch.no_grad():
        for batch in test_loader:
            input, target = batch

            input = tokenizer.encode_batch(input)
            target = tokenizer.encode_batch(target)

            output = model(input, target)
            loss = criterion(output, target)

            print(f"Test loss: {loss.item()}")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        root="learn_former\data\datasets",
        dataset_name="wmt16",
        batch_size=32,
        num_workers=4,
    )

    model = Transformer()
    optimizer = torch.optim.Adam(model, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    tokenizer = CustomTokenizer(
        tokenizer_path="learn_former\data\tokenizers\tokenizer.json",
        dataset_dir="learn_former\data\datasets",
        dataset_name="wmt16",
    )

    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        tokenizer,
    )
    print("Finished training")
