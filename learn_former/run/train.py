"""Train the model."""

import torch

from learn_former.model.transformer import Transformer
from learn_former.data.dataset import get_dataloaders


def train(model, train_loader, val_loader, optimizer, scheduler, criterion):
    for epoch in range(10):
        print(f"Epoch {epoch}")

        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            input, target = batch
            # input: [batch, tokens]
            # target: [batch, tokens]

            # Forward pass
            output = model(input, target)
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


def test(model, test_loader, criterion):
    with torch.no_grad():
        for batch in test_loader:
            input, target = batch
            output = model(input, target)
            loss = criterion(output, target)

            print(f"Test loss: {loss.item()}")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        root="data", splits=("train", "val", "test"), language_pair=("en", "de")
    )

    model = Transformer()
    optimizer = torch.optim.Adam(model, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion)
    print("Finished training")
