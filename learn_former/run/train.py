"""Train the model."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import CrossEntropyLoss

from learn_former.model.transformer import Transformer
from learn_former.data.dataset import get_dataloaders, get_dataset
from learn_former.data.tokenizer import CustomTokenizer


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    criterion: CrossEntropyLoss,
    tokenizer_de: CustomTokenizer,
    tokenizer_en: CustomTokenizer,
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

            # Encode input and target
            # Need two different tokenizers because the input and target languages are different
            input = tokenizer_de.encode_batch(input)
            target = tokenizer_en.encode_batch(target)

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


def get_tokenizer(
    tokenizer_path: Path, dataset_dir: str, dataset_name: str, language: str
) -> CustomTokenizer:

    if tokenizer_path.exists():
        tokenizer = CustomTokenizer.from_pretrained_tokenizer(tokenizer_path)

    else:
        tokenizer = CustomTokenizer.from_dataset(
            dataset_path=dataset_dir,
            dataset_name=dataset_name,
            split="train",
            language=language,
        )
        tokenizer.save_tokenizer(tokenizer_path)

    return tokenizer


if __name__ == "__main__":
    from config import PACKAGE_PATH

    dataset_dir = PACKAGE_PATH / "learn_former\data\datasets"
    dataset_name = "wmt16"
    tokenizer_de_path = PACKAGE_PATH / r"learn_former\data\tokenizers\de_tokenizer.json"
    tokenizer_en_path = PACKAGE_PATH / r"learn_former\data\tokenizers\en_tokenizer.json"

    ########################################
    ###### Get dataset #####################
    ########################################
    dataset = get_dataset(dataset_dir, dataset_name)

    ########################################
    ###### Get dataloaders #################
    ########################################
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=32,
        num_workers=4,
    )

    ########################################
    ###### Get tokenizers ##################
    ########################################

    tokenizer_de = get_tokenizer(
        tokenizer_de_path, "learn_former\data\datasets", "wmt16", "de"
    )
    tokenizer_en = get_tokenizer(
        tokenizer_en_path, "learn_former\data\datasets", "wmt16", "en"
    )

    vocabulary_size_de = tokenizer_de.tokenizer.get_vocab_size()
    vocabulary_size_en = tokenizer_en.tokenizer.get_vocab_size()

    ########################################
    ###### Train model #####################
    ########################################

    model = Transformer(
        d_model=512,
        d_ff=2048,
        num_head=8,
        num_layers=6,
        dropout=0.1,
        vocab_size_de=vocabulary_size_de,
        vocab_size_en=vocabulary_size_en,
        padding_id=3,
    )
    optimizer = torch.optim.Adam(model, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        tokenizer_de,
        tokenizer_en,
    )
    print("Finished training")
