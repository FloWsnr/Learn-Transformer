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


class Trainer:
    def __init__(
        self,
        model: Transformer,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: CrossEntropyLoss,
        train_loader: DataLoader,
        test_loader: DataLoader,
        tokenizer_de: CustomTokenizer,
        tokenizer_en: CustomTokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tokenizer_de = tokenizer_de
        self.tokenizer_en = tokenizer_en

        self.device = device
        self.model = self.model.to(self.device)

    def train(self, epochs: int = 10):
        print(f"Starting training for {epochs} epochs")
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            epoch_train_loss = 0
            epoch_val_loss = 0
            num_batches = 0
            for i, batch in enumerate(train_loader):
                print(f"Processing batch {i}")
                batch_loss = self._train_on_batch(batch)
                epoch_train_loss += batch_loss
                num_batches += 1

            epoch_train_loss /= num_batches
            print(f"Epoch training loss: {epoch_train_loss}")

            # Validation
            with torch.no_grad():
                for batch in val_loader:
                    loss = self._evaluate_on_batch(batch)
                    epoch_val_loss += loss

                epoch_val_loss /= len(val_loader)
                print(f"Validation loss: {loss.item()}")

            scheduler.step()

    def _evaluate_on_batch(self, batch: torch.Tensor):
        input, target = batch
        output = model(input, target)
        loss = criterion(output, target)
        return loss.item()

    def _train_on_batch(self, batch: torch.Tensor) -> float:

        #################################
        # Tokenization ##################
        #################################
        input = batch["de"]
        target = batch["en"]

        # Encode input and target
        # Need two different tokenizers because the input and target languages are different
        input = self.tokenizer_de.encode_batch(input)
        target = self.tokenizer_en.encode_batch(target)

        #################################
        # Convert to tensor #############
        #################################
        input = torch.tensor(input)
        target = torch.tensor(target)

        # move to device
        input = input.to(self.device)
        target = target.to(self.device)

        # input: [batch, tokens]
        # target: [batch, tokens]
        # Last target token is not used as input to the model
        # Forward pass
        output: torch.Tensor = self.model(input, target[:, :-1])

        # Calculate loss
        # output: [batch, tokens, vocab_size]
        # Reshape target to [batch, tokens * vocab_size]
        output = output.view(-1, output.shape[-1])

        # Remove first token from target to match output
        target = target[:, 1:]
        # Reshape target to [batch * tokens]
        target = target.contiguous().view(-1)

        loss = self.criterion(output, target)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        batch_loss = loss.item()
        print(f"\tBatch loss: {batch_loss}")
        return batch_loss


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
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = CustomTokenizer.from_pretrained_tokenizer(tokenizer_path)

    else:
        print(f"Training tokenizer and saving to {tokenizer_path}")
        tokenizer = CustomTokenizer.from_dataset(
            dataset_path=dataset_dir,
            dataset_name=dataset_name,
            split="train",
            language=language,
        )
        tokenizer.save_tokenizer(tokenizer_path)

    return tokenizer


def get_model(
    d_model: int,
    d_ff: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    vocab_size_de: int,
    vocab_size_en: int,
    padding_id: int,
    device: str = "cuda",
) -> Transformer:

    model = Transformer(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        vocab_size_de=vocab_size_de,
        vocab_size_en=vocab_size_en,
        padding_id=padding_id,
    )

    # Move to device
    model = model.to(device)
    return model


if __name__ == "__main__":
    from config import PACKAGE_PATH

    dataset_dir = PACKAGE_PATH / "learn_former\data\datasets"
    dataset_name = "wmt16"
    tokenizer_de_path = PACKAGE_PATH / r"learn_former\data\tokenizers\de_tokenizer.json"
    tokenizer_en_path = PACKAGE_PATH / r"learn_former\data\tokenizers\en_tokenizer.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ########################################
    ###### Get dataset #####################
    ########################################
    print("Loading dataset")
    dataset = get_dataset(dataset_dir, dataset_name)

    ########################################
    ###### Get dataloaders #################
    ########################################
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=16,
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

    padding_id_de = tokenizer_de.tokenizer.token_to_id("[PAD]")
    padding_id_en = tokenizer_en.tokenizer.token_to_id("[PAD]")
    if padding_id_de != padding_id_en:
        raise ValueError("Padding ids are different")

    vocabulary_size_de = tokenizer_de.tokenizer.get_vocab_size()
    vocabulary_size_en = tokenizer_en.tokenizer.get_vocab_size()

    ########################################
    ###### Train model #####################
    ########################################

    model = get_model(
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        vocab_size_de=vocabulary_size_de,
        vocab_size_en=vocabulary_size_en,
        padding_id=padding_id_de,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer_de=tokenizer_de,
        tokenizer_en=tokenizer_en,
        device=device,
    )
    trainer.train(epochs=10)
    print("Finished training")
