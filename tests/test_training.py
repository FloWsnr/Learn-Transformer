import pytest
import torch
from tokenizers import Tokenizer

from learn_former.run.train import Trainer
from learn_former.data.tokenizer import CustomTokenizer


def test_train_on_batch(model, tokenizer: Tokenizer):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Make a dummy trainer, trainloader and testloader not needed
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=None,
        test_loader=None,
        tokenizer_de=tokenizer,
        tokenizer_en=tokenizer,
        device="cpu",
    )

    # Create a dummy batch of random characters
    batch_size = 16
    seq_len = 10

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    indices = torch.randint(0, len(alphabet), (batch_size, seq_len))
    text = ["".join([alphabet[i] for i in indices[j]]) for j in range(batch_size)]

    batch = {
        "tr": text,
        "en": text,
    }

    batch_loss = trainer._train_on_batch(batch)
    # Test if loss is not nan
    assert not torch.isnan(torch.tensor(batch_loss)).any()


def test_train_on_batch_cuda(model, tokenizer: Tokenizer):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Make a dummy trainer, trainloader and testloader not needed
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=None,
        test_loader=None,
        tokenizer_de=tokenizer,
        tokenizer_en=tokenizer,
        device="cuda",
    )

    # Create a dummy batch of random characters
    batch_size = 16
    seq_len = 10

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    indices = torch.randint(0, len(alphabet), (batch_size, seq_len))
    text = ["".join([alphabet[i] for i in indices[j]]) for j in range(batch_size)]

    batch = {
        "tr": text,
        "en": text,
    }

    batch_loss = trainer._train_on_batch(batch)
    # Test if loss is not nan
    assert not torch.isnan(torch.tensor(batch_loss)).any()


def test_evaluate_on_batch(model, tokenizer: Tokenizer):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Make a dummy trainer, trainloader and testloader not needed
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=None,
        test_loader=None,
        tokenizer_de=tokenizer,
        tokenizer_en=tokenizer,
        device="cpu",
    )

    # Create a dummy batch of random characters
    batch_size = 16
    seq_len = 10

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    indices = torch.randint(0, len(alphabet), (batch_size, seq_len))
    text = ["".join([alphabet[i] for i in indices[j]]) for j in range(batch_size)]

    batch = {
        "tr": text,
        "en": text,
    }

    batch_loss = trainer._evaluate_on_batch(batch)
    # Test if loss is not nan
    assert not torch.isnan(torch.tensor(batch_loss)).any()
