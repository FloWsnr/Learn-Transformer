import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:

    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max)

    return x_exp / x_exp.sum(dim=-1, keepdim=True)


def cel(logits: torch.Tensor, targets: torch.Tensor):
    """
    logits: B, SEQ, Tokens
    targets: B, Seq
    """
    b, seq, toks = logits.shape
    logits = logits.view(b * seq, toks)
    targets = targets.view(b * seq)

    log_probs = torch.log_softmax(logits, dim=-1)
    index = torch.arange(0, b * seq)

    return -log_probs[
        index, targets
    ].mean()  # for each prediction (log_prob row fetched by index), get the log prob of the target token (col)


def training_loop():
    model = torch.nn.Identity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    model.train()

    dataset = torch.utils.data.Dataset()
    dataloader = torch.utils.data.DataLoader(dataset)

    for x, y in dataloader:
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = cel(pred, y)
        loss.backward()
        optimizer.step()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, len: int, max_seq_len: int, max_tok: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_tok = max_tok
        self.data = torch.randint(0, self.max_tok, (len, self.max_seq_len))

    def __getitem__(self, index) -> torch.Tensor:
        cut = torch.randint(1, self.max_seq_len, (1,))
        return self.data[index, :cut]

    def __len__(self):
        return self.data.shape[0]


def llm_training_loop():
    model = torch.nn.Linear(32, 20)
    dataloader = torch.utils.data.DataLoader(
        TextDataset(len=10, max_seq_len=16, max_tok=20)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for input in dataloader:
        opt.zero_grad()

        input = input.to("cpu")  # B, seq, toks

        # shift
        target = input[:, 1:]
        input = input[:, :-1]

        logits = model(input)
        loss: torch.Tensor = loss_fn(logits.reshape(-1, 20), target.reshape(-1, 20))
        loss.backward()
        opt.step()


if __name__ == "__main__":
    x = torch.rand(32, 16, 8)
    targets = torch.randint(0, 7, size=(32, 16))
    x = cel(x, targets)
    print(x)
