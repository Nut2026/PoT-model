import torch
import torch.nn as nn
from model.pot import TinyPoT
from tokeniser.char_tokeniser import CharTokeniser
import config

text = open("data/raw.txt").read()
tokeniser = CharTokeniser(text)

data = torch.tensor(tokeniser.encode(text))

model = TinyPoT(tokeniser.vocab_size, config).to(config.device)
optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def get_batch():
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)

for step in range(config.max_iters):
    xb, yb = get_batch()

    logits = model(xb)
    B, T, C = logits.shape

    loss = nn.functional.cross_entropy(
        logits.view(B*T, C),
        yb.view(B*T)
    )

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if step % 10 == 0:
        print("step:", step, "loss:", loss.item())
    
    if step % 250 == 0:
        torch.save(model.state_dict(), "model.pt")
        print(f"Model of {step}-step training saved to model.pt")

torch.save(model.state_dict(), "model.pt")
print("Model of 5000-step training saved to model.pt")