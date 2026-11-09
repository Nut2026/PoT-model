import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    
    return idx

def chat(model, tokeniser, prompt, config):
    idx = torch.tensor([tokeniser.encode(prompt)], dtype=torch.long).to(config.device)

    out = generate(
        model,
        idx,
        max_new_tokens=200,
        block_size=config.block_size,
        temperature=0.8
    )

    print(tokeniser.decode(out[0].tolist()))
