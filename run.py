from model.pot import TinyPoT
from tokeniser.char_tokeniser import CharTokeniser
import config
from generate.generate import chat
import torch

text = open("data/raw.txt").read()
tokeniser = CharTokeniser(text)

model = TinyPoT(tokeniser.vocab_size, config).to(config.device)

model.load_state_dict(torch.load("model.pt", map_location=config.device))

chat(model, tokeniser, "hello", config)