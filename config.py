# Model hyperparameters (PoT, AKA Parrot)
block_size = 64
batch_size = 32
embedding_dim = 128
num_heads = 4
num_layers = 4
learning_rate = 3e-4
max_iters = 5000

# Training
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"