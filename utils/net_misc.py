import transformers
from torch import nn

def disentangle_bart_embedding_weights(model):
    config = model.config
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).cuda()
    lm_head.weight.data.copy_(model.lm_head.weight.data)
    model.lm_head = lm_head