from torch import nn as nn

from bert_modules.embedding import BERTEmbedding
from bert_modules.transformer import TransformerBlock

class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        max_len = args.maxlen
        n_layers = args.bert_num_blocks
        heads = args.bert_num_heads
        vocab_size = args.id_vocab
        hidden = args.id_dimension
        self.hidden = hidden
        dropout = args.bert_dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

    def init_weights(self):
        pass
