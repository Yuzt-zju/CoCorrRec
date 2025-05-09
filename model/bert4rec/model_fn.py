import torch.nn
import torch.nn as nn

from module import encoder, common, initializer
from util import consts
from . import config
from ..model_meta import MetaType, model
import logging
logger = logging.getLogger(__name__)
from .bert_modules.embedding import BERTEmbedding
from .bert_modules.transformer import TransformerBlock

@model("bert4rec", MetaType.ModelBuilder)
class BERT(nn.Module):
    def __init__(self, model_conf):
        super().__init__()
        n_layers = model_conf.mlp_layers
        heads = model_conf.nhead
        vocab_size = model_conf.id_vocab

        self.hidden = model_conf.id_dimension
        dropout = 0.1

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=self.hidden,max_len=10, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, heads, self.hidden * 4, dropout) for _ in range(n_layers)])
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self,features,train=True):
        target_embed = self.embedding.token(features[consts.FIELD_TARGET_ID])
        
        item_seq = features[consts.FIELD_CLK_SEQUENCE]
        batch_size = int(item_seq.shape[0])
        mask = torch.not_equal(item_seq, 0)
        item_seq_len = torch.maximum(torch.sum(mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
        item_seq_len = item_seq_len.to(torch.long)

        x = item_seq
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
            
        seq_output = x[range(batch_size), item_seq_len, :]
        return torch.sum(seq_output * target_embed, dim=1, keepdim=True)
