from .ttt import *
from torch import nn
import torch.nn.init as init
import torch
from . import common, initializer
import math
class CoCorrRec(nn.Module):
    def __init__(self,model_conf,config):
        super(CoCorrRec, self).__init__()
        self.ttt = TTTModelIR(config)

        self._target_trans = common.StackedDense(
            model_conf['id_dimension'], [model_conf['id_dimension']] * 2, [torch.nn.Tanh, None]
        )

    def forward(self,features):
        target_embed = self.ttt.embed_tokens(features['target_id'])
        target_embed = self._target_trans(target_embed)

        with torch.no_grad():
            click_seq = features['clk_sequence']
            batch_size = int(click_seq.shape[0])
            mask = torch.not_equal(click_seq, 0)
            seq_length = torch.maximum(torch.sum(mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
            seq_length = seq_length.to(torch.long)

        logits = self.ttt(click_seq,attention_mask = mask)
        user_state = logits.last_hidden_state[range(batch_size), seq_length, :]

        return torch.sum(user_state * target_embed, dim=1, keepdim=True)
    

class CorrNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=1, output_size=1):
        super(CorrNet, self).__init__()
        hidden_size = 64
        
        self.fc1 = nn.Linear(input_size, hidden_size,bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size,bias=False)
        self.fc3 = nn.Linear(hidden_size, output_size,bias=False)
        self._gru_cell = nn.GRU(64,64,batch_first=True)
        self.xavier_init()
        
    def xavier_init(self):
        a = 4
        init.kaiming_uniform_(self.fc1.weight,nonlinearity='leaky_relu', a=math.sqrt(a))
        init.kaiming_uniform_(self.fc2.weight,nonlinearity='leaky_relu', a=math.sqrt(a))
        init.kaiming_uniform_(self.fc3.weight,nonlinearity='leaky_relu', a=math.sqrt(a))
        init.kaiming_uniform_(self._gru_cell.weight_ih_l0, a=math.sqrt(a))
        init.kaiming_uniform_(self._gru_cell.weight_hh_l0, a=math.sqrt(a))
    def forward(self, x,i):
        mask = x.ne(0)
        sequence_lengths = torch.sum(mask, dim=2)
        mask = sequence_lengths.ne(0)
        sequence_lengths = torch.maximum(torch.sum(mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
        sequence_lengths = sequence_lengths.to(torch.long)
        x, _ = self._gru_cell(x)
        x = x[range(x.shape[0]), sequence_lengths, :]

        x = F.leaky_relu(self.fc1(x))
        x =  F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
import time

class TTTLinearIR(TTTBase):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        self.count = 0
        if config.mode == 'corr':
            input_size = config.hidden_size
            self.P = nn.Parameter(torch.tensor([0.0]))
            self.Q = nn.Parameter(torch.tensor([0.25]))
            hidden_size_b = self.num_heads*self.mini_batch_size*self.head_dim
            self.b1_corrnet = CorrNet(input_size ,hidden_size_b//2 ,hidden_size_b)
            hidden_size_att = self.num_heads*self.mini_batch_size*self.mini_batch_size
            self.Att_corrnet = CorrNet(input_size ,hidden_size_att//2 ,hidden_size_att)
        self.correction = False
    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        click_features = None,
        cache_params: Optional[TTTCache] = None,
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        # in this case, we are decoding
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict(self.layer_idx)

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0
        # use_dual_form = False

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, f]
            b1_init = params_dict["b1_states"]

            # [B,nh,K,f], K=mini_batch_size
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            # [B,nh,K,f]
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)
            if use_dual_form:
                # [B,nh,K,K]
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1)) 
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                if self.correction:
                    pos_seq = inputs['i']
                    click_feature = inputs['X']
                    Attn1_bar_corr = self.Att_corrnet(click_feature,pos_seq).reshape(Attn1.shape)
                    b1_bar_corr = self.b1_corrnet(click_feature,pos_seq).reshape(b1_bar.shape)
                       
                    mean_Attn1_bar = Attn1.mean(dim=-1, keepdim=True)
                    std_Attn1_bar = Attn1.std(dim=-1, keepdim=True)
                    mean_b1_bar = b1_bar.mean(dim=-1, keepdim=True)
                    std_b1_bar = b1_bar.std(dim=-1, keepdim=True)

                    mean_Attn1_bar_corr = Attn1_bar_corr.mean(dim=-1, keepdim=True)
                    std_Attn1_bar_corr = Attn1_bar_corr.std(dim=-1, keepdim=True) +1e-6
                    Attn1_bar_corr = (Attn1_bar_corr - mean_Attn1_bar_corr) / std_Attn1_bar_corr
                    
                    mean_b1_bar_corr = b1_bar_corr.mean(dim=-1, keepdim=True)
                    std_b1_bar_corr = b1_bar_corr.std(dim=-1, keepdim=True) +1e-6
                    b1_bar_corr = (b1_bar_corr - mean_b1_bar_corr) / std_b1_bar_corr

                    Attn1_bar_corr = Attn1_bar_corr * std_Attn1_bar + mean_Attn1_bar
                    b1_bar_corr = b1_bar_corr * std_b1_bar + mean_b1_bar
                    Attn1 = Attn1 + self.P*Attn1_bar_corr 
                    b1_bar = b1_bar + self.Q*b1_bar_corr 

                # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar #最终计算

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch

                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))

        # [B,num_heads, num_mini_batch, mini_batch_size, f] 1024, 4, 10, 1, 16-> [num_mini_batch, B, num_heads, mini_batch_size, f] 10*1024*4*1*16
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        inputs['X'] = click_features
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dict, self.layer_idx, L)

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict

class BlockIR(nn.Module):
    def __init__(self, config: TTTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pre_conv = config.pre_conv

        if config.ttt_layer_type == "linear":
            ttt_layer = TTTLinearIR
        elif config.ttt_layer_type == "mlp":
            ttt_layer = TTTMLP
        else:
            raise ValueError(f"Invalid ttt_layer_type: {config.ttt_layer_type}")

        self.seq_modeling_block = ttt_layer(config=config, layer_idx=layer_idx)

        # self.mlp = SwiGluMLP(config)
        if self.pre_conv:
            self.conv = Conv(config, layer_idx)

        self.seq_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        if self.pre_conv:
            residual = hidden_states
            hidden_states = self.conv(hidden_states, cache_params=cache_params)
            hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.seq_norm(hidden_states)

        # TTT Layer
        hidden_states = self.seq_modeling_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
        )

        return hidden_states



class TTTModelIR(TTTPreTrainedModel):
    """
    Decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Block`]

    Args:
        config: TTTConfig
    """

    def __init__(self, config: TTTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([BlockIR(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[TTTCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        self.layers[0].seq_modeling_block.time_start = time.time()
        if cache_params is None and use_cache:
            cache_params = TTTCache(self, inputs_embeds.size(0))

        seqlen_offset = 0
        if cache_params is not None:
            seqlen_offset = cache_params.seqlen_offset
        position_ids = torch.arange(
            seqlen_offset,
            seqlen_offset + inputs_embeds.shape[1],
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0)

        hidden_states = inputs_embeds

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    cache_params,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_params=cache_params,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return TTTOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )

