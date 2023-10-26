import torch
#import torch.nn as nn
from models.base import BaseRegressor as Base
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPreTrainedModel
from transformers import BertConfig


class BERT_seq2seq(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()
    
    def forward(self, x):
        embedding_output = self.embeddings(inputs_embeds=x.permute(0,2,1))

        encoder_outputs = self.encoder(embedding_output)

        return encoder_outputs[0].permute(0,2,1)

    
class Model(Base):
    def __init__(self, params, state_dict):
        super(Model, self).__init__(params)


        assert("bert" in params)

        default_params={
            "bert":{
                "attention_probs_dropout_prob": 0.1,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 12,#768,
                "initializer_range": 0.02,
                "intermediate_size": 48,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 4,
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "vocab_size": 1,
                "type_vocab_size": 1
            }
        }
        for key in default_params["bert"]:
            params["bert"][key] = params["bert"].get(key, default_params["bert"][key])

        self.seq2seq = BERT_seq2seq(BertConfig(**params["bert"]))

        if state_dict!=None:
            self.load_state_dict(torch.load(state_dict))
