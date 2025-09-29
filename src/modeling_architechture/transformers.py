import torch
import torch.nn as nn
from input_embedding_layer import inputEmbeddingLayer
from positional_embedding_layer import positionalEncodingLayer
from encoder import encoder
from decoder import decoder
from final_projection_layer import finalProjectionLayer

class transformers(nn.Module):
    def __init__(self,model_config,tokenizer_config):
        super().__init__()
        self.encoder_emb_layer=inputEmbeddingLayer(tokenizer_config['vocab_size'],model_config['enc_cfg']['emb_dim'])
        self.enc_positional_emb_layer=positionalEncodingLayer(model_config['enc_max_seq_len'],model_config['enc_cfg']['emb_dim'],model_config['enc_cfg']['pos_emb_dropout'])
        self.encoder=encoder(
            no_of_enc_blk=model_config['enc_cfg']['no_of_enc_blk'],
            emb_dim=model_config['enc_cfg']['emb_dim'],
            n_heads=model_config['enc_cfg']['n_heads'],
            mha_dropout=model_config['enc_cfg']['mha_dropout'],
            expand_dim=model_config['enc_cfg']['expand_dim'],
            ff_dropout=model_config['enc_cfg']['ff_dropout'],
            sk_dropout=model_config['enc_cfg']['sk_dropout']
        )
        
        self.decoder_emb_layer=inputEmbeddingLayer(tokenizer_config['vocab_size'],model_config['dec_cfg']['emb_dim'])
        self.dec_positional_emb_layer=positionalEncodingLayer(model_config['dec_max_seq_len'],model_config['dec_cfg']['emb_dim'],model_config['dec_cfg']['pos_emb_dropout'])
        self.decoder=decoder(
            no_of_dec_blk=model_config['dec_cfg']['no_of_dec_blk'],
            emb_dim=model_config['dec_cfg']['emb_dim'],
            n_heads=model_config['dec_cfg']['n_heads'],
            mha_dropout=model_config['dec_cfg']['mha_dropout'],
            expand_dim=model_config['dec_cfg']['expand_dim'],
            ff_dropout=model_config['dec_cfg']['ff_dropout'],
            sk_dropout=model_config['dec_cfg']['sk_dropout']
        )
        self.decoder_final_projection=finalProjectionLayer(model_config['dec_cfg']['emb_dim'],tokenizer_config['vocab_size'])

    def encode(self,x,mask=None):
        encoder_input_embedding=self.encoder_emb_layer(x)
        positional_encoded_input_embedding=self.enc_positional_emb_layer(encoder_input_embedding)
        encoder_contexual_embedding=self.encoder(positional_encoded_input_embedding,mask)
        return encoder_contexual_embedding

    def decode(self,x,encoder_output,mask1=None,mask2=None):
        decoder_input_embedding=self.decoder_emb_layer(x)
        positional_encoded_input_embedding=self.dec_positional_emb_layer(decoder_input_embedding)
        decoder_contexual_embedding=self.decoder(positional_encoded_input_embedding,encoder_output,mask1,mask2)
        final_output=self.decoder_final_projection(decoder_contexual_embedding)
        return final_output
    
    ###forward will be used during training.
    def forward(self,encoder_input,decoder_input,src_mask,tgt_mask):
        encoder_output=self.encode(encoder_input,src_mask)
        decoder_output=self.decode(decoder_input,encoder_output,src_mask,tgt_mask)
        return decoder_output