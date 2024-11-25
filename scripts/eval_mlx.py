import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from moshi_mlx import models

def config1b() -> models.lm.LmConfig:
    transformer = models.lm.TransformerConfig(
        d_model=2048,
        num_heads=16,
        num_layers=16,
        dim_feedforward=2048 * 4,  # dim * hidden_scale
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=750,
        max_period=100000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=4096,
    )
    depformer = models.lm.DepFormerConfig(
        transformer=models.lm.TransformerConfig(
            d_model=1024,
            num_heads=16,
            num_layers=6,
            dim_feedforward=1024 * 4,  # dim * hidden_scale
            causal=True,
            norm_first=True,
            bias_ff=False,
            bias_attn=False,
            layer_scale=None,
            context=8,
            max_period=10000,
            use_conv_block=False,
            use_conv_bias=True,
            cross_attention=False,
            gating=True,
            norm="rms_norm",
            positional_embedding="none",
            conv_layout=False,
            conv_kernel_size=3,
            kv_repeat=1,
            max_seq_len=4096,
        ),
        num_slices=0,
    )
    return models.lm.LmConfig(
        transformer=transformer,
        depformer=depformer,
        audio_vocab_size=2049,
        text_in_vocab_size=48001,
        text_out_vocab_size=48000,
        audio_codebooks=8,
        audio_delays=[0]*16
    )
home_dir = os.path.expanduser("~")

lm_config = config1b()
model = models.Lm(lm_config)
model.set_dtype(mx.bfloat16)
model.load_weights(home_dir + "/tmp/asr-1b-8d2516b9@150.safetensors", strict=True)

def run_one():
    text_token_ids = mx.array([48000]).reshape(1, 1)
    audio_token_ids = [mx.array([2048]).reshape(1, 1)] * 8
    xs = model.text_emb(text_token_ids)
    for token_ids, emb in zip(audio_token_ids, model.audio_embs):
        xs = xs + emb(token_ids)
    transformer_out = model.transformer(xs, cache=model.transformer_cache)
    transformer_out = model.out_norm(transformer_out)
    text_logits = model.text_linear(transformer_out)
    print(text_logits)

run_one()
