# we have changed the context_length from 1024 to 256 here

GPT_2_small = { "vocab_size": 50257, 
                   "context_length": 256, 
                   "emb_dim": 768, 
                   "n_heads": 12, 
                   "n_layers": 12, 
                   "drop_rate": 0.1, 
                   "qkv_bias": False 
                   }