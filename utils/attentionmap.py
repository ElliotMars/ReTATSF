class AttentionMapExtractor:
    def __init__(self):
        self.attn_maps = []

    def hook_fn(self, module, input, output):
        # output 是 decoder layer 的输出，不含 attention map，但 input[1] 是 memory
        # 我们从 module.multihead_attn 里拿
        q = module.norm2(input[0]) if module.norm_first else input[0]  # tgt
        k = input[1]  # memory
        attn_output, attn_weights = module.multihead_attn(
            query=q,
            key=k,
            value=k,
            need_weights=True,
            average_attn_weights=False  # 保留各个 head
        )
        self.attn_maps.append(attn_weights.detach().cpu())

    def register_hooks(self, transformer_decoder):
        for layer in transformer_decoder.layers:
            layer.register_forward_hook(self.hook_fn)

    def get_map(self, layer_idx=0, batch_idx=0, head_idx=0):
        return self.attn_maps[layer_idx][batch_idx][head_idx]

    def clear(self):
        self.attn_maps.clear()

    def visualize(self, layer_idx=0, batch_idx=0, head_idx=0, title="Cross Attention"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        attn = self.get_map(layer_idx, batch_idx, head_idx)
        plt.figure(figsize=(8, 6))
        sns.heatmap(attn, cmap='viridis')
        plt.title(title)
        plt.xlabel("Memory (KV)")
        plt.ylabel("Query (TGT)")
        plt.show()
