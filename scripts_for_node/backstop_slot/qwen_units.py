"""Qwen2.5 prefill as CUDA-graph units with several chunk sizes.

Same unit structure as ``qwen_pieces`` (prep, one unit per layer, head), but a
request may process each chunk with a different token count: small chunks give
short units for tight gaps, large chunks read the weights fewer times per token.
All chunk sizes share one prompt buffer and one KV cache, so a request can mix
them chunk by chunk.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class QwenUnits:
    def __init__(self, model: str, device: int, chunks=(128, 512), max_ctx: int = 1024) -> None:
        self.device = torch.device("cuda", device)
        torch.cuda.set_device(self.device)
        torch.set_grad_enabled(False)
        self.hf = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, local_files_only=True
        ).to(self.device).eval()
        cfg = self.hf.config
        self.layers = cfg.num_hidden_layers
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // self.heads
        self.scale = self.head_dim ** -0.5
        self.chunks = tuple(sorted(chunks))
        self.max_ctx = max_ctx
        dev = self.device
        self.prompt = torch.zeros(max_ctx, dtype=torch.long, device=dev)
        self.chunk_start = torch.zeros((), dtype=torch.long, device=dev)
        self.last_index = torch.zeros((), dtype=torch.long, device=dev)
        self.key_index = torch.arange(max_ctx, device=dev)
        self.k_cache = torch.zeros(self.layers, 1, self.kv_heads, max_ctx, self.head_dim,
                                   dtype=torch.float16, device=dev)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.token = torch.zeros((), dtype=torch.long, device=dev)
        self.logits = torch.zeros(cfg.vocab_size, dtype=torch.float16, device=dev)
        self.buf = {}
        for c in self.chunks:
            self.buf[c] = {
                "offsets": torch.arange(c, device=dev),
                "positions": torch.zeros(c, dtype=torch.long, device=dev),
                "hidden": torch.zeros(1, c, cfg.hidden_size, dtype=torch.float16, device=dev),
                "cos": torch.zeros(1, c, self.head_dim, dtype=torch.float16, device=dev),
                "sin": torch.zeros(1, c, self.head_dim, dtype=torch.float16, device=dev),
                "mask": torch.zeros(1, 1, c, max_ctx, dtype=torch.float32, device=dev),
            }
        self.stream = torch.cuda.Stream(device=dev)
        self.graphs: dict[tuple[int, str], torch.cuda.CUDAGraph] = {}
        self._prompt_len = 0
        self._capture()

    def _prep(self, c: int) -> None:
        b = self.buf[c]
        positions = self.chunk_start + b["offsets"]
        b["positions"].copy_(positions)
        ids = self.prompt.index_select(0, positions.clamp(max=self.max_ctx - 1))
        b["hidden"].copy_(self.hf.model.embed_tokens(ids)[None])
        cos, sin = self.hf.model.rotary_emb(b["hidden"], positions[None])
        b["cos"].copy_(cos)
        b["sin"].copy_(sin)
        allowed = self.key_index[None, :] <= positions[:, None]
        b["mask"].copy_(torch.where(allowed, 0.0, float("-inf"))[None, None])

    def _layer(self, c: int, index: int) -> None:
        b = self.buf[c]
        layer = self.hf.model.layers[index]
        attn = layer.self_attn
        x = b["hidden"]
        h = layer.input_layernorm(x)
        q = attn.q_proj(h).view(1, c, self.heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(1, c, self.kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(1, c, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, b["cos"], b["sin"])
        self.k_cache[index].index_copy_(2, b["positions"], k)
        self.v_cache[index].index_copy_(2, b["positions"], v)
        group = self.heads // self.kv_heads
        qg = (q * self.scale).reshape(1, self.kv_heads, group * c, self.head_dim)
        scores = torch.matmul(qg, self.k_cache[index].transpose(-1, -2)).float()
        scores = scores.view(1, self.kv_heads, group, c, self.max_ctx) + b["mask"][:, :, None]
        probs = torch.softmax(scores, dim=-1).to(torch.float16)
        o = torch.matmul(probs.view(1, self.kv_heads, group * c, self.max_ctx), self.v_cache[index])
        o = o.view(1, self.heads, c, self.head_dim)
        x = x + attn.o_proj(o.transpose(1, 2).reshape(1, c, self.heads * self.head_dim))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        b["hidden"].copy_(x)

    def _head(self, c: int) -> None:
        last = self.buf[c]["hidden"][0].index_select(0, self.last_index[None])
        logits = self.hf.lm_head(self.hf.model.norm(last))[0]
        self.logits.copy_(logits)
        self.token.copy_(torch.argmax(logits))

    def _capture(self) -> None:
        bodies = {}
        for c in self.chunks:
            bodies[(c, "prep")] = (lambda cc: (lambda: self._prep(cc)))(c)
            bodies[(c, "head")] = (lambda cc: (lambda: self._head(cc)))(c)
            for i in range(self.layers):
                bodies[(c, f"layer{i}")] = (lambda cc, ii: (lambda: self._layer(cc, ii)))(c, i)
        with torch.cuda.stream(self.stream):
            for body in bodies.values():
                body()
                body()
            self.stream.synchronize()
            for key, body in bodies.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    body()
                self.graphs[key] = graph
        self.stream.synchronize()

    def chunk_units(self, c: int) -> list[str]:
        return ["prep"] + [f"layer{i}" for i in range(self.layers)]

    def load_prompt(self, token_ids: torch.Tensor) -> None:
        with torch.cuda.stream(self.stream):
            self.prompt[:token_ids.numel()].copy_(token_ids, non_blocking=True)
        self._prompt_len = int(token_ids.numel())

    def launch(self, c: int, name: str, start: int) -> None:
        with torch.cuda.stream(self.stream):
            if name == "prep":
                self.chunk_start.fill_(start)
            elif name == "head":
                self.last_index.fill_(self._prompt_len - 1 - start)
            self.graphs[(c, name)].replay()

    @torch.inference_mode()
    def reference_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.hf(token_ids[None].to(self.device), use_cache=False).logits[0, -1].float()
