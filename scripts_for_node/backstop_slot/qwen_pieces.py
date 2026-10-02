"""Qwen2.5 prefill split into CUDA-graph units that fit sub-millisecond gaps.

A prompt is processed in fixed-size token chunks.  Each chunk is a sequence of
units: one ``prep`` unit (embedding, rotary tables, causal mask) and one unit
per decoder layer; the request ends with a ``head`` unit (final norm, LM head
on the last prompt token, argmax).  Every unit is one captured CUDA graph over
static buffers, so its GPU time does not depend on the chunk's position and can
be bounded ahead of time.  The KV cache and hidden state stay on the GPU
between units, so consecutive units may run in different gaps.
"""

from __future__ import annotations

import math

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


class QwenPieces:
    def __init__(self, model: str, device: int, chunk: int = 64, max_ctx: int = 1024) -> None:
        self.device = torch.device("cuda", device)
        torch.cuda.set_device(self.device)
        self.hf = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, local_files_only=True
        ).to(self.device).eval()
        cfg = self.hf.config
        self.layers = cfg.num_hidden_layers
        self.heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // self.heads
        self.scale = self.head_dim ** -0.5
        self.hidden_size = cfg.hidden_size
        self.chunk = chunk
        self.max_ctx = max_ctx
        dev = self.device
        torch.set_grad_enabled(False)
        with torch.no_grad():
            self.prompt = torch.zeros(max_ctx, dtype=torch.long, device=dev)
            self.chunk_start = torch.zeros((), dtype=torch.long, device=dev)
            self.last_index = torch.zeros((), dtype=torch.long, device=dev)
            self.offsets = torch.arange(chunk, device=dev)
            self.key_index = torch.arange(max_ctx, device=dev)
            self.positions = torch.zeros(chunk, dtype=torch.long, device=dev)
            self.hidden = torch.zeros(1, chunk, self.hidden_size, dtype=torch.float16, device=dev)
            self.cos = torch.zeros(1, chunk, self.head_dim, dtype=torch.float16, device=dev)
            self.sin = torch.zeros_like(self.cos)
            self.mask = torch.zeros(1, 1, chunk, max_ctx, dtype=torch.float32, device=dev)
            self.k_cache = torch.zeros(self.layers, 1, self.kv_heads, max_ctx, self.head_dim,
                                       dtype=torch.float16, device=dev)
            self.v_cache = torch.zeros_like(self.k_cache)
            self.token = torch.zeros((), dtype=torch.long, device=dev)
            self.logits = torch.zeros(cfg.vocab_size, dtype=torch.float16, device=dev)
        self.stream = torch.cuda.Stream(device=dev)
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self._capture()

    # ---- unit bodies -------------------------------------------------
    def _prep(self) -> None:
        positions = self.chunk_start + self.offsets
        self.positions.copy_(positions)
        ids = self.prompt.index_select(0, positions.clamp(max=self.max_ctx - 1))
        self.hidden.copy_(self.hf.model.embed_tokens(ids)[None])
        cos, sin = self.hf.model.rotary_emb(self.hidden, positions[None])
        self.cos.copy_(cos)
        self.sin.copy_(sin)
        allowed = self.key_index[None, :] <= positions[:, None]
        self.mask.copy_(torch.where(allowed, 0.0, float("-inf"))[None, None])

    def _layer(self, index: int) -> None:
        layer = self.hf.model.layers[index]
        attn = layer.self_attn
        c = self.chunk
        x = self.hidden
        h = layer.input_layernorm(x)
        q = attn.q_proj(h).view(1, c, self.heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(1, c, self.kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(1, c, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, self.cos, self.sin)
        self.k_cache[index].index_copy_(2, self.positions, k)
        self.v_cache[index].index_copy_(2, self.positions, v)
        # Grouped-query attention over the static cache with fp16 tensor-core
        # GEMMs: the query heads that share one KV head are stacked, so each
        # KV head is read once.  (SDPA with a mask and GQA falls back to an
        # fp32 path that is ten times slower here.)
        group = self.heads // self.kv_heads
        qg = (q * self.scale).reshape(1, self.kv_heads, group * c, self.head_dim)
        scores = torch.matmul(qg, self.k_cache[index].transpose(-1, -2)).float()
        scores = scores.view(1, self.kv_heads, group, c, self.max_ctx) + self.mask[:, :, None]
        probs = torch.softmax(scores, dim=-1).to(torch.float16)
        o = torch.matmul(probs.view(1, self.kv_heads, group * c, self.max_ctx),
                         self.v_cache[index])
        o = o.view(1, self.heads, c, self.head_dim)
        x = x + attn.o_proj(o.transpose(1, 2).reshape(1, c, self.heads * self.head_dim))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        self.hidden.copy_(x)

    def _head(self) -> None:
        last = self.hidden[0].index_select(0, self.last_index[None])
        logits = self.hf.lm_head(self.hf.model.norm(last))[0]
        self.logits.copy_(logits)
        self.token.copy_(torch.argmax(logits))

    def _capture(self) -> None:
        bodies = {"prep": self._prep, "head": self._head}
        for index in range(self.layers):
            bodies[f"layer{index}"] = (lambda i: (lambda: self._layer(i)))(index)
        with torch.no_grad(), torch.cuda.stream(self.stream):
            for body in bodies.values():   # warm kernels and allocator
                body()
                body()
            self.stream.synchronize()
            for name, body in bodies.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    body()
                self.graphs[name] = graph
        self.stream.synchronize()

    # ---- request interface --------------------------------------------
    def units_for(self, prompt_len: int) -> list[tuple[str, int]]:
        """Unit sequence (name, chunk index) for one prompt."""
        chunks = math.ceil(prompt_len / self.chunk)
        units = []
        for chunk in range(chunks):
            units.append(("prep", chunk))
            units.extend((f"layer{i}", chunk) for i in range(self.layers))
        units.append(("head", chunks - 1))
        return units

    def load_prompt(self, token_ids: torch.Tensor) -> None:
        length = int(token_ids.numel())
        if length > self.max_ctx:
            raise ValueError("prompt exceeds the static context")
        with torch.cuda.stream(self.stream):
            self.prompt[:length].copy_(token_ids, non_blocking=True)
            self._prompt_len = length

    def launch(self, name: str, chunk: int) -> None:
        """Enqueue one unit on the runner stream (no host synchronization)."""
        with torch.cuda.stream(self.stream):
            if name == "prep":
                self.chunk_start.fill_(chunk * self.chunk)
            elif name == "head":
                self.last_index.fill_((self._prompt_len - 1) - chunk * self.chunk)
            self.graphs[name].replay()

    @torch.inference_mode()
    def reference_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        out = self.hf(token_ids[None].to(self.device), use_cache=False)
        return out.logits[0, -1].float()
