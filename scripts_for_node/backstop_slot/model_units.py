"""AI work as CUDA-graph units for several kinds of models.

``ChatUnits``     Qwen2.5 prefill units (as ``qwen_units.QwenUnits``) plus decode: one output
                  token is ``decode_groups`` units, each a run of consecutive layers on one
                  token position.  The first group writes the last generated token into the
                  prompt buffer; the last group produces the next token.
``EncoderUnits``  encoder-only models (BERT-style text embedding, ViT image classification):
                  a batch of B items is an ``embed`` unit, one unit per layer, and a ``pool``
                  unit; one set of graphs per batch size.

Every unit is one captured CUDA graph on fixed buffers, so its GPU time alone can be measured
once (``bench_units5.py``) and used as its time bound.
"""

from __future__ import annotations

import torch
from transformers import AutoModel

from qwen_units import QwenUnits


class ChatUnits(QwenUnits):
    """One chat session: prompt buffer, KV cache, prefill units, decode units."""

    def __init__(self, model: str, device: int, chunks=(128, 512, 1024), max_ctx: int = 1152,
                 decode_groups: int = 4) -> None:
        self.decode_groups = decode_groups
        super().__init__(model, device, chunks=tuple(sorted(set(chunks) | {1})), max_ctx=max_ctx)
        self.prefill_chunks = tuple(sorted(c for c in chunks if c != 1))

    def _decode_group(self, group: int) -> None:
        per = -(-self.layers // self.decode_groups)
        if group == 0:
            # Feed: the token produced by the last head becomes the prompt token at this position.
            self.prompt.index_copy_(0, self.chunk_start.view(1), self.token.view(1))
            self._prep(1)
        for index in range(group * per, min(self.layers, (group + 1) * per)):
            self._layer(1, index)
        if group == self.decode_groups - 1:
            self._head(1)

    def _capture(self) -> None:
        super()._capture()
        with torch.cuda.stream(self.stream):
            self.last_index.fill_(0)
            for group in range(self.decode_groups):
                self._decode_group(group)
                self._decode_group(group)
            self.stream.synchronize()
            for group in range(self.decode_groups):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=self.stream):
                    self._decode_group(group)
                self.graphs[("dec", group)] = graph
        self.stream.synchronize()

    def launch_decode(self, group: int, position: int) -> None:
        with torch.cuda.stream(self.stream):
            if group == 0:
                self.chunk_start.fill_(min(position, self.max_ctx - 1))
            if group == self.decode_groups - 1:
                self.last_index.fill_(0)
            self.graphs[("dec", group)].replay()


class EncoderUnits:
    """Encoder-only model on fixed batch sizes: units embed, layer0..layerN-1, pool."""

    def __init__(self, model: str, device: int, batches=(1, 4, 16), seq: int = 128) -> None:
        self.device = torch.device("cuda", device)
        torch.cuda.set_device(self.device)
        torch.set_grad_enabled(False)
        self.hf = AutoModel.from_pretrained(model, torch_dtype=torch.float16, local_files_only=True
                                            ).to(self.device).eval()
        cfg = self.hf.config
        self.vision = cfg.model_type == "vit"
        self.layers = cfg.num_hidden_layers
        self.batches = tuple(sorted(batches))
        self.seq = seq
        dev = self.device
        self.buf = {}
        for b in self.batches:
            if self.vision:
                size = cfg.image_size
                inputs = torch.zeros(b, cfg.num_channels, size, size, dtype=torch.float16, device=dev)
                tokens = (size // cfg.patch_size) ** 2 + 1
            else:
                inputs = torch.zeros(b, seq, dtype=torch.long, device=dev)
                tokens = seq
            self.buf[b] = {"inputs": inputs,
                           "hidden": torch.zeros(b, tokens, cfg.hidden_size, dtype=torch.float16, device=dev),
                           "out": torch.zeros(b, cfg.hidden_size, dtype=torch.float16, device=dev)}
        # Synthetic inputs, written once: the unit times do not depend on the values.
        generator = torch.Generator().manual_seed(7)
        for b in self.batches:
            buf = self.buf[b]["inputs"]
            if self.vision:
                buf.copy_(torch.randn(buf.shape, generator=generator).to(torch.float16))
            else:
                buf.copy_(torch.randint(1000, 20000, buf.shape, generator=generator))
        self.stream = torch.cuda.Stream(device=dev)
        self.graphs: dict[tuple[int, str], torch.cuda.CUDAGraph] = {}
        self._capture()

    def _embed(self, b: int) -> None:
        buf = self.buf[b]
        if self.vision:
            buf["hidden"].copy_(self.hf.embeddings(buf["inputs"]))
        else:
            buf["hidden"].copy_(self.hf.embeddings(input_ids=buf["inputs"]))

    def _layer(self, b: int, index: int) -> None:
        buf = self.buf[b]
        out = self.hf.encoder.layer[index](buf["hidden"])
        buf["hidden"].copy_(out[0] if isinstance(out, (tuple, list)) else out)

    def _pool(self, b: int) -> None:
        buf = self.buf[b]
        hidden = self.hf.layernorm(buf["hidden"]) if self.vision else buf["hidden"]
        buf["out"].copy_(hidden[:, 0])

    def unit_names(self) -> list[str]:
        return ["embed"] + [f"layer{i}" for i in range(self.layers)] + ["pool"]

    def _capture(self) -> None:
        bodies = {}
        for b in self.batches:
            bodies[(b, "embed")] = (lambda bb: (lambda: self._embed(bb)))(b)
            bodies[(b, "pool")] = (lambda bb: (lambda: self._pool(bb)))(b)
            for i in range(self.layers):
                bodies[(b, f"layer{i}")] = (lambda bb, ii: (lambda: self._layer(bb, ii)))(b, i)
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

    def load_inputs(self, b: int, generator: torch.Generator | None = None) -> None:
        """Inputs are synthetic and already in the batch buffer; nothing to copy per batch."""

    def launch(self, b: int, name: str) -> None:
        with torch.cuda.stream(self.stream):
            self.graphs[(b, name)].replay()
