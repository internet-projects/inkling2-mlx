# Inkling 2 Mini on Apple Silicon (MLX)

MLX-VLM with support for [Thinking Machines' Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), the
looped mixture-of-experts model (26.4B parameters, about 3.4B active per token; the middle 12 of its 24 layers run three
times per token). Text and image input, its DSpark speculative-decoding draft, tool calls, and the released NVFP4 / MXFP8
weights loaded as they are.

Since 21 November 2026, Inkling 2 Mini is supported in MLX-VLM itself (pull request #2778, merged into
[Neywa-Labs/mlx-vlm](https://github.com/Neywa-Labs/mlx-vlm) main; not yet in a PyPI release). From v0.2.0 this repository is
that upstream MLX-VLM, pinned at commit `fd6404f`, plus one addition: `mlx_vlm/tools/parsers/inkling.py`, Inkling's tool-call
format, so `mlx_vlm.server` returns OpenAI `tool_calls`.

Our Hugging Face builds (below) use the upstream tensor names since 21 November 2026, so they load both with this repository
and with stock MLX-VLM from main. If you downloaded one of them before that, download it again. v0.1.0 of this repository
(our own looped-model code) only loads the earlier files.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with Thinking Machines.

## Install

Python 3.10 or newer on an Apple Silicon Mac:

```bash
pip install "git+https://github.com/internet-projects/inkling2-mlx@v0.2.3" torch torchvision
```

This installs `mlx-vlm` version `0.8.0+inkling2mini.5`. Without the tool-call parser, stock MLX-VLM from main works too:
`pip install "git+https://github.com/Neywa-Labs/mlx-vlm" torch torchvision`. PyTorch and torchvision are only used on the CPU by the image
processor.

## Models

| Model | Size | Notes |
|---|---|---|
| [netprojects/Inkling-2-Mini-MLX-4-8bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4-8bit) | 19.5 GB + 0.64 GB draft | 4-bit NVFP4 + 8-bit MXFP8 experts as released (lossless), recommended |
| [netprojects/Inkling-2-Mini-MLX-4bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4bit) | 16.5 GB + 0.64 GB draft | plain affine 4-bit |
| [netprojects/Inkling-2-Mini-MLX-3bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-3bit) | 14.1 GB + 0.64 GB draft | 3/4-bit experts, for 24 GB Macs (`prefill_step_size=256`) |
| [netprojects/Inkling-2-Mini-MLX-2.5bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-2.5bit) | 10.8 GB + 0.64 GB draft | 2/3-bit experts, smallest; 16 GB Macs with a raised GPU limit (see its card) |


## Use

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("netprojects/Inkling-2-Mini-MLX-4-8bit")
prompt = apply_chat_template(processor, model.config, "Explain in two sentences why the sky is blue.", reasoning_effort="low")
print(generate(model, processor, prompt, max_tokens=2048, temperature=0.0).text)
```

`reasoning_effort` is part of Inkling's chat template: `none`, `minimal`, `low`, `medium`, `high` (default), `xhigh`, `max`,
or a number from 0.0 to 0.99. The model can think at length even at `none`, so always set `max_tokens`.

Images: pass `num_images=1` to `apply_chat_template` and `image=["picture.png"]` to `generate`.

DSpark speculative decoding, with the draft bundled in the model repository:

```python
from mlx_vlm.speculative.drafters import load_drafter
draft, kind = load_drafter("netprojects/Inkling-2-Mini-MLX-4-8bit")
print(generate(model, processor, prompt, max_tokens=2048, draft_model=draft, draft_kind=kind, draft_block_size=4).text)
```

A block size of 3–4 is fastest on this model (the draft's default of 8 is slower). On an idle MacBook Pro M5 Max, generation
with DSpark was ×1.29 (4-8bit, block 3) to ×1.84 (2.5bit, block 4) faster; the model cards have the full speed tables.
Server: add `--draft-model <same repo> --draft-kind dflash --draft-block-size 4`.

Faster one-token decode (v0.2.3, on by default): the attention no longer copies its key/value history at every step.
On an M5 Max with the 4-bit build, plain decode went from 41.0 to 44.6 tok/s at a 1K prompt and from 36.4 to 41.7 at 16K,
and DSpark (block 4) from 85.4 to 90.9 tok/s at 16K. The output matches the old code up to bf16 rounding (same greedy
tokens in our checks; drafted check rows are bit-identical). `INKLING2_FAST_DECODE=0` switches back to the old code.

OpenAI-compatible server (tool calls returned as `tool_calls`, thinking as `reasoning_content`):

```bash
mlx_vlm.server --model netprojects/Inkling-2-Mini-MLX-4-8bit --port 8000
```

## Checks

On a MacBook Pro M5 Max (128 GB), 21 November 2026, with this repository's MLX-VLM (`fd6404f`), against Thinking Machines' Transformers implementation (pull request
#51050) running on the CPU with the released weights, 12 short chat texts (488 answer positions):

| Build | Same next token | Mean KL | GSM8K (50, reasoning `none`) |
|---|---|---|---|
| 4-8bit (the release, lossless) | 98.4% | 0.0018 | 47/50 |
| 4bit | 95.9% | 0.029 | 47/50 |
| 3bit | 95.5% | 0.042 | 48/50 |
| 2.5bit | 87.5% | 0.190 | 45/50 |

The 4-8bit build's greedy answers to 3 prompts match Thinking Machines' code token for token (24 tokens each).
Tool calls, coding tasks, thinking and image prompts pass on every build. 

Known limitation: when the model calls a tool through `mlx_vlm.server`, `content` can also carry a short sentence of its
reasoning.

## License

MLX-VLM is under the MIT License (`LICENSE`). The Inkling tool-call parser is under the Apache License 2.0
(`LICENSE-APACHE-2.0`); `NOTICE` describes it and the history of this repository. Inkling 2 Mini itself is Apache-2.0 with Thinking Machines'
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).

MLX-VLM's own documentation: [github.com/Neywa-Labs/mlx-vlm](https://github.com/Neywa-Labs/mlx-vlm) (MIT License, see `LICENSE`).
