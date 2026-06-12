# 🌀 DiffusionGemma Denoising Visualizer

Watch [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
generate text the diffusion way: each block starts as pure noise (every position
masked `░░░░`) and is iteratively **denoised** — tokens get revealed in parallel
over successive steps, roughly in order of model confidence, instead of strictly
left-to-right.

![DiffusionGemma denoising an acrostic ocean poem: the O/C/E/A/N/S line-start letters appear across the canvas before the lines are filled in](https://alainbrown.com/assets/posts/diffusiongemma/denoise-acrostic.gif)

*Denoising the prompt "Write a six-line poem about the ocean where the lines start with the letters O, C, E, A, N, S in that order." Watch the line-start letters land before the lines they head. ([sharper MP4](https://alainbrown.com/assets/posts/diffusiongemma/denoise-acrostic.mp4))*

Each denoising step is rendered as an animated frame:

- `░` gray blocks — still noise (masked positions)
- **green** tokens — revealed *this* step
- **purple** tokens — draft tokens from earlier steps, not yet committed
- white text — committed output

It runs on **Apple silicon** via `mlx-vlm`'s `stream_generate` (`app_mlx.py`),
which exposes the intermediate canvas at every denoising step
(`diffusion_show_unmasking=True`).

## Demo prompts

Structured tasks make the parallel, non-left-to-right denoising most visible —
the model places scaffolding (line starts, list markers, rhymes) early and fills
the middles in later steps:

1. *"Write a six-line poem about the ocean where the lines start with the letters O, C, E, A, N, S in that order."* — acrostic constraints appear before the lines are filled in.
2. *"Why is the sky blue? Answer in exactly three sentences."* — sentence boundaries materialize early.
3. *"Write a Python function `is_palindrome(s)` with a docstring and three doctest examples."* — code structure (def/docstring/tests) denoises as a skeleton first.
4. *"List the eight planets from Neptune inward to Mercury, one per line, each with a one-sentence fun fact."* — list scaffolding shows up across the whole block at once.
5. *"Tell a 100-word story that begins and ends with the sentence 'The last train left at midnight.'"* — both endpoints can be fixed before the middle exists.
6. *"Compose a haiku about machine learning, then briefly explain how it follows the 5-7-5 syllable structure."*

## Run it (Apple silicon / MLX)

On an Apple-silicon Mac you can run the real model natively with MLX using the
pre-quantized 4-bit checkpoint
([`mlx-community/diffusiongemma-26B-A4B-it-4bit`](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit),
~16.6 GB download — fits a 24 GB Mac):

```bash
./run_mlx.sh        # installs MLX deps into .venv on first run, then serves on :7860
```

Then open http://localhost:7860. `app_mlx.py` drives the visualizer through
`mlx-vlm`'s `stream_generate`; `diffusion_show_unmasking=True` surfaces the
per-step canvas. Env-var knobs: `MODEL_ID`; `MAX_CANVAS_LENGTH` (default 128 —
caps per-step memory so the 26B model fits the Metal working set on a 24 GB Mac;
raise it on a larger Mac); `MAX_DENOISING_STEPS`.

MLX uses Metal, so this runs natively on the host (a Mac), not in a container.

## Model & footprint

DiffusionGemma is a 26B-parameter MoE (≈4B active). The MLX 4-bit checkpoint is
~16.6 GB and fits a 24 GB Mac. `MODEL_ID` overrides the checkpoint; the model is
ungated, so no `HF_TOKEN` is required.

## License

Code is MIT — see [`LICENSE`](LICENSE). The model,
[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it),
is Apache-2.0.
