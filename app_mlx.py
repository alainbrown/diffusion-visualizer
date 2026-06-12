"""Gradio app that visualizes the block-diffusion denoising process of
DiffusionGemma on Apple silicon via MLX.

It runs the *pre-quantized MLX* checkpoint
(mlx-community/diffusiongemma-26B-A4B-it-4bit, ~16.6GB) natively on Apple
silicon via mlx-vlm.

mlx-vlm already exposes the intermediate denoising canvas: stream_generate()
yields GenerationResult objects, and when diffusion_show_unmasking=True it emits
one `is_draft=True` result per denoising step whose `draft_text` is the current
canvas (revealed tokens as text, unrevealed positions as the literal "[Mask]").
Committed segments arrive as ordinary (non-draft) results carrying `.text`.
That maps one-to-one onto the draft/commit events this UI renders.

NOTE: MLX uses Metal and runs natively on the Mac host (the .venv). See README
for the launch command.

Env vars:
  MODEL_ID   override the model (default mlx-community/diffusiongemma-26B-A4B-it-4bit)
  MAX_DENOISING_STEPS  cap denoising steps per canvas (default: model default)
"""

import html
import os
import queue
import threading
import time

import gradio as gr

MODEL_ID = os.environ.get("MODEL_ID", "mlx-community/diffusiongemma-26B-A4B-it-4bit")
MAX_DENOISING_STEPS = os.environ.get("MAX_DENOISING_STEPS")
# The model's native canvas is 256 positions; denoising a full 256-wide canvas
# (× ~256k vocab per step) OOMs the ~19GB Metal working set on a 24GB Mac. Cap
# the canvas per block so each step fits — longer outputs run as several blocks.
MAX_CANVAS_LENGTH = int(os.environ.get("MAX_CANVAS_LENGTH", "128"))

MASK_MARKER = "[Mask]"  # what mlx-vlm puts in draft_text for unrevealed positions
MASK_GLYPH = "&#9617;"  # ░ shown in its place
MAX_NEW_TOKENS_LIMIT = 1024

DEMO_PROMPTS = [
    "Write a six-line poem about the ocean where the lines start with the letters O, C, E, A, N, S in that order.",
    "Why is the sky blue? Answer in exactly three sentences.",
    "Write a Python function `is_palindrome(s)` with a docstring and three doctest examples.",
    "List the eight planets from Neptune inward to Mercury, one per line, each with a one-sentence fun fact.",
    "Tell a 100-word story that begins and ends with the sentence 'The last train left at midnight.'",
    "Compose a haiku about machine learning, then briefly explain how it follows the 5-7-5 syllable structure.",
]


# ---------------------------------------------------------------------------
# MLX worker thread
# ---------------------------------------------------------------------------
#
# MLX streams have thread affinity: an op must run under a stream that was
# created in the *same* thread, and a stream created on thread A raises
# "There is no Stream(gpu, N) in current thread" when used on thread B. mlx-vlm
# creates its `generation_stream` at import time (the main thread), but Gradio
# runs request handlers in its own worker threads — and may even advance a
# single generator across different threadpool threads. So we keep *all* MLX
# work on one dedicated thread that owns the stream, loads the model, and runs
# every generation. The Gradio handler never touches MLX; it only passes a
# prompt in and reads rendered events back out over plain queues.

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.generate import common as _common, diffusion as _diffusion, dispatch as _dispatch
from mlx_vlm.prompt_utils import apply_chat_template

_requests: "queue.Queue" = queue.Queue()
_ready = threading.Event()


def _mlx_worker():
    """Owns the MLX stream + model; serves generation requests one at a time."""
    stream = mx.new_stream(mx.default_device())  # created in THIS thread
    for _m in (_common, _diffusion, _dispatch):
        if hasattr(_m, "generation_stream"):
            _m.generation_stream = stream
    mx.set_default_stream(stream)

    # The model (~16.6GB) nearly fills the Metal working set (~19GB on a 24GB
    # Mac). Raise the wired limit to the recommended max so generation gets the
    # full headroom for activations; without it MLX OOM-aborts (a hard crash
    # that kills the process) on larger canvases.
    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))

    print(f"Loading {MODEL_ID} via mlx-vlm ...", flush=True)
    model, processor = load(MODEL_ID)
    print("Model loaded.", flush=True)
    _ready.set()

    while True:
        prompt, max_new_tokens, out = _requests.get()
        try:
            formatted = apply_chat_template(processor, model.config, prompt, num_images=0)
            gen_kwargs = {
                "max_tokens": int(max_new_tokens),
                "temperature": 0.0,
                "diffusion_show_unmasking": True,  # yields per-step draft canvases
                "skip_special_tokens": True,       # drop <eos>/<pad> from the canvas
                "diffusion_max_canvas_length": MAX_CANVAS_LENGTH,  # bound per-step memory
            }
            if MAX_DENOISING_STEPS:
                gen_kwargs["max_denoising_steps"] = int(MAX_DENOISING_STEPS)
            for resp in stream_generate(model, processor, formatted, **gen_kwargs):
                if resp.is_draft:
                    out.put(("draft", resp.draft_text,
                             resp.diffusion_step, resp.diffusion_total_steps))
                elif resp.text:
                    out.put(("commit", resp.text, 0, 0))
            out.put(("end", None, 0, 0))
            mx.clear_cache()  # release transient buffers between generations
        except Exception as e:  # surface in the UI instead of a dead spinner
            out.put(("error", f"{type(e).__name__}: {e}", 0, 0))


threading.Thread(target=_mlx_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(committed_text: str, draft_text: str, prev_words: set, step: int,
                 total: int, elapsed: float, done: bool = False) -> str:
    """Render one denoising frame as HTML.

    `draft_text` is the canvas string from mlx-vlm: revealed tokens as text,
    unrevealed positions as the literal "[Mask]" marker. We split on
    whitespace and style each piece: masks → gray block, words not seen in the
    previous draft → highlighted (just revealed), everything else → draft.
    """
    parts = []
    if committed_text:
        parts.append(f'<span class="committed">{html.escape(committed_text)}</span>')
    if committed_text and draft_text:
        parts.append(" ")
    for word in draft_text.split(" "):
        if word == "":
            continue
        if word == MASK_MARKER:
            parts.append(f'<span class="mask">{MASK_GLYPH}</span> ')
        elif word not in prev_words:
            parts.append(f'<span class="fresh">{html.escape(word)}</span> ')
        else:
            parts.append(f'<span class="draft">{html.escape(word)}</span> ')
    status = "done" if done else "denoising"
    step_label = f"step {step}/{total}" if total else f"step {step}"
    badge = (
        f'<div class="statusbar"><span class="badge {status}">'
        f'{"&#10003; complete" if done else "&#8635; denoising"}</span>'
        f"<span>{step_label}</span><span>{elapsed:.1f}s</span></div>"
    )
    return f'<div class="canvas">{badge}<div class="text">{"".join(parts)}</div></div>'


def _words(draft_text: str) -> set:
    return {w for w in draft_text.split(" ") if w and w != MASK_MARKER}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(prompt: str, max_new_tokens: int):
    if not prompt or not prompt.strip():
        yield "<div class='canvas'><div class='text'>Enter a prompt.</div></div>", ""
        return

    # Hand the request to the MLX worker thread and stream its events back. This
    # handler runs on a Gradio worker thread and deliberately does no MLX work.
    out: "queue.Queue" = queue.Queue()
    _requests.put((prompt, int(max_new_tokens), out))

    committed = ""
    draft_text = ""
    prev_words: set = set()
    step = 0
    total = 0
    t0 = time.time()

    while True:
        kind, value, vstep, vtotal = out.get()
        if kind == "end":
            break
        if kind == "error":
            yield (
                f'<div class="canvas"><div class="text">&#9888; '
                f"{html.escape(value)}</div></div>",
                "",
            )
            return
        if kind == "draft":
            step = vstep or (step + 1)
            total = vtotal or total
            prev_words = _words(draft_text)
            draft_text = value
            yield render_frame(committed, draft_text, prev_words, step, total,
                               time.time() - t0), ""
        elif kind == "commit":
            committed += value
            draft_text = ""

    elapsed = time.time() - t0
    final = committed.strip()
    yield render_frame(final, "", set(), step, total, elapsed, done=True), final


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

CSS = """
.canvas {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: #0d1117; border-radius: 12px; padding: 16px;
  min-height: 280px; font-size: 15px; line-height: 1.9;
}
.canvas .text { white-space: pre-wrap; word-break: break-word; }
.statusbar { display: flex; gap: 14px; align-items: center; color: #8b949e;
  font-size: 12px; margin-bottom: 12px; }
.badge { padding: 2px 10px; border-radius: 999px; font-weight: 600; }
.badge.denoising { background: #1f2a44; color: #79c0ff; }
.badge.done { background: #15392b; color: #56d364; }
.committed { color: #e6edf3; }
.draft { color: #d2a8ff; }
.fresh { color: #0d1117; background: #56d364; border-radius: 3px;
  padding: 0 1px; }
.mask { color: #30363d; letter-spacing: 1px; }
"""

with gr.Blocks(title="DiffusionGemma Denoising Visualizer (MLX)") as demo:
    gr.Markdown(
        f"""
# 🌀 DiffusionGemma — watch text get denoised (MLX / Apple silicon)

Unlike autoregressive LLMs that emit one token at a time,
[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) is a **diffusion language
model**: it starts each block from pure noise (all positions masked
<span style="color:#666">░░░░</span>) and iteratively *denoises* it, revealing
tokens over successive steps. <span style="background:#56d364;color:#0d1117;border-radius:3px;padding:0 3px">green</span>
tokens were just revealed this step, <span style="color:#a371f7">purple</span>
tokens are still drafts, white text is committed. Running locally via mlx-vlm.
"""
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Ask anything…",
                lines=3,
                value=DEMO_PROMPTS[0],
            )
            max_tokens = gr.Slider(
                64, MAX_NEW_TOKENS_LIMIT, value=128, step=32, label="Max new tokens"
            )
            go = gr.Button("Denoise ✨", variant="primary")
            gr.Examples(examples=[[p] for p in DEMO_PROMPTS], inputs=[prompt],
                        label="Demo prompts (structured tasks show off parallel denoising)")
        with gr.Column(scale=3):
            canvas = gr.HTML(
                value='<div class="canvas"><div class="text">'
                '<span class="mask">' + MASK_GLYPH * 48 + "</span></div></div>",
                label="Denoising canvas",
            )
            final_text = gr.Textbox(label="Final output", lines=6, interactive=False)

    go.click(generate, inputs=[prompt, max_tokens], outputs=[canvas, final_text])
    prompt.submit(generate, inputs=[prompt, max_tokens], outputs=[canvas, final_text])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=7860, css=CSS
    )
