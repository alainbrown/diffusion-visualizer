"""Drive the real DiffusionGemma Gradio app in a headed-capable Chromium and
record the denoising animation to a WebM via Playwright's record_video_dir.

Usage:
  python record_run.py <prompt_index> <max_tokens> <out_dir> <tag>

Writes into <out_dir>:
  <tag>.webm          raw Playwright video
  <tag>-legend.png    a mid-denoise full-page still (good legend frame)
  <tag>-canvasbox.json  {x,y,width,height} of the .canvas element (for ffmpeg crop)
  <tag>-final.txt     the final generated text
  <tag>-meta.json     {steps_seen, duration_s, ...}
"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROMPTS = [
    "Write a six-line poem about the ocean where the lines start with the letters O, C, E, A, N, S in that order.",
    "Why is the sky blue? Answer in exactly three sentences.",
    "Write a Python function `is_palindrome(s)` with a docstring and three doctest examples.",
    "List the eight planets from Neptune inward to Mercury, one per line, each with a one-sentence fun fact.",
    "Tell a 100-word story that begins and ends with the sentence 'The last train left at midnight.'",
    "Compose a haiku about machine learning, then briefly explain how it follows the 5-7-5 syllable structure.",
]


def main():
    idx = int(sys.argv[1])
    max_tokens = int(sys.argv[2])
    out_dir = Path(sys.argv[3])
    tag = sys.argv[4]
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = PROMPTS[idx]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            device_scale_factor=2,
            record_video_dir=str(out_dir),
            record_video_size={"width": 1280, "height": 900},
        )
        page = context.new_page()
        page.goto("http://localhost:7860", wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(1500)

        # Set the prompt deterministically.
        ta = page.locator("textarea").first
        ta.click()
        ta.fill(prompt)

        # Bump Max new tokens via the slider's number input if present, else keep.
        try:
            num = page.locator("input[type=number]").first
            if num.count():
                num.fill(str(max_tokens))
                num.press("Enter")
        except Exception as e:
            print("slider set skipped:", e)

        page.wait_for_timeout(500)

        # Click Denoise.
        page.get_by_role("button", name="Denoise").click()

        t0 = time.time()
        steps_seen = set()
        legend_saved = False
        last_step_label = ""
        # Poll until badge.done appears or timeout.
        deadline = t0 + 900  # up to 15 min for the real 26B model
        done = False
        while time.time() < deadline:
            # Track step labels for a rough denoising-step count.
            try:
                bar = page.query_selector(".statusbar")
                if bar:
                    txt = bar.inner_text()
                    steps_seen.add(txt.strip())
                    last_step_label = txt.strip()
            except Exception:
                pass

            # Save a legend still once we are clearly mid-denoise (some fresh/draft
            # tokens visible but not yet done).
            if not legend_saved:
                try:
                    has_fresh = page.query_selector(".canvas .fresh") is not None
                    has_mask = page.query_selector(".canvas .mask") is not None
                    badge_done = page.query_selector(".badge.done") is not None
                    if has_fresh and has_mask and not badge_done and time.time() - t0 > 3:
                        page.screenshot(path=str(out_dir / f"{tag}-legend.png"))
                        legend_saved = True
                        print("legend saved at t=%.1f" % (time.time() - t0))
                except Exception:
                    pass

            if page.query_selector(".badge.done") is not None:
                # Confirm final output textbox filled too.
                final_val = page.locator("textarea").nth(1).input_value()
                if final_val.strip():
                    done = True
                    break
            page.wait_for_timeout(120)

        dur = time.time() - t0
        final_val = page.locator("textarea").nth(1).input_value()

        # Fallback legend if mid-denoise window was missed: take one now.
        if not legend_saved:
            page.screenshot(path=str(out_dir / f"{tag}-legend.png"))
            print("legend fallback saved")

        # Hold on the final complete frame a moment so the recording ends on it.
        page.wait_for_timeout(1500)

        # Grab canvas bounding box for ffmpeg cropping (CSS px; multiply by DSR=2).
        box = None
        try:
            el = page.query_selector(".canvas")
            if el:
                box = el.bounding_box()
        except Exception as e:
            print("bbox err:", e)

        # Also grab the prompt column box so we can include the prompt if wanted.
        prompt_box = None
        try:
            pel = page.query_selector("textarea")
            if pel := pel if False else page.query_selector("textarea"):
                prompt_box = pel.bounding_box()
        except Exception:
            pass

        video_path = page.video.path()
        context.close()  # flushes/saves the video
        browser.close()

    # Rename the video to a stable name.
    final_webm = out_dir / f"{tag}.webm"
    Path(video_path).replace(final_webm)

    (out_dir / f"{tag}-final.txt").write_text(final_val)
    if box:
        (out_dir / f"{tag}-canvasbox.json").write_text(json.dumps(box))
    meta = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "done": done,
        "duration_s": round(dur, 1),
        "distinct_statusbar_states": len(steps_seen),
        "last_step_label": last_step_label,
        "canvas_box": box,
        "prompt_box": prompt_box,
        "video": str(final_webm),
    }
    (out_dir / f"{tag}-meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("FINAL TEXT:\n" + final_val)


if __name__ == "__main__":
    main()
