"""SD 1.5 UI -- a standalone Gradio app for Stable Diffusion 1.5.

Same shape as Kandinsky UI, Sana UI and Z-Image UI: no Jupyter, no
Colab. Run it, a page opens in your browser, type a prompt, hit
Generate, a picture appears underneath. On top of that house shape
this app carries the classic AUTOMATIC1111 / Forge feature set for
SD 1.5, because SD 1.5 is the model with the biggest ecosystem of
checkpoints and LoRAs ever made:

  - Checkpoint picker: the built-in runwayml/stable-diffusion-v1-5,
    or any single-file .safetensors / .ckpt dropped into models/.
  - Prompt weighting in A1111 syntax -- (word), (word:1.2), [word] --
    through the compel library, for every tab.
  - Prompt blending: a second prompt mixed into the first at the
    embedding level (SD 1.5's fixed [1, 77, 768] CLIP embeddings
    make this the cleanest blend of any model in the family).
  - The familiar sampler list: Euler a, Euler, DPM++ 2M Karras,
    DPM++ SDE Karras, DDIM, LMS -- swapped per generation.
  - Two LoRA slots (files in loras/) with weights.
  - Hires fix, ControlNet (Canny / Depth / OpenPose), clip skip.
  - Text-to-image, image-to-image and inpaint tabs sharing one
    loaded model -- the extra pipelines are built from the loaded
    components, so no second copy of the weights is ever held.
  - An Animation tab: a port of Animation Script v6.0 by
    Animator-Anon (Deforum-inspired, written for A1111). Keyframed
    prompts crossfade through the same embedding machinery, motion
    is applied frame to frame exactly like the script's zoom_at2,
    and finished frames can be assembled to GIF / MP4 / WEBM with
    the ffmpeg binary the imageio-ffmpeg package ships -- no
    system ffmpeg install needed -- and the movie plays right in
    the page. See the Animation section below for what was and
    was not ported.

House features, as in the other apps: a prompt queue (one prompt
per line), {a|b|c} dynamic prompts, a Random prompt button
(random_prompts.txt next to this file, plus built-in word banks),
a session history (last 10, picking one restores the prompt AND
its settings), Stop, Star batch (copies a batch into
images_out/favorites/), Unload models, and batch folders under
images_out/<batch name>/ with continuous 6-digit frame numbers
and a settings.txt block per run logging every setting and seed.

How to run (on a Windows PC with an NVIDIA GPU):

    install.bat      (once -- uses the Python bundled in
                      python_embeded/, installs torch from the
                      cu121 index plus the packages)
    run.bat          (starts the app)

No Python install is needed and no system Python is touched: the
bundle carries its own Python in python_embeded/.

First Generate downloads the base model (about 4 GB) into the
Hugging Face cache on your user profile. Single-file checkpoints
also fetch the small SD 1.5 config files once. ControlNet models
download (about 1.6 GB each) the first time one is used.

Memory: SD 1.5 in full float32 is about 4 GB of weights, so on a
12 GB card it runs with CPU offload OFF and everything resident.
Offload is offered as a checkbox for smaller cards. Memory is
freed between images inside every run (the Z-Image lesson: a
long queue otherwise creeps upward until Windows kills it).

Deliberate exclusions in this version (also in the README):
no VAE picker (most checkpoints bake their VAE in), no face
restoration (GFPGAN / CodeFormer), no X/Y/Z plots, and from the
animation script: no paste-props overlay system and no rendertext
text overlays. Depth / OpenPose ControlNet take a preprocessed
control image (only Canny is preprocessed automatically).

This build is code-verified (stub tests) only until it has run
on the target PC.
"""

import gc
import os
import random
import re
import shutil
import subprocess
from datetime import datetime

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = APP_DIR  # flat bundle: app.py sits next to run.bat
IMAGES_DIR = os.path.join(ROOT_DIR, "images_out")
FAVORITES_DIR = os.path.join(IMAGES_DIR, "favorites")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LORAS_DIR = os.path.join(ROOT_DIR, "loras")
RANDOM_PROMPTS_FILE = os.path.join(ROOT_DIR, "random_prompts.txt")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
BUILTIN_LABEL = "Built-in SD 1.5"

CONTROLNET_MODELS = {
    "Canny": "lllyasviel/sd-controlnet-canny",
    "Depth": "lllyasviel/sd-controlnet-depth",
    "OpenPose": "lllyasviel/sd-controlnet-openpose",
}

SAMPLER_NAMES = [
    "Euler a",
    "Euler",
    "DPM++ 2M Karras",
    "DPM++ SDE Karras",
    "DDIM",
    "LMS",
]

DEFAULT_PROMPT = (
    "A red fox in a sunflower field at golden hour, detailed fur, "
    "warm light"
)
DEFAULT_NEGATIVE = "low quality, bad quality"

_base_pipe = None        # the loaded txt2img pipeline (owns the weights)
_base_key = None         # (checkpoint label, offload) of the loaded pipe
_scheduler_config = None  # scheduler config dict captured at load time
_controlnets = {}         # ControlNet label -> loaded ControlNetModel
_lora_key = None          # tuple of (filename, weight) currently applied
_compel_cache = {}        # (id of text encoder, clip skip) -> Compel


# ------------------------------------------------------------------
# House helpers -- batch folders, frame numbers, seeds, settings.txt
# (the same implementations as Z-Image UI / Kandinsky UI)
# ------------------------------------------------------------------

def _safe_batch_name(name):
    """Strip characters Windows forbids in folder names. Falls back
    to a timestamped name when the box is empty."""
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        name = datetime.now().strftime("sd15-%Y%m%d-%H%M%S")
    return name


def _batch_folder(base):
    """The one folder for a batch name: images_out/<base>, created
    if missing, reused if it already exists. Repeat runs never get
    a _2/_3 folder -- frames just keep numbering upward."""
    return os.path.join(IMAGES_DIR, base)


def _unique_folder(base, root=IMAGES_DIR):
    """<root>/<base>, or <base>_2, <base>_3... if it already exists.
    Used only by Star batch; generations reuse one folder per batch
    name (see _batch_folder) instead of numbered folders."""
    folder = os.path.join(root, base)
    if not os.path.exists(folder):
        return folder
    i = 2
    while True:
        candidate = "{}_{}".format(folder, i)
        if not os.path.exists(candidate):
            return candidate
        i += 1


_FRAME_RE = re.compile(r"^(\d{6})\.png$")


def _next_frame_number(folder):
    """First unused frame number in a folder: one past the highest
    existing 000001-style frame, or 1 for a fresh folder. Only bare
    frame names count, so input_000006.png never affects it."""
    highest = 0
    if os.path.isdir(folder):
        for name in os.listdir(folder):
            match = _FRAME_RE.match(name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _frame_range_text(start, count):
    """'frame 000006' for one image, 'frames 000006-000008' for
    several -- used in status lines and settings headers."""
    if count == 1:
        return "frame {:06d}".format(start)
    return "frames {:06d}-{:06d}".format(start, start + count - 1)


def _seeds_for(seed, count):
    """Resolve the seed box to one seed per image. -1 (or empty)
    draws a fresh random seed per image; a fixed seed S gives S,
    S+1, S+2... so a multi-image run with a fixed seed stays
    reproducible."""
    if seed is None:
        seed = -1
    seed = int(seed)
    if seed < 0:
        return [random.randint(0, 2**31 - 1) for _ in range(count)]
    return [seed + i for i in range(count)]


# Dynamic prompts: single level only, one random pick per group.
# "{a|b|c}" becomes a, b, or c. Groups do not nest; a group with no
# pipe is just its own text with the braces removed.
_TEMPLATE_RE = re.compile(r"\{([^{}]*)\}")


def _expand_templates(prompt):
    """Expand {a|b|c} groups, one random pick each. Plain prompts
    (no braces) come back unchanged."""
    return _TEMPLATE_RE.sub(
        lambda match: random.choice(match.group(1).split("|")), prompt
    )


def _append_settings(folder, header, entries):
    """Append one run's block to settings.txt: a header line (date,
    time, frame range) followed by the run's 'key: value' lines and
    a blank line. Creates the file on first use; earlier runs'
    blocks are never touched. A missing trailing newline is
    repaired so blocks never run together."""
    path = os.path.join(folder, "settings.txt")
    prefix = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(prefix + header + "\n")
        handle.write("\n".join(entries) + "\n\n")


def _append_lines(folder, lines):
    """Append raw lines to settings.txt (used by the animation
    per-frame log). Creates the file if needed."""
    path = os.path.join(folder, "settings.txt")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _run_header(start, count):
    """The settings.txt header for one run: date, time, and the
    frame-number range it produced."""
    return "Run {}, {}".format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        _frame_range_text(start, count))


def _seed_entry(seeds):
    """The settings.txt seed line: 'seed: X' for one image,
    'seeds: X, Y' for several."""
    if len(seeds) == 1:
        return "seed: {}".format(seeds[0])
    return "seeds: {}".format(", ".join(str(s) for s in seeds))


def _expanded_entries(expanded_list):
    """The settings.txt expanded-prompt line(s): one line for one
    image, a comma-joined list for several."""
    if len(expanded_list) == 1:
        return ["expanded_prompt: {}".format(expanded_list[0])]
    return ["expanded_prompts: {}".format(", ".join(expanded_list))]


def _free_memory():
    """Drop cached GPU memory. Best effort: never raises, so a
    cleanup step cannot break the run that called it."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- freeing memory must not fail
        pass


# ------------------------------------------------------------------
# History -- last 10 txt2img jobs, most recent first. Each entry
# keeps the job's settings so picking it restores them, not just
# the prompt text.
# ------------------------------------------------------------------

def _history_add(history, entry):
    """Prepend a job entry (a dict) to the session history: most
    recent first, last 10, duplicates by prompt collapsed."""
    history = [e for e in (history or [])
               if e.get("prompt") != entry.get("prompt")]
    history.insert(0, entry)
    return history[:10]


def _history_update(history):
    """The history dropdown update: choices are the entry prompts,
    most recent first, value cleared so picking the same prompt
    twice still fires."""
    import gradio as gr
    choices = [e.get("prompt", "") for e in (history or [])]
    return gr.update(choices=choices, value=None)


def _history_restore(choice, history):
    """Return the txt2img control values stored for the picked
    history entry: prompt, negative, width, height, steps, CFG,
    sampler, seed. Falls back to the prompt text alone when the
    entry is not found."""
    import gradio as gr
    for entry in (history or []):
        if entry.get("prompt") == choice:
            return (
                entry.get("prompt", ""),
                entry.get("negative", DEFAULT_NEGATIVE),
                entry.get("width", 512),
                entry.get("height", 512),
                entry.get("steps", 25),
                entry.get("cfg", 7.0),
                entry.get("sampler", "Euler a"),
                entry.get("seed", -1),
            )
    return (choice or "", gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update())


def _early(message, last_folder, history):
    """A single yield for an early exit (bad input, load failure):
    empty gallery, the message as status, history untouched."""
    yield [], message, _history_update(history), last_folder, history


def _early_anim(message, last_folder, history):
    """_early for the Animation tab: its event carries one more
    output than the other tabs (the Movie player), which is None
    on an early exit."""
    for out in _early(message, last_folder, history):
        yield out + (None,)


def _noop():
    """Stop-button target: cancels the Generate event, does
    nothing itself."""
    return None


# ------------------------------------------------------------------
# Model loading -- one base load owns the weights; the img2img,
# inpaint and ControlNet pipelines are built from its components
# per call, so switching tabs never holds a second copy.
# ------------------------------------------------------------------

def _list_checkpoints():
    """The checkpoint dropdown choices: the built-in model first,
    then every .safetensors / .ckpt file in models/, sorted."""
    found = []
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            if name.lower().endswith((".safetensors", ".ckpt")):
                found.append(name)
    return [BUILTIN_LABEL] + found


def _list_loras():
    """The LoRA dropdown choices: 'None', then every .safetensors
    file in loras/, sorted."""
    found = []
    if os.path.isdir(LORAS_DIR):
        for name in sorted(os.listdir(LORAS_DIR)):
            if name.lower().endswith(".safetensors"):
                found.append(name)
    return ["None"] + found


def _unload_all():
    """Drop the loaded pipeline, any ControlNet models, the LoRA
    state and the compel cache, and free the memory."""
    global _base_pipe, _base_key, _scheduler_config
    global _controlnets, _lora_key, _compel_cache
    _base_pipe = None
    _base_key = None
    _scheduler_config = None
    _controlnets = {}
    _lora_key = None
    _compel_cache = {}
    _free_memory()


def unload_models():
    """Free the generator (VRAM and RAM)."""
    _unload_all()
    return "Models unloaded -- VRAM and RAM freed."


def _load_base(checkpoint, offload):
    """Load (or reuse) the base txt2img pipeline for a checkpoint.

    The built-in model loads with from_pretrained; a file from
    models/ loads with from_single_file (which fetches the small
    SD 1.5 config files from the Hub once -- an internet connection
    is needed the first time a single-file checkpoint is used).
    Everything loads in float32: at about 4 GB total it fits a
    12 GB card outright, and float32 avoids the fp16 VAE artifacts
    SD 1.5 is known for. The A1111 safety checker is not loaded
    (A1111 has none either); that is stated in the README.

    Raises RuntimeError with a plain sentence on failure."""
    global _base_pipe, _base_key, _scheduler_config, _compel_cache
    key = (checkpoint or BUILTIN_LABEL, bool(offload))
    if _base_pipe is not None and _base_key == key:
        return _base_pipe
    _unload_all()

    import torch
    from diffusers import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU found. SD 1.5 UI needs an NVIDIA GPU with "
            "a recent driver -- CPU-only mode is not supported.")
    label = key[0]
    try:
        if label == BUILTIN_LABEL:
            pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID, torch_dtype=torch.float32,
                safety_checker=None, requires_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                os.path.join(MODELS_DIR, label),
                torch_dtype=torch.float32,
                safety_checker=None, requires_safety_checker=False)
    except Exception as exc:  # noqa: BLE001 -- report, don't crash
        raise RuntimeError(
            "Could not load {} ({}). Check the internet connection "
            "and that Hugging Face is reachable, then try Generate "
            "again.".format(label, exc))
    if key[1]:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    _base_pipe = pipe
    _base_key = key
    _scheduler_config = dict(pipe.scheduler.config)
    _compel_cache = {}
    return pipe


def _place(pipe, offload):
    """Put a freshly built variant pipeline where the base load
    lives: model CPU offload when the checkbox is on, else fully
    on the GPU."""
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def _variant_pipe(kind, offload, controlnet_label=None):
    """Build the pipeline variant for one generation from the base
    pipeline's components (shared weights -- no second copy).

    kind is "txt2img" (the base pipe itself), "img2img",
    "inpaint" or "controlnet" (which also needs a ControlNet
    model label). Variants are built fresh per call and dropped
    afterwards, so offload hooks and schedulers never stack up
    across tabs."""
    base = _base_pipe
    if base is None:
        raise RuntimeError("No model is loaded.")
    if kind == "txt2img":
        return base
    components = base.components
    from diffusers import (
        StableDiffusionControlNetPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
    )
    if kind == "img2img":
        pipe = StableDiffusionImg2ImgPipeline(**components)
    elif kind == "inpaint":
        pipe = StableDiffusionInpaintPipeline(**components)
    elif kind == "controlnet":
        controlnet = _load_controlnet(controlnet_label)
        pipe = StableDiffusionControlNetPipeline(
            **components, controlnet=controlnet)
    else:
        raise RuntimeError("Unknown pipeline kind: {}".format(kind))
    return _place(pipe, offload)


def _load_controlnet(label):
    """Load (once, then cached) a ControlNet model by UI label."""
    if label in _controlnets:
        return _controlnets[label]
    import torch
    from diffusers import ControlNetModel
    repo = CONTROLNET_MODELS[label]
    try:
        model = ControlNetModel.from_pretrained(
            repo, torch_dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001 -- plain status text
        raise RuntimeError(
            "Could not load the {} ControlNet model ({}). Check "
            "the internet connection, then try again.".format(
                label, exc))
    _controlnets[label] = model
    return model


def _make_scheduler(name):
    """Build the scheduler for a sampler name from the loaded
    pipeline's own scheduler config (so betas and timestep
    spacing always match the checkpoint's training).

    Clip-relevant mappings (A1111 name -> Diffusers class):
      Euler a          -> EulerAncestralDiscreteScheduler
      Euler            -> EulerDiscreteScheduler
      DPM++ 2M Karras  -> DPMSolverMultistepScheduler, karras
                          (its default algorithm is dpmsolver++)
      DPM++ SDE Karras -> DPMSolverSinglestepScheduler, karras
      DDIM             -> DDIMScheduler
      LMS              -> LMSDiscreteScheduler, karras (needs scipy)
    """
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LMSDiscreteScheduler,
    )
    config = _scheduler_config
    if name == "Euler":
        return EulerDiscreteScheduler.from_config(config)
    if name == "DPM++ 2M Karras":
        return DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True)
    if name == "DPM++ SDE Karras":
        return DPMSolverSinglestepScheduler.from_config(
            config, use_karras_sigmas=True)
    if name == "DDIM":
        return DDIMScheduler.from_config(config)
    if name == "LMS":
        return LMSDiscreteScheduler.from_config(
            config, use_karras_sigmas=True)
    return EulerAncestralDiscreteScheduler.from_config(config)


def _make_generator(seed):
    """A CPU generator for one image (the house pattern: Diffusers
    moves it to the right device itself)."""
    import torch
    return torch.Generator("cpu").manual_seed(int(seed))


# ------------------------------------------------------------------
# Conditioning -- prompt weighting via compel, with a plain
# tokenization fallback that is numerically identical for
# unweighted text (verified against compel at build time).
#
# Clip skip follows A1111's numbering, NOT the raw Diffusers
# parameter (they differ by one): UI 1 = the CLIP last hidden
# state, the regular SD conditioning; UI 2 = the penultimate
# hidden state run through the final layer norm -- what A1111
# calls stopping one layer early. In compel's terms those are
# LAST_HIDDEN_STATES_NORMALIZED ("SD1/2 regular") and
# PENULTIMATE_HIDDEN_STATES_NORMALIZED ("SD1.5 with clip skip").
# ------------------------------------------------------------------

def _compel_for(pipe, clip_skip):
    """The cached Compel instance for the loaded text encoder at
    one clip-skip value. truncate_long_prompts stays at compel's
    own default (True), so embeddings are always [1, 77, 768] --
    the fixed shape that makes blending a clean lerp."""
    key = (id(pipe.text_encoder), int(clip_skip))
    if key in _compel_cache:
        return _compel_cache[key]
    from compel import Compel, ReturnedEmbeddingsType
    if int(clip_skip) >= 2:
        kind = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
    else:
        kind = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
    compel = Compel(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        returned_embeddings_type=kind,
        requires_pooled=False,
        truncate_long_prompts=True)
    _compel_cache[key] = compel
    return compel


def _encode_text(pipe, text, clip_skip):
    """Encode one prompt side to a [1, 77, 768] embedding tensor.

    Returns (tensor, used_compel). Compel handles the A1111
    weighting syntax -- (word), (word:1.2), [word]. If compel
    fails for any reason, falls back to plain tokenization with
    the same clip-skip layer choice, and reports used_compel
    False so the caller can say so in the status and settings."""
    import torch
    text = text or ""
    try:
        compel = _compel_for(pipe, clip_skip)
        with torch.no_grad():
            embeds = compel(text)
        return embeds, True
    except Exception:  # noqa: BLE001 -- fall back, and say so
        pass
    tokenizer = pipe.tokenizer
    encoder = pipe.text_encoder
    device = next(encoder.parameters()).device
    inputs = tokenizer(
        [text], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt")
    with torch.no_grad():
        if int(clip_skip) >= 2:
            hidden = encoder(
                inputs.input_ids.to(device),
                output_hidden_states=True).hidden_states[-2]
            embeds = encoder.text_model.final_layer_norm(hidden)
        else:
            embeds = encoder(inputs.input_ids.to(device))[0]
    return embeds, False


def _conditioning(pipe, prompt, negative, clip_skip):
    """(prompt_embeds, negative_prompt_embeds, used_compel) for a
    plain generation: both sides encoded once."""
    pos, ok_pos = _encode_text(pipe, prompt, clip_skip)
    neg, ok_neg = _encode_text(pipe, negative, clip_skip)
    return pos, neg, (ok_pos and ok_neg)


def _blended_conditioning(pipe, prompt_a, prompt_b, weight,
                          negative, clip_skip):
    """Prompt blending: encode both prompts, take the weighted
    mix (weight 0 = prompt A only, 1 = prompt B only), and encode
    the single negative prompt once. Both sides are [1, 77, 768],
    so the mix is a plain lerp -- the same embedding-level blend
    Kandinsky does in its prior, and A1111's AND composition in
    spirit. Weighting syntax applies inside each side."""
    weight = max(0.0, min(1.0, float(weight)))
    if (prompt_a or "") == (prompt_b or ""):
        return _conditioning(pipe, prompt_a, negative, clip_skip)
    embeds_a, ok_a = _encode_text(pipe, prompt_a, clip_skip)
    embeds_b, ok_b = _encode_text(pipe, prompt_b, clip_skip)
    pos = (1.0 - weight) * embeds_a + weight * embeds_b
    neg, ok_neg = _encode_text(pipe, negative, clip_skip)
    return pos, neg, (ok_a and ok_b and ok_neg)


# ------------------------------------------------------------------
# LoRA -- two slots over the shared unet / text encoder. Adapters
# live on the shared modules, so every pipeline variant sees them.
# ------------------------------------------------------------------

def _lora_state_text(lora_key):
    """settings.txt wording for the applied LoRA state."""
    if not lora_key:
        return "none"
    return ", ".join("{} @ {}".format(name, weight)
                     for name, weight in lora_key)


def _reconcile_loras(pipe, slot1, weight1, slot2, weight2):
    """Make the loaded adapters match the two UI slots. Loads or
    unloads only when the desired state actually changed. Returns
    the state text for settings.txt. Raises RuntimeError with a
    plain sentence when a LoRA file fails to load."""
    global _lora_key
    desired = []
    for filename, weight in ((slot1, weight1), (slot2, weight2)):
        if filename and filename != "None":
            desired.append((filename, float(weight)))
    desired_key = tuple(desired)
    if desired_key == _lora_key:
        return _lora_state_text(_lora_key)
    try:
        if _lora_key:
            pipe.unload_lora_weights()
        names = []
        weights = []
        for index, (filename, weight) in enumerate(desired):
            adapter = "lora{}".format(index)
            pipe.load_lora_weights(
                LORAS_DIR, weight_name=filename, adapter_name=adapter)
            names.append(adapter)
            weights.append(weight)
        if names:
            pipe.set_adapters(names, adapter_weights=weights)
    except Exception as exc:  # noqa: BLE001 -- plain status text
        _lora_key = None
        raise RuntimeError(
            "Could not load a LoRA ({}). Check the file is an "
            "SD 1.5 LoRA in the loras folder, then try again.".format(
                exc))
    _lora_key = desired_key
    return _lora_state_text(_lora_key)


# ------------------------------------------------------------------
# Random prompt -- random_prompts.txt next to app.py, read fresh
# on every click; the built-in word banks are the fallback when
# the file is missing or empty.
# ------------------------------------------------------------------

_SUBJECTS = [
    "a lighthouse", "a fox", "a mountain cabin", "a castle",
    "a harbor town", "a red panda", "a sailing ship",
    "a robot gardener", "a treehouse village", "a steam train",
]
_SETTINGS = [
    "a rocky cliff", "a misty forest", "desert dunes",
    "a snowy village", "a neon city street", "a quiet lake",
    "a sunflower field", "a space station", "a coral reef",
    "an old library",
]
_LIGHTS = [
    "golden hour light", "storm light", "moonlight",
    "neon dusk", "soft morning fog", "candlelight",
    "sunset clouds", "starlight",
]
_STYLES = [
    "photorealistic", "oil painting", "anime style",
    "watercolor", "cyberpunk", "storybook illustration",
]


def random_prompt():
    """One random prompt: a line from random_prompts.txt when the
    file has lines, else the built-in word banks."""
    try:
        with open(RANDOM_PROMPTS_FILE, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        if lines:
            return random.choice(lines)
    except OSError:
        pass
    return "A {} in {}, {}, {}, detailed.".format(
        random.choice(_SUBJECTS),
        random.choice(_SETTINGS),
        random.choice(_LIGHTS),
        random.choice(_STYLES),
    )


def _control_image(raw_image, label, width, height):
    """Prepare the ControlNet conditioning image at the
    generation size. Canny gets its edges extracted here (the
    one automatic preprocessor); Depth and OpenPose take the
    uploaded image as the already-preprocessed control map,
    exactly as the UI label and README say."""
    from PIL import Image
    img = raw_image.convert("RGB").resize(
        (int(width), int(height)), Image.LANCZOS)
    if label == "Canny":
        import cv2
        import numpy as np
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        img = Image.fromarray(edges).convert("RGB")
    return img


def _round8(value):
    """Nearest multiple of 8, at least 64 -- SD sizes must divide
    by 8 for the VAE."""
    return max(64, int(round(int(value) / 8.0)) * 8)


def _common_entries(mode, checkpoint, sampler, clip_skip, weighting_ok):
    """The settings.txt lines shared by the three image tabs."""
    return [
        "mode: {}".format(mode),
        "checkpoint: {}".format(checkpoint or BUILTIN_LABEL),
        "sampler: {}".format(sampler),
        "clip_skip: {}".format(int(clip_skip)),
        "prompt_weighting: {}".format(
            "compel" if weighting_ok else "plain -- compel unavailable"),
    ]


def generate(prompt_text, negative_text, queue_text, width, height,
             steps, cfg, sampler, clip_skip, seed, batch_name, count,
             blend_prompt_text, blend_weight,
             hires_enabled, hires_factor, hires_denoise,
             cn_enabled, cn_model, cn_image, cn_weight,
             checkpoint, offload,
             lora1, lora1_weight, lora2, lora2_weight,
             last_folder, history):
    """Text-to-image generation (a generator: it yields after
    every image so the page updates live and Stop can cancel).

    Queue beats the prompt box. Dynamics {a|b|c} re-expand per
    image. Blending is on when the blend prompt box has text.
    Hires fix and ControlNet follow their accordions."""
    history = history or []
    count = int(count) if count else 1
    count = max(1, min(10, count))
    width = _round8(max(64, min(1024, int(width))))
    height = _round8(max(64, min(1024, int(height))))
    steps = max(1, min(150, int(steps)))
    offload = bool(offload)

    lines = [line.strip() for line in (queue_text or "").splitlines()
             if line.strip()]
    if lines:
        templates = lines
    elif prompt_text and prompt_text.strip():
        templates = [prompt_text.strip()]
    else:
        yield from _early("Type a prompt first, or fill the queue.",
                          last_folder, history)
        return
    negative = (negative_text or "").strip()

    if cn_enabled and cn_image is None:
        yield from _early(
            "ControlNet is on but no control image is uploaded.",
            last_folder, history)
        return

    try:
        pipe = _load_base(checkpoint, offload)
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return
    try:
        lora_text = _reconcile_loras(
            pipe, lora1, lora1_weight, lora2, lora2_weight)
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return

    blending = bool(blend_prompt_text and blend_prompt_text.strip())
    blend_template = (blend_prompt_text or "").strip()
    control_img = None
    if cn_enabled:
        control_img = _control_image(cn_image, cn_model, width, height)

    base = _safe_batch_name(batch_name)
    folder = _batch_folder(base)
    os.makedirs(folder, exist_ok=True)

    gallery = []
    weighting_ok = True
    for template in templates:
        start = _next_frame_number(folder)
        seeds = _seeds_for(seed, count)
        expanded_list = []
        for i in range(count):
            expanded = _expand_templates(template)
            expanded_list.append(expanded)
            try:
                if blending:
                    blend_expanded = _expand_templates(blend_template)
                    pos, neg, ok = _blended_conditioning(
                        pipe, expanded, blend_expanded,
                        blend_weight, negative, clip_skip)
                else:
                    pos, neg, ok = _conditioning(
                        pipe, expanded, negative, clip_skip)
                weighting_ok = weighting_ok and ok
                if cn_enabled:
                    work = _variant_pipe("controlnet", offload,
                                         cn_model)
                else:
                    work = _variant_pipe("txt2img", offload)
                work.scheduler = _make_scheduler(sampler)
                kwargs = dict(
                    prompt_embeds=pos,
                    negative_prompt_embeds=neg,
                    height=height, width=width,
                    num_inference_steps=steps,
                    guidance_scale=float(cfg),
                    generator=_make_generator(seeds[i]))
                if cn_enabled:
                    kwargs["image"] = control_img
                    kwargs["controlnet_conditioning_scale"] = \
                        float(cn_weight)
                image = work(**kwargs).images[0]
                if hires_enabled:
                    factor = float(hires_factor)
                    target_w = _round8(width * factor)
                    target_h = _round8(height * factor)
                    from PIL import Image as PILImage
                    upscaled = image.resize(
                        (target_w, target_h), PILImage.LANCZOS)
                    img2img = _variant_pipe("img2img", offload)
                    img2img.scheduler = _make_scheduler(sampler)
                    if target_w * target_h > 1572864:
                        # The extreme corner (1024 base at 2x):
                        # slice attention so it fits the card.
                        img2img.enable_attention_slicing()
                        img2img.enable_vae_slicing()
                    image = img2img(
                        prompt_embeds=pos,
                        negative_prompt_embeds=neg,
                        image=upscaled,
                        strength=float(hires_denoise),
                        num_inference_steps=steps,
                        guidance_scale=float(cfg),
                        generator=_make_generator(seeds[i]),
                    ).images[0]
            except Exception as exc:  # noqa: BLE001 -- plain status
                yield (gallery,
                       "Generation failed on {}: {}".format(
                           _frame_range_text(start + i, 1), exc),
                       _history_update(history), folder, history)
                return
            frame = start + i
            image.save(os.path.join(
                folder, "{:06d}.png".format(frame)))
            gallery.append(image)
            status = "Saved {} in {} ({}x{}, {} steps, CFG {}). " \
                "Seeds so far: {}.".format(
                    _frame_range_text(start, i + 1), folder,
                    width, height, steps, cfg,
                    ", ".join(str(s) for s in seeds[:i + 1]))
            if not weighting_ok:
                status += " Note: prompt weighting fell back to " \
                    "plain tokenization this run."
            yield (gallery, status, _history_update(history),
                   folder, history)
            _free_memory()
        entries = _common_entries(
            "txt2img", checkpoint, sampler, clip_skip, weighting_ok)
        entries += [
            "size: {}x{}".format(width, height),
            "steps: {}".format(steps),
            "guidance_scale: {}".format(float(cfg)),
            "offload: {}".format(offload),
            "loras: {}".format(lora_text),
            "prompt: {}".format(template),
        ]
        entries += _expanded_entries(expanded_list)
        entries += ["negative_prompt: {}".format(negative)]
        if blending:
            entries += [
                "blend_prompt: {}".format(blend_template),
                "blend_weight: {}".format(float(blend_weight)),
            ]
        else:
            entries += ["blending: off"]
        if hires_enabled:
            entries += [
                "hires_fix: on",
                "hires_factor: {}".format(float(hires_factor)),
                "hires_denoise: {}".format(float(hires_denoise)),
                "hires_size: {}x{}".format(
                    _round8(width * float(hires_factor)),
                    _round8(height * float(hires_factor))),
            ]
        else:
            entries += ["hires_fix: off"]
        if cn_enabled:
            entries += [
                "controlnet: {}".format(cn_model),
                "controlnet_weight: {}".format(float(cn_weight)),
            ]
        else:
            entries += ["controlnet: off"]
        entries += [_seed_entry(seeds)]
        _append_settings(folder, _run_header(start, count), entries)
        history = _history_add(history, {
            "prompt": template,
            "negative": negative,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": float(cfg),
            "sampler": sampler,
            "seed": int(seed) if seed is not None else -1,
        })
        status = "Done: {} in {} ({}x{}, {} steps, CFG {}). " \
            "Seeds: {}.".format(
                _frame_range_text(start, count), folder,
                width, height, steps, cfg,
                ", ".join(str(s) for s in seeds))
        yield (gallery, status, _history_update(history),
               folder, history)


def star_batch(last_folder):
    """Copy the most recently generated batch folder into
    images_out/favorites/ under a unique name (never overwrite).
    If nothing has run yet this session, say so in the status."""
    if not last_folder or not os.path.isdir(last_folder):
        return "No batch has run yet in this session."
    os.makedirs(FAVORITES_DIR, exist_ok=True)
    base = os.path.basename(os.path.normpath(last_folder))
    dest = _unique_folder(base, root=FAVORITES_DIR)
    shutil.copytree(last_folder, dest)
    return "Starred to {}".format(dest)


def _prepare_init_image(raw_image):
    """An uploaded image as RGB at a multiple-of-8 size (the VAE
    needs it), resizing only when the size is off."""
    from PIL import Image
    img = raw_image.convert("RGB")
    width = _round8(img.size[0])
    height = _round8(img.size[1])
    if (width, height) != img.size:
        img = img.resize((width, height), Image.LANCZOS)
    return img


def generate_img2img(init_image, prompt_text, negative_text, denoise,
                     steps, cfg, sampler, clip_skip, seed, batch_name,
                     count, checkpoint, offload,
                     lora1, lora1_weight, lora2, lora2_weight,
                     last_folder, history):
    """Image-to-image: the uploaded picture, repainted at the
    denoise strength. Same generate core as txt2img -- prompt
    weighting, samplers, LoRAs and clip skip all apply."""
    history = history or []
    count = int(count) if count else 1
    count = max(1, min(10, count))
    steps = max(1, min(150, int(steps)))
    offload = bool(offload)
    if init_image is None:
        yield from _early("Upload an image first.",
                          last_folder, history)
        return
    if not prompt_text or not prompt_text.strip():
        yield from _early("Type a prompt first.",
                          last_folder, history)
        return
    prompt = prompt_text.strip()
    negative = (negative_text or "").strip()

    try:
        pipe = _load_base(checkpoint, offload)
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return
    try:
        lora_text = _reconcile_loras(
            pipe, lora1, lora1_weight, lora2, lora2_weight)
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return

    init = _prepare_init_image(init_image)
    width, height = init.size
    base = _safe_batch_name(batch_name)
    folder = _batch_folder(base)
    os.makedirs(folder, exist_ok=True)
    start = _next_frame_number(folder)
    init.save(os.path.join(
        folder, "input_{:06d}.png".format(start)))

    seeds = _seeds_for(seed, count)
    expanded_list = []
    gallery = []
    weighting_ok = True
    for i in range(count):
        expanded = _expand_templates(prompt)
        expanded_list.append(expanded)
        try:
            pos, neg, ok = _conditioning(
                pipe, expanded, negative, clip_skip)
            weighting_ok = weighting_ok and ok
            work = _variant_pipe("img2img", offload)
            work.scheduler = _make_scheduler(sampler)
            image = work(
                prompt_embeds=pos,
                negative_prompt_embeds=neg,
                image=init,
                strength=float(denoise),
                num_inference_steps=steps,
                guidance_scale=float(cfg),
                generator=_make_generator(seeds[i]),
            ).images[0]
        except Exception as exc:  # noqa: BLE001 -- plain status
            yield (gallery,
                   "Generation failed on {}: {}".format(
                       _frame_range_text(start + i, 1), exc),
                   _history_update(history), folder, history)
            return
        frame = start + i
        image.save(os.path.join(folder, "{:06d}.png".format(frame)))
        gallery.append(image)
        yield (gallery,
               "Saved {} in {} (img2img, denoise {}). Seeds so "
               "far: {}.".format(
                   _frame_range_text(start, i + 1), folder,
                   float(denoise),
                   ", ".join(str(s) for s in seeds[:i + 1])),
               _history_update(history), folder, history)
        _free_memory()
    entries = _common_entries(
        "img2img", checkpoint, sampler, clip_skip, weighting_ok)
    entries += [
        "size: {}x{}".format(width, height),
        "steps: {}".format(steps),
        "guidance_scale: {}".format(float(cfg)),
        "denoise: {}".format(float(denoise)),
        "offload: {}".format(offload),
        "loras: {}".format(lora_text),
        "prompt: {}".format(prompt),
    ]
    entries += _expanded_entries(expanded_list)
    entries += ["negative_prompt: {}".format(negative),
                _seed_entry(seeds)]
    _append_settings(folder, _run_header(start, count), entries)
    status = "Done: {} in {} (img2img, denoise {}). Seeds: {}.".format(
        _frame_range_text(start, count), folder, float(denoise),
        ", ".join(str(s) for s in seeds))
    yield (gallery, status, _history_update(history), folder, history)


def generate_inpaint(base_image, mask_image, prompt_text,
                     negative_text, steps, cfg, sampler, clip_skip,
                     seed, batch_name, count, checkpoint, offload,
                     lora1, lora1_weight, lora2, lora2_weight,
                     last_folder, history):
    """Inpainting: repaint the white parts of the mask, at full
    denoise (1.0) as inpainting is meant to run. The base image
    and mask are saved next to the frames for provenance."""
    history = history or []
    count = int(count) if count else 1
    count = max(1, min(10, count))
    steps = max(1, min(150, int(steps)))
    offload = bool(offload)
    if base_image is None:
        yield from _early("Upload an image first.",
                          last_folder, history)
        return
    if mask_image is None:
        yield from _early(
            "Upload a mask image too -- white is repainted.",
            last_folder, history)
        return
    if not prompt_text or not prompt_text.strip():
        yield from _early("Type a prompt first.",
                          last_folder, history)
        return
    prompt = prompt_text.strip()
    negative = (negative_text or "").strip()

    try:
        pipe = _load_base(checkpoint, offload)
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return
    try:
        lora_text = _reconcile_loras(
            pipe, lora1, lora1_weight, lora2, lora2_weight)
    except RuntimeError as exc:
        yield from _early(str(exc), last_folder, history)
        return

    from PIL import Image
    init = _prepare_init_image(base_image)
    width, height = init.size
    mask = mask_image.convert("L")
    if mask.size != init.size:
        mask = mask.resize(init.size, Image.NEAREST)

    base = _safe_batch_name(batch_name)
    folder = _batch_folder(base)
    os.makedirs(folder, exist_ok=True)
    start = _next_frame_number(folder)
    init.save(os.path.join(
        folder, "input_{:06d}.png".format(start)))
    mask.save(os.path.join(
        folder, "mask_{:06d}.png".format(start)))

    seeds = _seeds_for(seed, count)
    expanded_list = []
    gallery = []
    weighting_ok = True
    for i in range(count):
        expanded = _expand_templates(prompt)
        expanded_list.append(expanded)
        try:
            pos, neg, ok = _conditioning(
                pipe, expanded, negative, clip_skip)
            weighting_ok = weighting_ok and ok
            work = _variant_pipe("inpaint", offload)
            work.scheduler = _make_scheduler(sampler)
            image = work(
                prompt_embeds=pos,
                negative_prompt_embeds=neg,
                image=init,
                mask_image=mask,
                strength=1.0,
                num_inference_steps=steps,
                guidance_scale=float(cfg),
                generator=_make_generator(seeds[i]),
            ).images[0]
        except Exception as exc:  # noqa: BLE001 -- plain status
            yield (gallery,
                   "Generation failed on {}: {}".format(
                       _frame_range_text(start + i, 1), exc),
                   _history_update(history), folder, history)
            return
        frame = start + i
        image.save(os.path.join(folder, "{:06d}.png".format(frame)))
        gallery.append(image)
        yield (gallery,
               "Saved {} in {} (inpaint). Seeds so far: {}.".format(
                   _frame_range_text(start, i + 1), folder,
                   ", ".join(str(s) for s in seeds[:i + 1])),
               _history_update(history), folder, history)
        _free_memory()
    entries = _common_entries(
        "inpaint", checkpoint, sampler, clip_skip, weighting_ok)
    entries += [
        "size: {}x{}".format(width, height),
        "steps: {}".format(steps),
        "guidance_scale: {}".format(float(cfg)),
        "denoise: 1.0",
        "offload: {}".format(offload),
        "loras: {}".format(lora_text),
        "prompt: {}".format(prompt),
    ]
    entries += _expanded_entries(expanded_list)
    entries += ["negative_prompt: {}".format(negative),
                _seed_entry(seeds)]
    _append_settings(folder, _run_header(start, count), entries)
    status = "Done: {} in {} (inpaint). Seeds: {}.".format(
        _frame_range_text(start, count), folder,
        ", ".join(str(s) for s in seeds))
    yield (gallery, status, _history_update(history), folder, history)


# ------------------------------------------------------------------
# Animation -- a port of Animation Script v6.0 by Animator-Anon
# (Deforum-inspired; Alan's long-time A1111 tool) into this app's
# Diffusers machinery. The script cannot run literally here (it
# drives A1111's modules API), so its BEHAVIOR is re-implemented:
#
#   - zoom_at2, addnoise and morph are ported nearly line for
#     line from the script.
#   - The keyframe parser follows run(): lines of
#     "time_s | command | ..." become per-frame series. Prompt
#     keyframes pair up (current + next) with a weight that is 1
#     at a keyframe, ramps to 0 at the frame before the next
#     keyframe, and resets -- exactly the script's dataframe
#     construction, so a segment crossfades current -> next and
#     lands on the next keyframe's prompt at its frame. A1111
#     rendered that as an AND composition; here the same maths
#     runs at the embedding level: w * E(current) + (1 - w) *
#     E(next), for the negative side too.
#   - Motion is per second in the UI (zoom factor, pixel shifts,
#     rotation) and converted to per-frame amounts with the
#     script's own formulas (shift / fps, zoom ** (1 / fps),
#     rotation / fps); the dataframe interpolation between
#     keyframes is LINEAR in this port (the script used a
#     polynomial pass for columns with many keyframes).
#   - Frame 1 is txt2img; every later frame is the previous
#     frame transformed, noised (when Add_Noise is on), then
#     img2img at the frame's denoise value.
#
# Deliberate differences from the script, all stated in the
# README too: seed keyframes switch the seed outright (plus the
# Seed_March rule) instead of A1111's subseed noise blend; the
# default no-keyframe seed walk is the tab's Seed_March rule,
# not the script's interpolated ramp; frames before the first
# prompt keyframe use that keyframe's prompt (the script leaves
# those rows empty); the paste-props overlay system, the
# rendertext text overlays, external video/image frame sources
# and the colour-correction keyframes are recognised in the
# keyframes box, counted, and reported -- not applied.
# ------------------------------------------------------------------

def _anim_addnoise(img, percent):
    """The script's addnoise, verbatim in spirit: draw
    int(50 * percent) random coloured ellipses, black outline,
    sizes up to int(50 * percent). It looks crude on its own;
    blurred, it makes the background zoom_at2 pastes onto, and
    at low strengths it feeds img2img fresh detail."""
    from PIL import ImageDraw
    w2, h2 = img.size
    draw = ImageDraw.Draw(img)
    for _ in range(int(50 * float(percent))):
        x2 = random.randint(0, w2)
        y2 = random.randint(0, h2)
        s2 = random.randint(0, int(50 * float(percent)))
        draw.ellipse(
            (x2, y2, x2 + s2, y2 + s2),
            fill=(random.randint(0, 255), random.randint(0, 255),
                  random.randint(0, 255)),
            outline=(0, 0, 0))
    return img


def _anim_zoom_at2(img, rot, x, y, zoom):
    """The script's zoom_at2: scale the frame by zoom, rotate it
    by rot degrees, and paste it centred (plus the x/y offsets)
    onto a background made from a noise-added, blurred copy of
    the same frame. One guard the script lacks: the scaled size
    is clamped to at least 1 pixel so extreme zoom-outs cannot
    crash a long run."""
    from PIL import Image, ImageFilter
    w, h = img.size
    img2 = img.resize(
        (max(1, int(w * zoom)), max(1, int(h * zoom))),
        Image.LANCZOS)
    padding = 2
    resimg = _anim_addnoise(img.copy(), 0.75).resize(
        (w + padding * 2, h + padding * 2), Image.LANCZOS).filter(
            ImageFilter.GaussianBlur(5)).crop(
                (padding, padding, w + padding, h + padding))
    resimg.paste(
        img2.rotate(rot),
        (int((w - img2.size[0]) / 2 + x),
         int((h - img2.size[1]) / 2 + y)))
    return resimg


def _anim_morph(img1, img2, count):
    """The script's morph: `count` crossfade frames between two
    images, evenly spaced (fractions 1/(count+1) .. count/(count+1))."""
    import numpy as np
    from PIL import Image
    arr1 = np.array(img1)
    diff = np.array(img2).astype("int16") - arr1.astype("int16")
    frames = []
    for step in range(1, count + 1):
        frames.append(Image.fromarray(
            (arr1 + diff * (step / (count + 1))).astype("uint8"),
            "RGB"))
    return frames


def _png_prompt_info(path):
    """Read A1111-style generation info from a PNG (the script's
    get_pnginfo, via piexif / the PNG text chunks): returns
    (found, positive, negative, seed). Used by the
    prompt_from_png keyframe command."""
    from PIL import Image
    try:
        image = Image.open(path)
        items = dict(image.info or {})
    except Exception:  # noqa: BLE001 -- a bad path just misses
        return False, "", "", None
    geninfo = ""
    if "exif" in items:
        try:
            import piexif
            import piexif.helper
            exif = piexif.load(items["exif"])
            comment = (exif or {}).get("Exif", {}).get(
                piexif.ExifIFD.UserComment, b"")
            try:
                geninfo = piexif.helper.UserComment.load(comment)
            except ValueError:
                geninfo = comment.decode("utf8", errors="ignore")
        except Exception:  # noqa: BLE001 -- fall through to text
            geninfo = ""
    if items.get("parameters"):
        geninfo = items["parameters"]
    if not geninfo or "\nSteps:" not in geninfo:
        return False, "", "", None
    if "\nNegative prompt:" in geninfo:
        pos = geninfo[:geninfo.find("\nNegative prompt:")]
        neg = geninfo[geninfo.find("\nNegative prompt:") + 18:
                      geninfo.rfind("\nSteps:")]
    else:
        pos = geninfo[:geninfo.find("\nSteps:")]
        neg = ""
    seed = None
    params = geninfo[geninfo.rfind("\nSteps:") + 1:]
    spot = params.find("Seed: ")
    if spot >= 0:
        try:
            seed = int(params[spot + 6:params.find(",", spot + 6)])
        except ValueError:
            seed = None
    return True, pos.strip(), neg.strip(), seed


# Keyframe commands this port recognises but does not apply (the
# two named exclusions, plus the script features with no clean
# Diffusers equivalent). They are counted and reported, never
# silently dropped.
_ANIM_UNPORTED = (
    "prop", "set_stamp", "clear_stamp",      # paste-props system
    "set_text", "clear_text",                # rendertext overlays
    "source",                                # video/image sources
    "subseed",                               # A1111 subseed blend
    "col_set", "col_clear",                  # colour corrections
)


def _clean_field(text):
    """The script's keyframe field cleanup: strip whitespace,
    then stray commas, then whitespace again."""
    return (text or "").strip().strip(",").strip()


def _parse_keyframes(text, fps, frame_count, tmpl_pos, tmpl_neg,
                     initials):
    """Parse the Keyframes box the way the script's run() does.

    initials carries the tab's starting values (denoise, noise,
    x_shift, y_shift, zoom, rotation -- per-second forms; this
    function converts them to per-frame points at frame 0).

    Returns a dict: prompts (sorted [(frame, pos, neg)]),
    seed_points (sorted [(frame, seed)]), model_keyframes
    ({frame: name}), motion (per-column {frame: per-frame value}
    point dicts), the (possibly replaced) templates, a count of
    recognised-but-unported and unknown commands, a bad-line
    count, and notes."""
    motion = {
        "denoise": {0: float(initials["denoise"])},
        "noise": {0: float(initials["noise"])},
        "x_shift": {0: float(initials["x_shift"]) / fps},
        "y_shift": {0: float(initials["y_shift"]) / fps},
        "zoom": {0: float(initials["zoom"]) ** (1.0 / fps)},
        "rotation": {0: float(initials["rotation"]) / fps},
    }
    prompts = []
    seed_points = []
    model_keyframes = {}
    ignored = {}
    bad_lines = 0
    notes = []
    for raw_line in (text or "").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split("|")
        if len(parts) < 2:
            bad_lines += 1
            continue
        try:
            frame = int(float(parts[0]) * fps)
        except (TypeError, ValueError):
            bad_lines += 1
            continue
        frame = max(0, min(frame_count, frame))
        command = parts[1].lower().strip()
        try:
            if command == "transform" and len(parts) == 6:
                motion["x_shift"][frame] = float(parts[3]) / fps
                motion["y_shift"][frame] = float(parts[4]) / fps
                motion["zoom"][frame] = float(parts[2]) ** (1.0 / fps)
                motion["rotation"][frame] = float(parts[5]) / fps
            elif command == "denoise" and len(parts) == 3:
                motion["denoise"][frame] = float(parts[2])
            elif command == "noise" and len(parts) == 3:
                motion["noise"][frame] = float(parts[2])
            elif command == "seed" and len(parts) == 3:
                seed_points.append((frame, int(parts[2])))
            elif command == "prompt" and len(parts) == 4:
                prompts.append((frame, _clean_field(parts[2]),
                                _clean_field(parts[3])))
            elif command == "template" and len(parts) == 4:
                # The script applies templates globally after
                # parsing, so the LAST template keyframe wins
                # for the whole run -- mirrored here.
                tmpl_pos = _clean_field(parts[2])
                tmpl_neg = _clean_field(parts[3])
            elif command == "prompt_from_png" and len(parts) == 3:
                found, pos, neg, seed = _png_prompt_info(
                    _clean_field(parts[2]))
                if found:
                    prompts.append((frame, pos, neg))
                    if seed is not None:
                        seed_points.append((frame, seed))
                else:
                    notes.append(
                        "prompt_from_png at {}s found no "
                        "generation info in {}".format(
                            parts[0].strip(), _clean_field(parts[2])))
            elif command == "model" and len(parts) == 3:
                model_keyframes[frame] = parts[2].strip()
            elif command in _ANIM_UNPORTED:
                ignored[command] = ignored.get(command, 0) + 1
            else:
                ignored[command] = ignored.get(command, 0) + 1
        except (TypeError, ValueError):
            bad_lines += 1
    prompts.sort(key=lambda item: item[0])
    seed_points.sort(key=lambda item: item[0])
    return {
        "prompts": prompts,
        "seed_points": seed_points,
        "model_keyframes": model_keyframes,
        "motion": motion,
        "tmpl_pos": tmpl_pos,
        "tmpl_neg": tmpl_neg,
        "ignored": ignored,
        "bad_lines": bad_lines,
        "notes": notes,
    }


def _interp_points(count, points):
    """A per-frame series from {frame: value} points: linear
    interpolation between points, edge values held outside
    (the script's limit_direction='both')."""
    frames = sorted(points)
    series = []
    for i in range(count):
        if i <= frames[0]:
            series.append(float(points[frames[0]]))
        elif i >= frames[-1]:
            series.append(float(points[frames[-1]]))
        else:
            for j in range(len(frames) - 1):
                lo, hi = frames[j], frames[j + 1]
                if lo <= i <= hi:
                    if hi == lo:
                        series.append(float(points[lo]))
                    else:
                        frac = (i - lo) / float(hi - lo)
                        series.append(
                            float(points[lo]) * (1.0 - frac)
                            + float(points[hi]) * frac)
                    break
    return series


def _build_frame_table(frame_count, motion, prompts, seed_points,
                       initial_seed):
    """The per-frame table, following the script's dataframe
    construction (see the Animation section header):

    rows 0..frame_count (the last row is an interpolation anchor,
    never rendered). Prompt sides step at keyframes (with the
    frames before the first keyframe back-filled from it -- the
    script leaves those empty), the prompt weight is 1 at each
    keyframe, 0 at the frame before the next, and 0 at the last
    row. Seeds step at seed keyframes."""
    rows = frame_count + 1
    table = {}
    for name in ("denoise", "noise", "x_shift", "y_shift", "zoom",
                 "rotation"):
        table[name] = _interp_points(rows, motion[name])

    sides_points = {}
    weight_points = {}
    if not prompts:
        sides_points[0] = ("", "", "", "")
        weight_points[0] = 1.0
    elif len(prompts) == 1:
        # A single prompt keyframe: the script's pairing loop
        # never runs for it and its prompts come out empty; the
        # port uses the prompt on both sides instead (no visible
        # crossfade, the prompt simply holds).
        only = prompts[0]
        sides_points[only[0]] = (only[1], only[2], only[1], only[2])
        weight_points[only[0]] = 1.0
    else:
        for x in range(len(prompts) - 1):
            frame_x = prompts[x][0]
            sides_points[frame_x] = (
                prompts[x][1], prompts[x][2],
                prompts[x + 1][1], prompts[x + 1][2])
            weight_points[frame_x] = 1.0
            if x > 0:
                weight_points[frame_x - 1] = 0.0
    weight_points[frame_count] = 0.0
    table["weight"] = _interp_points(rows, weight_points)

    side_frames = sorted(sides_points)
    sides = []
    for i in range(rows):
        chosen = side_frames[0]
        for frame_x in side_frames:
            if frame_x <= i:
                chosen = frame_x
        sides.append(sides_points[chosen])
    table["pos1"] = [side[0] for side in sides]
    table["neg1"] = [side[1] for side in sides]
    table["pos2"] = [side[2] for side in sides]
    table["neg2"] = [side[3] for side in sides]

    seed_base = []
    seed_since = []
    current_seed = int(initial_seed)
    current_since = 0
    point_map = dict(seed_points)
    for i in range(rows):
        if i in point_map:
            current_seed = point_map[i]
            current_since = i
        seed_base.append(current_seed)
        seed_since.append(current_since)
    table["seed_base"] = seed_base
    table["seed_since"] = seed_since
    return table


def _join_prompt(template, part):
    """template + ", " + part, skipping empty pieces (the script
    concatenates with a leading comma when the template is empty;
    the conditioning is the same either way)."""
    pieces = [piece.strip() for piece in (template, part)
              if piece and piece.strip()]
    return ", ".join(pieces)


def _anim_conditioning(pipe, row, tmpl_pos, tmpl_neg, clip_skip):
    """One animation frame's conditioning: the template is
    prepended to both keyframe sides, then the sides crossfade
    by the frame's weight -- w * E(side 1) + (1 - w) * E(side 2)
    -- for the negative side too. Weighting syntax applies
    inside each side, as everywhere in this app."""
    weight = float(row["weight"])
    pos1 = _join_prompt(tmpl_pos, row["pos1"])
    pos2 = _join_prompt(tmpl_pos, row["pos2"])
    neg1 = _join_prompt(tmpl_neg, row["neg1"])
    neg2 = _join_prompt(tmpl_neg, row["neg2"])
    if pos1 == pos2:
        pos, ok_pos = _encode_text(pipe, pos1, clip_skip)
    else:
        embeds1, ok1 = _encode_text(pipe, pos1, clip_skip)
        embeds2, ok2 = _encode_text(pipe, pos2, clip_skip)
        pos = weight * embeds1 + (1.0 - weight) * embeds2
        ok_pos = ok1 and ok2
    if neg1 == neg2:
        neg, ok_neg = _encode_text(pipe, neg1, clip_skip)
    else:
        embeds1, ok1 = _encode_text(pipe, neg1, clip_skip)
        embeds2, ok2 = _encode_text(pipe, neg2, clip_skip)
        neg = weight * embeds1 + (1.0 - weight) * embeds2
        ok_neg = ok1 and ok2
    return pos, neg, (ok_pos and ok_neg), pos1, pos2


def _fps_text(value):
    """'15' rather than '15.0' for ffmpeg arguments."""
    return ("%f" % float(value)).rstrip("0").rstrip(".")


def _find_ffmpeg():
    """Locate an ffmpeg executable, best source first:

    1. a bundled copy at tools/ffmpeg.exe next to the app (a
       user can drop one in; it wins when present);
    2. the static ffmpeg binary the imageio-ffmpeg package
       carries (it is in requirements.txt for exactly this, so
       a normal install always has a movie assembler);
    3. whatever shutil.which finds on PATH.

    None when all three come up empty -- the caller then keeps
    the frames and says so plainly, exactly like the original
    script behaved on a machine without ffmpeg."""
    bundled = os.path.join(APP_DIR, "tools", "ffmpeg.exe")
    if os.path.isfile(bundled):
        return bundled
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and os.path.isfile(candidate):
            return candidate
    except Exception:  # noqa: BLE001 -- package missing or
        pass               # broken: fall through to PATH
    return shutil.which("ffmpeg")


def _video_command(fmt, folder, first_frame, fps_value, out_path):
    """The ffmpeg command for one format, built as an argument
    list (never a shell string), mirroring the script's
    make_gif / make_mp4 / make_webm flags. -start_number aims
    the %06d pattern at this run's first frame in the shared
    batch folder."""
    pattern = os.path.join(folder, "%06d.png")
    start = ["-start_number", str(int(first_frame))]
    fps = _fps_text(fps_value)
    if fmt == "gif":
        return (["ffmpeg", "-y", "-r", fps] + start
                + ["-i", pattern, out_path])
    if fmt == "webm":
        return (["ffmpeg", "-y", "-framerate", fps] + start
                + ["-i", pattern, "-crf", "50", "-preset",
                   "veryfast", out_path])
    return (["ffmpeg", "-y", "-r", fps] + start
            + ["-i", pattern, "-c:v", "libx264", "-vf",
               "fps=" + fps, "-pix_fmt", "yuv420p", "-crf", "17",
               "-preset", "veryfast", out_path])


def _resolve_checkpoint_name(name):
    """Resolve a model keyframe's name to a checkpoint dropdown
    value: exact match, match without extension, or the built-in
    model under its common names. None when nothing matches."""
    entries = _list_checkpoints()
    wanted = (name or "").strip()
    if wanted in entries:
        return wanted
    lowered = wanted.lower()
    if lowered in ("stable-diffusion-v1-5", "sd 1.5", "sd1.5",
                   "built-in", "builtin"):
        return BUILTIN_LABEL
    for entry in entries:
        stem = os.path.splitext(entry)[0]
        if stem == wanted or stem.lower() == lowered:
            return entry
        if entry.lower() == lowered:
            return entry
    return None


def generate_animation(tmpl_pos, tmpl_neg, keyframes_text,
                       total_time, fps_value, smoothing,
                       denoise, seed_march, zoom_factor,
                       x_shift, y_shift, rotation,
                       add_noise, noise_strength,
                       vid_gif, vid_mp4, vid_webm,
                       width, height, steps, cfg, sampler,
                       seed, batch_name,
                       clip_skip, checkpoint, offload,
                       lora1, lora1_weight, lora2, lora2_weight,
                       last_folder, history):
    """Run an animation (a generator: yields after every frame,
    so the page updates live and Stop cancels between frames --
    the frames already saved stay saved).

    Frame 1 is txt2img. Every later frame: the previous frame,
    transformed by the frame's motion values, noised when
    Add_Noise is on, then img2img at the frame's denoise."""
    history = history or []
    offload = bool(offload)
    seed_march = bool(seed_march)
    add_noise = bool(add_noise)
    try:
        total_time = float(total_time)
        fps_f = float(fps_value)
        zoom_factor = float(zoom_factor)
        x_shift = float(x_shift)
        y_shift = float(y_shift)
        rotation = float(rotation)
    except (TypeError, ValueError):
        yield from _early_anim(
            "Animation length, framerate, zoom, shifts and "
            "rotation must all be numbers.", last_folder, history)
        return
    if fps_f <= 0 or total_time <= 0:
        yield from _early_anim(
            "Animation length and framerate must be above zero.",
            last_folder, history)
        return
    frame_count = int(fps_f * total_time)
    if frame_count < 1:
        yield from _early_anim(
            "That length and framerate give no frames at all.",
            last_folder, history)
        return
    capped = False
    if frame_count > 3600:
        frame_count = 3600
        capped = True
    smoothing = max(0, min(32, int(smoothing)))
    steps = max(1, min(150, int(steps)))
    width = _round8(max(64, min(1024, int(width))))
    height = _round8(max(64, min(1024, int(height))))
    base_seed = int(seed) if seed is not None else -1
    if base_seed < 0:
        base_seed = random.randint(0, 2**31 - 1)

    parsed = _parse_keyframes(
        keyframes_text, fps_f, frame_count,
        (tmpl_pos or "").strip(), (tmpl_neg or "").strip(),
        {"denoise": denoise, "noise": noise_strength,
         "x_shift": x_shift, "y_shift": y_shift,
         "zoom": zoom_factor, "rotation": rotation})
    table = _build_frame_table(
        frame_count, parsed["motion"], parsed["prompts"],
        parsed["seed_points"], base_seed)
    tmpl_pos = parsed["tmpl_pos"]
    tmpl_neg = parsed["tmpl_neg"]
    final_fps = fps_f * (1 + smoothing)

    base = _safe_batch_name(batch_name)
    folder = _batch_folder(base)
    os.makedirs(folder, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # The frame table, dumped like the script dumps its dataframe
    # to CSV -- the full keyframe maths, auditable after the run.
    try:
        import csv
        csv_path = os.path.join(
            folder, "animation_{}_frames.csv".format(stamp))
        with open(csv_path, "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["frame", "pos1", "neg1", "pos2", "neg2",
                 "prompt_weight", "denoise", "noise", "x_shift",
                 "y_shift", "zoom", "rotation", "seed_base"])
            for i in range(frame_count):
                writer.writerow([
                    i,
                    table["pos1"][i], table["neg1"][i],
                    table["pos2"][i], table["neg2"][i],
                    "{:.4f}".format(table["weight"][i]),
                    "{:.4f}".format(table["denoise"][i]),
                    "{:.4f}".format(table["noise"][i]),
                    "{:.4f}".format(table["x_shift"][i]),
                    "{:.4f}".format(table["y_shift"][i]),
                    "{:.6f}".format(table["zoom"][i]),
                    "{:.4f}".format(table["rotation"][i]),
                    table["seed_base"][i]])
    except OSError:
        pass

    try:
        pipe = _load_base(checkpoint, offload)
    except RuntimeError as exc:
        yield from _early_anim(str(exc), last_folder, history)
        return
    try:
        lora_text = _reconcile_loras(
            pipe, lora1, lora1_weight, lora2, lora2_weight)
    except RuntimeError as exc:
        yield from _early_anim(str(exc), last_folder, history)
        return
    current_checkpoint = checkpoint or BUILTIN_LABEL

    entries = _common_entries(
        "animation", checkpoint, sampler, clip_skip, True)
    entries += [
        "size: {}x{}".format(width, height),
        "steps: {}".format(steps),
        "guidance_scale: {}".format(float(cfg)),
        "offload: {}".format(offload),
        "loras: {}".format(lora_text),
        "total_time_s: {}".format(total_time),
        "framerate: {}".format(fps_f),
        "frame_count: {}".format(frame_count),
        "smoothing_frames: {}".format(smoothing),
        "final_fps: {}".format(final_fps),
        "denoising_strength: {}".format(float(denoise)),
        "seed: {}".format(base_seed),
        "seed_march: {}".format(seed_march),
        "zoom_factor_per_s: {}".format(zoom_factor),
        "x_shift_per_s: {}".format(x_shift),
        "y_shift_per_s: {}".format(y_shift),
        "rotation_per_s: {}".format(rotation),
        "add_noise: {}".format(add_noise),
        "noise_strength: {}".format(float(noise_strength)),
        "template_positive: {}".format(tmpl_pos),
        "template_negative: {}".format(tmpl_neg),
        "video_gif: {}".format(bool(vid_gif)),
        "video_mp4: {}".format(bool(vid_mp4)),
        "video_webm: {}".format(bool(vid_webm)),
    ]
    if capped:
        entries.append(
            "note: frame count capped at 3600 for safety")
    if parsed["ignored"]:
        entries.append(
            "unported_keyframes: {}".format(
                ", ".join("{} x{}".format(name, count)
                          for name, count in
                          sorted(parsed["ignored"].items()))))
    if parsed["bad_lines"]:
        entries.append(
            "bad_keyframe_lines: {}".format(parsed["bad_lines"]))
    for note in parsed["notes"]:
        entries.append("note: {}".format(note))
    for line in (keyframes_text or "").splitlines():
        if line.strip():
            entries.append("keyframe: {}".format(line))
    first_frame = _next_frame_number(folder)
    _append_settings(
        folder,
        "Run {}, animation ({} frames planned)".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_count),
        entries)

    x_cumulative = 0.0
    y_cumulative = 0.0
    prev_image = None
    last_major = None
    file_no = first_frame
    saved = 0
    weighting_ok = True
    sample_gallery = []
    stop_note = ""
    for frame_no in range(frame_count):
        row = {name: table[name][frame_no] for name in table}
        try:
            if frame_no in parsed["model_keyframes"]:
                wanted = parsed["model_keyframes"][frame_no]
                resolved = _resolve_checkpoint_name(wanted)
                if resolved is None:
                    stop_note = (
                        "Stopped at frame {} of {}: unknown "
                        "checkpoint in model keyframe: {}".format(
                            frame_no + 1, frame_count, wanted))
                    break
                if resolved != current_checkpoint:
                    pipe = _load_base(resolved, offload)
                    lora_text = _reconcile_loras(
                        pipe, lora1, lora1_weight,
                        lora2, lora2_weight)
                    current_checkpoint = resolved
                    _append_lines(folder, [
                        "model switch at frame {}: {}".format(
                            frame_no + 1, resolved)])
            frame_seed = row["seed_base"] + (
                (frame_no - row["seed_since"]) if seed_march else 0)
            pos, neg, ok, side1, side2 = _anim_conditioning(
                pipe, row, tmpl_pos, tmpl_neg, clip_skip)
            weighting_ok = weighting_ok and ok
            generator = _make_generator(frame_seed)
            if frame_no == 0:
                work = _variant_pipe("txt2img", offload)
                work.scheduler = _make_scheduler(sampler)
                image = work(
                    prompt_embeds=pos,
                    negative_prompt_embeds=neg,
                    height=height, width=width,
                    num_inference_steps=steps,
                    guidance_scale=float(cfg),
                    generator=generator).images[0]
            else:
                init = prev_image
                x_cumulative += row["x_shift"]
                y_cumulative += row["y_shift"]
                init = _anim_zoom_at2(
                    init, row["rotation"],
                    int(x_cumulative), int(y_cumulative),
                    row["zoom"])
                x_cumulative -= int(x_cumulative)
                y_cumulative -= int(y_cumulative)
                if add_noise:
                    init = _anim_addnoise(init, row["noise"])
                work = _variant_pipe("img2img", offload)
                work.scheduler = _make_scheduler(sampler)
                image = work(
                    prompt_embeds=pos,
                    negative_prompt_embeds=neg,
                    image=init,
                    strength=float(row["denoise"]),
                    num_inference_steps=steps,
                    guidance_scale=float(cfg),
                    generator=generator).images[0]
        except Exception as exc:  # noqa: BLE001 -- frames stay saved
            stop_note = "Stopped at frame {} of {}: {}".format(
                frame_no + 1, frame_count, exc)
            break
        # Smoothed crossfade frames go in BEFORE this frame, the
        # script's order: morph from the last major frame to this
        # one, then the major frame itself.
        if frame_no > 0 and smoothing > 0:
            for morph_img in _anim_morph(
                    last_major, image, smoothing):
                morph_img.save(os.path.join(
                    folder, "{:06d}.png".format(file_no)))
                file_no += 1
                saved += 1
        image.save(os.path.join(
            folder, "{:06d}.png".format(file_no)))
        file_no += 1
        saved += 1
        last_major = image
        prev_image = image
        if int(fps_f) > 0 and frame_no % int(fps_f) == 0:
            sample_gallery.append(image)
        if side1 == side2:
            prompt_line = side1
        else:
            prompt_line = "{} <-> {} @ {:.2f}".format(
                side1, side2, row["weight"])
        _append_lines(folder, [
            "anim_frame {} (file {:06d}): seed {}, denoise "
            "{:.3f}, prompt: {}".format(
                frame_no + 1, file_no - 1, frame_seed,
                row["denoise"], prompt_line)])
        yield ([image],
               "Animation frame {} of {} saved ({} files in "
               "{}).".format(frame_no + 1, frame_count, saved,
                             folder),
               _history_update(history), folder, history, None)
        _free_memory()

    trailer = []
    if not weighting_ok:
        trailer.append(
            "prompt_weighting: plain -- compel unavailable this run")
    if stop_note:
        trailer.append("stopped: {}".format(stop_note))
    video_notes = []
    made = []
    wanted_formats = []
    if vid_gif:
        wanted_formats.append("gif")
    if vid_mp4:
        wanted_formats.append("mp4")
    if vid_webm:
        wanted_formats.append("webm")
    if wanted_formats and saved > 0:
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            video_notes.append(
                "no ffmpeg could be found (no bundled copy, "
                "the imageio-ffmpeg package unavailable, and "
                "nothing on PATH), so no video was assembled "
                "-- the frames are all in the batch folder.")
        else:
            for fmt in wanted_formats:
                out_path = os.path.join(
                    folder, "animation_{}.{}".format(stamp, fmt))
                cmd = _video_command(
                    fmt, folder, first_frame, final_fps, out_path)
                cmd[0] = ffmpeg
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, timeout=1800)
                    if result.returncode == 0 and \
                            os.path.exists(out_path):
                        made.append(out_path)
                    else:
                        tail = (result.stderr or b"").decode(
                            "utf-8", errors="ignore").strip()
                        video_notes.append(
                            "{} assembly failed (ffmpeg exit {}): "
                            "{}".format(fmt, result.returncode,
                                        tail.splitlines()[-1]
                                        if tail else "no output"))
                except Exception as exc:  # noqa: BLE001
                    video_notes.append(
                        "{} assembly failed: {}".format(fmt, exc))
    for note in video_notes:
        trailer.append("video: {}".format(note))
    for path in made:
        trailer.append("video_made: {}".format(path))
    trailer.append("frames_saved: {}".format(saved))
    _append_lines(folder, trailer)

    # The Movie player gets one file: MP4 when one was made,
    # else WEBM. A GIF on its own does not play in it -- the
    # GIF (and all the frames) stay in the batch folder.
    movie = None
    for ext in (".mp4", ".webm"):
        for path in made:
            if path.lower().endswith(ext):
                movie = path
                break
        if movie:
            break

    if tmpl_pos:
        history = _history_add(history, {
            "prompt": tmpl_pos,
            "negative": tmpl_neg,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": float(cfg),
            "sampler": sampler,
            "seed": base_seed,
        })
    if stop_note:
        status = "{} Frames saved: {} in {}.".format(
            stop_note, saved, folder)
    else:
        status = "Done: {} animation frames in {}.".format(
            saved, folder)
    if made:
        status += " Video: {}.".format(
            ", ".join(os.path.basename(p) for p in made))
    if made and movie is None:
        status += (" The GIF is in the batch folder -- GIFs "
                   "do not play in the Movie player.")
    for note in video_notes:
        status += " " + note
    if capped:
        status += " Note: the frame count was capped at 3600."
    yield (sample_gallery, status, _history_update(history),
           folder, history, movie)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

_SIZE_PRESETS = {
    "Square 512x512": (512, 512),
    "Portrait 512x768": (512, 768),
    "Landscape 768x512": (768, 512),
}


def _apply_preset(preset):
    """Size preset -> (width, height) slider values. 'Custom'
    leaves the sliders alone."""
    import gradio as gr
    if preset in _SIZE_PRESETS:
        return _SIZE_PRESETS[preset]
    return gr.update(), gr.update()


def _refresh_checkpoints(current):
    import gradio as gr
    choices = _list_checkpoints()
    value = current if current in choices else BUILTIN_LABEL
    return gr.update(choices=choices, value=value)


def _refresh_loras(current1, current2):
    import gradio as gr
    choices = _list_loras()
    value1 = current1 if current1 in choices else "None"
    value2 = current2 if current2 in choices else "None"
    return (gr.update(choices=choices, value=value1),
            gr.update(choices=choices, value=value2))


def build_ui():
    """The Gradio page: a global model section (A1111-style
    checkpoint picker on top), then the four tabs."""
    import gradio as gr

    with gr.Blocks(title="SD 1.5 UI") as demo:
        gr.Markdown(
            "# SD 1.5 UI\n"
            "Stable Diffusion 1.5 with the classic A1111 / Forge "
            "toolkit -- checkpoint picker, prompt weighting, "
            "blending, LoRAs, hires fix, ControlNet -- plus the "
            "house queue, dynamics and batch folders. First "
            "Generate downloads the base model (about 4 GB).")
        with gr.Row():
            checkpoint_dd = gr.Dropdown(
                label="Checkpoint", choices=_list_checkpoints(),
                value=BUILTIN_LABEL, scale=3)
            ckpt_refresh_btn = gr.Button("Refresh list", scale=1)
            clip_skip_radio = gr.Radio(
                label="Clip skip", choices=[1, 2], value=1,
                scale=1)
            offload_box = gr.Checkbox(
                label="CPU offload (small cards only -- SD 1.5 "
                      "fits a 12 GB card without it)",
                value=False, scale=2)
        with gr.Accordion(
                "LoRAs -- .safetensors files in the loras folder, "
                "applies to every tab", open=False):
            with gr.Row():
                lora1_dd = gr.Dropdown(
                    label="LoRA 1", choices=_list_loras(),
                    value="None")
                lora1_w = gr.Slider(
                    label="Weight 1", minimum=0.0, maximum=1.5,
                    step=0.05, value=0.8)
                lora2_dd = gr.Dropdown(
                    label="LoRA 2", choices=_list_loras(),
                    value="None")
                lora2_w = gr.Slider(
                    label="Weight 2", minimum=0.0, maximum=1.5,
                    step=0.05, value=0.8)
                lora_refresh_btn = gr.Button("Refresh list")
        lora_inputs = [lora1_dd, lora1_w, lora2_dd, lora2_w]
        global_inputs = [clip_skip_radio, checkpoint_dd, offload_box]

        with gr.Tabs():
            with gr.Tab("Text to image"):
                prompt_box = gr.Textbox(
                    label="Prompt", value=DEFAULT_PROMPT, lines=3)
                negative_box = gr.Textbox(
                    label="Negative prompt", value=DEFAULT_NEGATIVE,
                    lines=2)
                queue_box = gr.Textbox(
                    label="Prompt queue (one prompt per line -- "
                          "when this has lines, Generate runs the "
                          "queue instead of the prompt box; all "
                          "lines share one batch folder)",
                    lines=4)
                with gr.Row():
                    preset_dd = gr.Dropdown(
                        label="Size preset",
                        choices=list(_SIZE_PRESETS) + ["Custom"],
                        value="Square 512x512")
                    width_box = gr.Slider(
                        label="Width", minimum=256, maximum=1024,
                        step=64, value=512)
                    height_box = gr.Slider(
                        label="Height", minimum=256, maximum=1024,
                        step=64, value=512)
                with gr.Row():
                    steps_box = gr.Slider(
                        label="Steps", minimum=1, maximum=60,
                        step=1, value=25)
                    cfg_box = gr.Slider(
                        label="CFG scale", minimum=1, maximum=20,
                        step=0.5, value=7)
                    sampler_dd = gr.Dropdown(
                        label="Sampler", choices=SAMPLER_NAMES,
                        value="Euler a")
                with gr.Row():
                    seed_box = gr.Number(
                        label="Seed (-1 = random)", value=-1,
                        precision=0)
                    count_box = gr.Slider(
                        label="Images per Generate", minimum=1,
                        maximum=10, step=1, value=1)
                    batch_box = gr.Textbox(
                        label="Batch name (one folder per name)",
                        value="sd15-pictures")
                with gr.Accordion("Prompt blending", open=False):
                    gr.Markdown(
                        "A second prompt mixed into the first at "
                        "the embedding level. Leave the blend "
                        "prompt empty for no blending. Weight 0 "
                        "= main prompt only, 1 = blend prompt "
                        "only. Prompt weighting works inside "
                        "both prompts.")
                    blend_prompt_box = gr.Textbox(
                        label="Blend prompt", lines=2)
                    blend_weight_box = gr.Slider(
                        label="Blend weight", minimum=0.0,
                        maximum=1.0, step=0.05, value=0.5)
                with gr.Accordion("Hires fix", open=False):
                    hires_enabled_box = gr.Checkbox(
                        label="Enable hires fix", value=False)
                    with gr.Row():
                        hires_factor_box = gr.Radio(
                            label="Upscale factor",
                            choices=["1.5", "2.0"], value="2.0")
                        hires_denoise_box = gr.Slider(
                            label="Second-pass denoise",
                            minimum=0.0, maximum=1.0, step=0.01,
                            value=0.35)
                with gr.Accordion("ControlNet", open=False):
                    cn_enabled_box = gr.Checkbox(
                        label="Enable ControlNet", value=False)
                    with gr.Row():
                        cn_model_radio = gr.Radio(
                            label="ControlNet model",
                            choices=["Canny", "Depth", "OpenPose"],
                            value="Canny")
                        cn_weight_box = gr.Slider(
                            label="Control weight", minimum=0.0,
                            maximum=2.0, step=0.05, value=0.8)
                    cn_image_box = gr.Image(
                        label="Control image (Canny is extracted "
                              "from your image automatically; "
                              "Depth and OpenPose take an already-"
                              "preprocessed map / skeleton image)",
                        type="pil")
                with gr.Row():
                    random_btn = gr.Button("Random prompt")
                    history_dd = gr.Dropdown(
                        label="History (last 10 -- pick one to "
                              "reload its prompt and settings)",
                        choices=[])
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate", variant="primary")
                    stop_btn = gr.Button("Stop")
            with gr.Tab("Image to image"):
                i2i_image = gr.Image(label="Image", type="pil")
                i2i_prompt = gr.Textbox(
                    label="Prompt", value=DEFAULT_PROMPT, lines=3)
                i2i_negative = gr.Textbox(
                    label="Negative prompt",
                    value=DEFAULT_NEGATIVE, lines=2)
                with gr.Row():
                    i2i_denoise = gr.Slider(
                        label="Denoising strength", minimum=0.0,
                        maximum=1.0, step=0.01, value=0.6)
                    i2i_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=60,
                        step=1, value=25)
                    i2i_cfg = gr.Slider(
                        label="CFG scale", minimum=1, maximum=20,
                        step=0.5, value=7)
                    i2i_sampler = gr.Dropdown(
                        label="Sampler", choices=SAMPLER_NAMES,
                        value="Euler a")
                with gr.Row():
                    i2i_seed = gr.Number(
                        label="Seed (-1 = random)", value=-1,
                        precision=0)
                    i2i_count = gr.Slider(
                        label="Images per Generate", minimum=1,
                        maximum=10, step=1, value=1)
                    i2i_batch = gr.Textbox(
                        label="Batch name (one folder per name)",
                        value="sd15-img2img")
                with gr.Row():
                    i2i_generate_btn = gr.Button(
                        "Generate", variant="primary")
                    i2i_stop_btn = gr.Button("Stop")
            with gr.Tab("Inpaint"):
                with gr.Row():
                    inp_image = gr.Image(label="Image", type="pil")
                    inp_mask = gr.Image(
                        label="Mask (white = repaint)", type="pil")
                inp_prompt = gr.Textbox(
                    label="Prompt", value=DEFAULT_PROMPT, lines=3)
                inp_negative = gr.Textbox(
                    label="Negative prompt",
                    value=DEFAULT_NEGATIVE, lines=2)
                with gr.Row():
                    inp_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=60,
                        step=1, value=25)
                    inp_cfg = gr.Slider(
                        label="CFG scale", minimum=1, maximum=20,
                        step=0.5, value=7)
                    inp_sampler = gr.Dropdown(
                        label="Sampler", choices=SAMPLER_NAMES,
                        value="Euler a")
                with gr.Row():
                    inp_seed = gr.Number(
                        label="Seed (-1 = random)", value=-1,
                        precision=0)
                    inp_count = gr.Slider(
                        label="Images per Generate", minimum=1,
                        maximum=10, step=1, value=1)
                    inp_batch = gr.Textbox(
                        label="Batch name (one folder per name)",
                        value="sd15-inpaint")
                with gr.Row():
                    inp_generate_btn = gr.Button(
                        "Generate", variant="primary")
                    inp_stop_btn = gr.Button("Stop")
            with gr.Tab("Animation"):
                gr.Markdown(
                    "Animation Script v6.0 by Animator-Anon, "
                    "ported to this app. Frame 1 is text-to-image; "
                    "each later frame is the previous one moved, "
                    "then repainted.\n\n"
                    "**Keyframes**, one per line:\n\n"
                    "`time_s | prompt | positive | negative`\n\n"
                    "`time_s | template | positive | negative`\n\n"
                    "`time_s | prompt_from_png | file_path`\n\n"
                    "`time_s | transform | zoom | x_shift | "
                    "y_shift | rotation`\n\n"
                    "`time_s | seed | new_seed`\n\n"
                    "`time_s | denoise | value`\n\n"
                    "`time_s | noise | value`\n\n"
                    "`time_s | model | checkpoint_name`\n\n"
                    "Prompt keyframes crossfade into each other. "
                    "Props, stamps, text overlays, video/image "
                    "sources, subseed and colour-correction "
                    "keyframes are recognised and logged in "
                    "settings.txt but not applied in this port.")
                anim_tmpl_pos = gr.Textbox(
                    label="Positive Prompts (template, applied "
                          "to every keyframe)",
                    value=DEFAULT_PROMPT, lines=2)
                anim_tmpl_neg = gr.Textbox(
                    label="Negative Prompts (template)",
                    value=DEFAULT_NEGATIVE, lines=2)
                anim_keyframes = gr.Textbox(
                    label="Keyframes:", lines=5, value="")
                with gr.Row():
                    anim_total = gr.Number(
                        label="Total Animation Length (s)",
                        value=10.0)
                    anim_fps = gr.Number(label="Framerate",
                                         value=15)
                    anim_smoothing = gr.Slider(
                        label="Smoothing_Frames", minimum=0,
                        maximum=32, step=1, value=0)
                with gr.Row():
                    anim_denoise = gr.Slider(
                        label="Denoising Strength", minimum=0.0,
                        maximum=1.0, step=0.01, value=0.40)
                    anim_seed_march = gr.Checkbox(
                        label="Seed_March", value=False)
                    anim_add_noise = gr.Checkbox(
                        label="Add_Noise", value=False)
                    anim_noise = gr.Slider(
                        label="Noise Strength", minimum=0.0,
                        maximum=1.0, step=0.01, value=0.10)
                with gr.Row():
                    anim_zoom = gr.Number(
                        label="Zoom Factor (scale/s)", value=1.0)
                    anim_x = gr.Number(
                        label="X Pixel Shift (pixels/s)", value=0)
                    anim_y = gr.Number(
                        label="Y Pixel Shift (pixels/s)", value=0)
                    anim_rotation = gr.Number(
                        label="Rotation (deg/s)", value=0)
                with gr.Row():
                    anim_vid_gif = gr.Checkbox(label="GIF",
                                               value=False)
                    anim_vid_mp4 = gr.Checkbox(label="MP4",
                                               value=False)
                    anim_vid_webm = gr.Checkbox(label="WEBM",
                                                value=True)
                with gr.Row():
                    anim_preset = gr.Dropdown(
                        label="Size preset",
                        choices=list(_SIZE_PRESETS) + ["Custom"],
                        value="Square 512x512")
                    anim_width = gr.Slider(
                        label="Width", minimum=256, maximum=1024,
                        step=64, value=512)
                    anim_height = gr.Slider(
                        label="Height", minimum=256, maximum=1024,
                        step=64, value=512)
                with gr.Row():
                    anim_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=60,
                        step=1, value=25)
                    anim_cfg = gr.Slider(
                        label="CFG scale", minimum=1, maximum=20,
                        step=0.5, value=7)
                    anim_sampler = gr.Dropdown(
                        label="Sampler", choices=SAMPLER_NAMES,
                        value="Euler a")
                with gr.Row():
                    anim_seed = gr.Number(
                        label="Seed (-1 = random)", value=-1,
                        precision=0)
                    anim_batch = gr.Textbox(
                        label="Batch name (one folder per name)",
                        value="sd15-animation")
                with gr.Row():
                    anim_generate_btn = gr.Button(
                        "Generate animation", variant="primary")
                    anim_stop_btn = gr.Button("Stop")
        with gr.Row():
            star_btn = gr.Button("Star batch")
            unload_btn = gr.Button("Unload models")
        status_box = gr.Textbox(label="Status", lines=4)
        gallery = gr.Gallery(label="Pictures", columns=3)
        movie_video = gr.Video(label="Movie")

        last_folder_state = gr.State("")
        history_state = gr.State([])
        outputs = [gallery, status_box, history_dd,
                   last_folder_state, history_state]
        # The Animation event has one more output than the
        # others: the assembled movie, playable in the page.
        anim_outputs = outputs + [movie_video]

        ckpt_refresh_btn.click(
            fn=_refresh_checkpoints, inputs=checkpoint_dd,
            outputs=checkpoint_dd)
        lora_refresh_btn.click(
            fn=_refresh_loras, inputs=[lora1_dd, lora2_dd],
            outputs=[lora1_dd, lora2_dd])
        preset_dd.change(fn=_apply_preset, inputs=preset_dd,
                         outputs=[width_box, height_box])
        anim_preset.change(fn=_apply_preset, inputs=anim_preset,
                           outputs=[anim_width, anim_height])
        random_btn.click(fn=random_prompt, outputs=prompt_box)
        history_dd.change(
            fn=_history_restore,
            inputs=[history_dd, history_state],
            outputs=[prompt_box, negative_box, width_box,
                     height_box, steps_box, cfg_box, sampler_dd,
                     seed_box])

        generate_event = generate_btn.click(
            fn=generate,
            inputs=[prompt_box, negative_box, queue_box,
                    width_box, height_box, steps_box, cfg_box,
                    sampler_dd, clip_skip_radio, seed_box,
                    batch_box, count_box,
                    blend_prompt_box, blend_weight_box,
                    hires_enabled_box, hires_factor_box,
                    hires_denoise_box,
                    cn_enabled_box, cn_model_radio, cn_image_box,
                    cn_weight_box,
                    checkpoint_dd, offload_box] + lora_inputs + [
                    last_folder_state, history_state],
            outputs=outputs)
        stop_btn.click(fn=_noop, cancels=[generate_event])

        i2i_event = i2i_generate_btn.click(
            fn=generate_img2img,
            inputs=[i2i_image, i2i_prompt, i2i_negative,
                    i2i_denoise, i2i_steps, i2i_cfg, i2i_sampler,
                    clip_skip_radio, i2i_seed, i2i_batch,
                    i2i_count, checkpoint_dd, offload_box]
            + lora_inputs + [last_folder_state, history_state],
            outputs=outputs)
        i2i_stop_btn.click(fn=_noop, cancels=[i2i_event])

        inp_event = inp_generate_btn.click(
            fn=generate_inpaint,
            inputs=[inp_image, inp_mask, inp_prompt, inp_negative,
                    inp_steps, inp_cfg, inp_sampler,
                    clip_skip_radio, inp_seed, inp_batch,
                    inp_count, checkpoint_dd, offload_box]
            + lora_inputs + [last_folder_state, history_state],
            outputs=outputs)
        inp_stop_btn.click(fn=_noop, cancels=[inp_event])

        anim_event = anim_generate_btn.click(
            fn=generate_animation,
            inputs=[anim_tmpl_pos, anim_tmpl_neg, anim_keyframes,
                    anim_total, anim_fps, anim_smoothing,
                    anim_denoise, anim_seed_march, anim_zoom,
                    anim_x, anim_y, anim_rotation,
                    anim_add_noise, anim_noise,
                    anim_vid_gif, anim_vid_mp4, anim_vid_webm,
                    anim_width, anim_height, anim_steps, anim_cfg,
                    anim_sampler, anim_seed, anim_batch,
                    clip_skip_radio, checkpoint_dd, offload_box]
            + lora_inputs + [last_folder_state, history_state],
            outputs=anim_outputs)
        anim_stop_btn.click(fn=_noop, cancels=[anim_event])

        star_btn.click(fn=star_batch, inputs=last_folder_state,
                       outputs=status_box)
        unload_btn.click(fn=unload_models, outputs=status_box)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", inbrowser=True)
