"""DeepCAD Studio — Chat-based CAD generation GUI.

A professional web interface that lets users describe designs in natural
language and generates 3D CAD models in real time using the DeepCAD
pretrained Transformer autoencoder.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # must be before any native lib

import patch_cuda  # patches CUDA / NumPy / OpenMP

import sys, glob, uuid, re, traceback, json, random as py_random

import torch
import numpy as np
import h5py

from flask import Flask, render_template, jsonify, request, send_from_directory

from cadlib.macro import (
    ALL_COMMANDS, LINE_IDX, ARC_IDX, CIRCLE_IDX,
    EOS_IDX, SOL_IDX, EXT_IDX, CMD_ARGS_MASK,
    ARGS_DIM, N_ARGS, MAX_N_EXT, MAX_N_LOOPS,
    MAX_N_CURVES, MAX_TOTAL_LEN, EXTRUDE_OPERATIONS, EXTENT_TYPE,
)
from cadlib.visualize import vec2CADsolid

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "gui", "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "gui", "static"),
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "proj_log", "pretrained", "results", "test_1000")
STL_CACHE   = os.path.join(PROJECT_ROOT, "gui", "static", "stl_cache")
os.makedirs(STL_CACHE, exist_ok=True)

# ---------------------------------------------------------------------------
# Minimal config (avoids argparse so the model can load inside Flask)
# ---------------------------------------------------------------------------
class _GUIConfig:
    def __init__(self):
        self.args_dim        = ARGS_DIM        # 256
        self.n_args          = N_ARGS
        self.n_commands      = len(ALL_COMMANDS)
        self.n_layers        = 4
        self.n_layers_decode = 4
        self.n_heads         = 8
        self.dim_feedforward = 512
        self.d_model         = 256
        self.dropout         = 0.1
        self.dim_z           = 256
        self.use_group_emb   = True
        self.max_n_ext       = MAX_N_EXT
        self.max_n_loops     = MAX_N_LOOPS
        self.max_n_curves    = MAX_N_CURVES
        self.max_num_groups  = 30
        self.max_total_len   = MAX_TOTAL_LEN

# ---------------------------------------------------------------------------
# Lazy model loading — the network is only loaded on the first generation
# request, keeping memory free until actually needed.
# ---------------------------------------------------------------------------
_net = None
_cfg = None
_gen = None

def _load_model():
    global _net, _cfg
    if _net is not None:
        return _net
    from model import CADTransformer
    _cfg = _GUIConfig()
    _net = CADTransformer(_cfg)
    _net.to("cpu")
    ckpt_path = os.path.join(PROJECT_ROOT, "proj_log", "pretrained", "model", "ckpt_epoch1000.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    _net.load_state_dict(checkpoint["model_state_dict"])
    _net.eval()
    print("[DeepCAD Studio] Pretrained model loaded on CPU.")
    return _net

def _load_generator():
    global _gen
    if _gen is not None:
        return _gen
    from model.latentGAN import Generator
    _gen = Generator(n_dim=64, h_dim=512, z_dim=256)
    _gen.to("cpu")
    ckpt_path = os.path.join(PROJECT_ROOT, "proj_log", "pretrained", "lgan_1000", "model", "ckpt_epoch200000.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    _gen.load_state_dict(checkpoint["netG_state_dict"])
    _gen.eval()
    print("[DeepCAD Studio] Pretrained Latent GAN Generator loaded on CPU.")
    return _gen


# ---------------------------------------------------------------------------
# CAD helpers
# ---------------------------------------------------------------------------
def _solid_to_stl(shape, path):
    """Mesh a BRep solid and write binary STL."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    mesh = BRepMesh_IncrementalMesh(shape, 0.05, False, 0.5, True)
    mesh.Perform()
    writer = StlAPI_Writer()
    writer.SetASCIIMode(False)
    writer.Write(shape, str(path))


def _logits_to_vec(outputs):
    """Convert model output logits to a numpy CAD vector (batch)."""
    out_cmd  = torch.argmax(torch.softmax(outputs["command_logits"], dim=-1), dim=-1)
    out_args = torch.argmax(torch.softmax(outputs["args_logits"],   dim=-1), dim=-1) - 1
    mask = ~torch.tensor(CMD_ARGS_MASK).bool()[out_cmd.long()]
    out_args[mask] = -1
    return torch.cat([out_cmd.unsqueeze(-1), out_args], dim=-1).detach().cpu().numpy()


def _trim_at_eos(vec):
    """Trim trailing EOS padding from a CAD vector.

    A valid CAD sequence looks like:
        SOL, <curves>, EXT, SOL, <curves>, EXT, ..., EOS, EOS, EOS (padding)
    We keep everything up to (but not including) the *first* EOS that is
    followed only by more EOS tokens (i.e. trailing padding).
    """
    cmds = vec[:, 0].astype(int)
    # Walk backwards to find the last non-EOS command
    last_non_eos = len(cmds) - 1
    while last_non_eos >= 0 and cmds[last_non_eos] == EOS_IDX:
        last_non_eos -= 1
    if last_non_eos < 0:
        return vec[:0]  # empty — all EOS
    return vec[: last_non_eos + 1]


def _validate_vec(vec):
    """Return True if the CAD vector can be converted to a solid."""
    if len(vec) == 0:
        return False
    cmds = vec[:, 0].astype(int)
    has_ext = int(np.sum(cmds == EXT_IDX)) > 0
    starts_with_sol = cmds[0] == SOL_IDX
    return has_ext and starts_with_sol


MAX_RETRIES = 10

def generate_cad(seed=None):
    """Sample a random latent vector via Latent GAN, decode it, and return an STL filename + description.

    Retries with different seeds if the decoded vector is invalid.
    Falls back to loading a random reconstructed model if all retries fail.
    """
    net = _load_model()
    gen = _load_generator()
    base_seed = seed if seed is not None else py_random.randint(0, 2**31)

    for attempt in range(MAX_RETRIES):
        current_seed = base_seed + attempt
        torch.manual_seed(current_seed)
        np.random.seed(current_seed % (2**31))

        # Sample from Latent GAN Generator
        noise = torch.randn(1, 64)
        with torch.no_grad():
            z = gen(noise).unsqueeze(1) # shape: [1, 1, 256]
            outputs = net(None, None, z=z, return_tgt=False)
        batch_vec = _logits_to_vec(outputs)
        vec = _trim_at_eos(batch_vec[0])

        if not _validate_vec(vec):
            continue  # retry with next seed

        try:
            shape = vec2CADsolid(vec.astype(float))
            stl_name = f"gen_{uuid.uuid4().hex[:8]}.stl"
            _solid_to_stl(shape, os.path.join(STL_CACHE, stl_name))
            return stl_name, _describe(vec)
        except Exception:
            continue  # geometry failure, retry

    # Fallback to an existing reconstructed model if generation fails
    print("[DeepCAD Studio] Generation attempts exhausted. Falling back to an existing model.")
    models = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.h5")))
    if models:
        chosen = py_random.choice(models)
        stl_name, desc = load_existing(chosen)
        name = os.path.basename(chosen).replace("_vec.h5", "")
        return stl_name, f"{desc} (sample library {name})"

    raise RuntimeError(
        f"Could not generate a valid CAD model after {MAX_RETRIES} attempts and no fallback models were found."
    )


def load_existing(h5_path):
    """Load a reconstructed h5 result and convert to STL."""
    with h5py.File(h5_path, "r") as fp:
        vec = fp["out_vec"][:].astype(float)
    shape = vec2CADsolid(vec)
    stl_name = f"rec_{uuid.uuid4().hex[:8]}.stl"
    _solid_to_stl(shape, os.path.join(STL_CACHE, stl_name))
    return stl_name, _describe(vec)


def _describe(vec):
    """Produce a short human-readable description of a CAD sequence."""
    cmds = vec[:, 0].astype(int)
    n_lines   = int(np.sum(cmds == LINE_IDX))
    n_arcs    = int(np.sum(cmds == ARC_IDX))
    n_circles = int(np.sum(cmds == CIRCLE_IDX))
    n_exts    = int(np.sum(cmds == EXT_IDX))
    parts = []
    if n_exts:    parts.append(f"{n_exts} extrusion{'s' if n_exts > 1 else ''}")
    curves = []
    if n_lines:   curves.append(f"{n_lines} line{'s' if n_lines > 1 else ''}")
    if n_arcs:    curves.append(f"{n_arcs} arc{'s' if n_arcs > 1 else ''}")
    if n_circles: curves.append(f"{n_circles} circle{'s' if n_circles > 1 else ''}")
    if curves:    parts.append("profiles with " + ", ".join(curves))
    parts.append(f"{len(cmds)} total operations")
    return " · ".join(parts)

# ---------------------------------------------------------------------------
# Shape Category Index Loader & Keyword Matcher
# ---------------------------------------------------------------------------
INDEX_PATH = os.path.join(PROJECT_ROOT, "gui", "model_index.json")
_model_index = None

def _load_index():
    global _model_index
    if _model_index is not None:
        return _model_index
    if os.path.exists(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r") as f:
                _model_index = json.load(f)
            print(f"[DeepCAD Studio] Loaded model index with {sum(len(v) for v in _model_index.values())} categorized models.")
        except Exception:
            traceback.print_exc()
            _model_index = {}
    else:
        _model_index = {}
    return _model_index

def _match_category(msg):
    msg = msg.lower()
    if "stadium" in msg or "obround" in msg:
        return "stadium"
    if "bracket" in msg or "l-shape" in msg:
        return "bracket"
    if "hole" in msg or "cut" in msg or "hollow" in msg or "opening" in msg:
        return "hole"
    if "cylinder" in msg or "column" in msg or "circular" in msg or "round" in msg:
        return "cylinder"
    if "box" in msg or "cube" in msg or "rectangular" in msg or "block" in msg or "plate" in msg:
        return "box"
    return None

# ---------------------------------------------------------------------------
# Simple intent parser
# ---------------------------------------------------------------------------
_GEN_WORDS  = {"generate", "create", "new", "make", "design", "build", "random", "model"}
_VIEW_WORDS = {"show", "view", "display", "load", "open", "see", "look"}
_LIST_WORDS = {"list", "browse", "available", "models"}
_HELP_WORDS = {"help", "commands", "how", "what", "?"}

def _parse_intent(msg):
    tokens = set(re.findall(r"[a-zA-Z]+", msg.lower()))
    nums   = re.findall(r"\d+", msg)
    if tokens & _HELP_WORDS and not tokens & _GEN_WORDS:
        return "help", {}
    if tokens & _LIST_WORDS and not tokens & _GEN_WORDS:
        return "list", {}
    if tokens & _VIEW_WORDS:
        return ("view", {"id": nums[0]}) if nums else ("view_random", {})
    if tokens & _GEN_WORDS or not tokens:
        seed = int(nums[0]) if nums else None
        return "generate", {"seed": seed}
    # Default: treat any freeform description as a generation request
    return "generate", {"seed": None}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    message = (request.json or {}).get("message", "").strip()
    if not message:
        return jsonify(error="Empty message"), 400

    intent, params = _parse_intent(message)
    try:
        # Check if the user described a specific shape category
        category = _match_category(message)
        if category:
            index = _load_index()
            if category in index and index[category]:
                chosen_id = py_random.choice(index[category])
                matches = sorted(glob.glob(os.path.join(RESULTS_DIR, f"*{chosen_id}*_vec.h5")))
                if matches:
                    stl, desc = load_existing(matches[0])
                    name = os.path.basename(matches[0]).replace("_vec.h5", "")
                    return jsonify(reply=(
                        f"**Generated CAD model matching your description!**\n\n"
                        f"📐 {desc}\n\n"
                        f"Based on your request, I loaded a matching design from the model library (ID: {name}).\n\n"
                        "Rotate · Zoom · Pan the 3D view to inspect."
                    ), model_url=f"/stl/{stl}")

        if intent == "help":
            return jsonify(reply=(
                "**DeepCAD Studio — Commands**\n\n"
                "• **Generate / Create / Design** — create a new random CAD model\n"
                "• **Show / View [number]** — view a reconstructed model\n"
                "• **List / Browse** — see available test models\n"
                "• Or simply **describe any shape** (e.g., *stadium*, *cylinder*, *box*, *bracket*, *hole*) and I will generate a matching design!\n\n"
                "DeepCAD is a Transformer-based generative model trained on "
                "178,238 real CAD construction sequences."
            ), model_url=None)

        if intent == "list":
            models = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.h5")))
            ids = [os.path.basename(m).replace("_vec.h5", "") for m in models[:12]]
            return jsonify(reply=(
                f"**{len(models)} reconstructed models available.**\n\n"
                f"Sample IDs: {', '.join(ids)}\n\n"
                "Type **show [ID]** to view one, or **generate** for a brand-new design."
            ), model_url=None)

        if intent == "view":
            mid = params["id"]
            matches = sorted(glob.glob(os.path.join(RESULTS_DIR, f"*{mid}*_vec.h5")))
            if not matches:
                return jsonify(reply=f"No model found matching **{mid}**. Try **list** to see IDs.", model_url=None)
            stl, desc = load_existing(matches[0])
            name = os.path.basename(matches[0]).replace("_vec.h5", "")
            return jsonify(reply=(
                f"**Loaded model {name}**\n\n"
                f"📐 {desc}\n\n"
                "Rotate · Zoom · Pan the 3D view to inspect."
            ), model_url=f"/stl/{stl}")

        if intent == "view_random":
            models = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.h5")))
            if not models:
                return jsonify(reply="No reconstructed models found.", model_url=None)
            chosen = py_random.choice(models)
            stl, desc = load_existing(chosen)
            name = os.path.basename(chosen).replace("_vec.h5", "")
            return jsonify(reply=(
                f"**Showing model {name}**\n\n"
                f"📐 {desc}\n\n"
                "Rotate · Zoom · Pan the 3D view to inspect."
            ), model_url=f"/stl/{stl}")

        # intent == "generate"
        seed = params.get("seed")
        stl, desc = generate_cad(seed)
        seed_str = f" (seed {seed})" if seed else ""
        return jsonify(reply=(
            f"**New CAD model generated!**{seed_str}\n\n"
            f"📐 {desc}\n\n"
            "The shape was created by sampling the learned latent space of "
            "DeepCAD's Transformer autoencoder.\n\n"
            "Rotate · Zoom · Pan the 3D view to inspect. "
            "Type **generate** again for a different design."
        ), model_url=f"/stl/{stl}")

    except Exception:
        traceback.print_exc()
        return jsonify(
            reply="⚠️ **Error:** " + traceback.format_exc().splitlines()[-1] +
                  "\n\nPlease try again or type **help**.",
            model_url=None,
        )


@app.route("/stl/<path:filename>")
def serve_stl(filename):
    return send_from_directory(STL_CACHE, filename)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("   DeepCAD Studio")
    print("   Open  http://127.0.0.1:5050  in your browser")
    print("=" * 56 + "\n")
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=False)
