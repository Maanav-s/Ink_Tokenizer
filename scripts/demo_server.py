"""Demo: write an expression with a stylus, the recognizer reads it as LaTeX,
Claude answers with LaTeX, and the student writes the answer as ink.

Serves scripts/demo/index.html and three JSON endpoints the page calls in turn:

  POST /api/read   {"strokes": [[[x, y], ...], ...]}   -> {"latex": ...}
  POST /api/claude {"latex": ..., "instruction": ..., "box": ..., "board": ...}
                                                       -> {"latex": ..., "box": ...}
                   (without "instruction", Claude suggests the right-hand side once the LaTeX
                   ends in =, and "latex" may be empty)
  POST /api/write  {"latex": ...}                      -> {"strokes": [[[x, y], ...], ...]}

Strokes are in any unit with y down (the page sends canvas pixels); the
written ink comes back in the student's own units, for the page to scale into
its box. Boxes are {"left", "top", "right", "bottom"} in the page's units and
the board is {"width", "height"}: the page sends the box around what was
written, and Claude guesses the box its reply goes in, which the page draws
the written ink into.

/api/claude needs the `demo` extra and ANTHROPIC_API_KEY, from the shell or
from a .env at the repository root (see .env.example). Claude is told the student's vocabulary, and a
reply that uses other tokens is refused rather than written.

Usage: python scripts/demo_server.py --student models/student/spline_flow/checkpoint.pt
           --recognizer models/recognizer/<run>/checkpoint.pt [--port 8000] [--device cuda]
           [--model claude-opus-5-5]
"""
import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

from ink_corpus import MAX_TEXT, char_tokens, split_text
from mathwriting_corpus import to_offsets
from recognizer_model import Recognizer, ink_positions
from sample_student import load_model

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "index.html")
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

VOCABULARY = """Your whole reply is handwritten onto the board exactly as you send it, so work silently: never
explain, show steps or reason in the reply, because every character of it ends up on the board. It is
written by a model that can only write these LaTeX tokens (one per line), and at most {max_text} of them:

{vocabulary}"""

PLACEMENT = """You are also given the board's size and the bounding box of the user's writing, in pixels with x to
the right and y down. Give the box your reply is written into, as its left, top, right and bottom edges:
where the user would write it, with symbols the same size as theirs, inside the board and not overlapping
their writing. A right-hand side starts just right of the equals sign, which ends the writing, and is
centered on the same line; anything else goes wherever it reads naturally next to what is written.
"""

INSTRUCTION = """The user handwrote a math expression on a whiteboard, and a recognizer read it as the LaTeX below.
Follow the user's instruction and reply with a single LaTeX expression and nothing else: no $ signs, no
code fences, no words. """ + PLACEMENT + VOCABULARY

AUTOCOMPLETE = """The user is handwriting math on a whiteboard, and a recognizer read what they have written so far
as the LaTeX below. It ends with an equals sign. Suggest the right-hand side, the way code autocomplete
would: reply with only the LaTeX that goes after the equals sign, not what is already written, and nothing
else: no $ signs, no code fences, no words. Give the form the user would most likely write next. A wrong
suggestion is worse than none, so if you are not confident, reply with nothing. For example, for
\\int_{{0}}^{{2}}x^{{3}}\\,dx= the whole reply is: 4
""" + PLACEMENT + VOCABULARY

BOX = {"type": "object", "properties": {edge: {"type": "number"} for edge in ("left", "top", "right", "bottom")},
       "required": ["left", "top", "right", "bottom"], "additionalProperties": False}
# At low effort Claude sometimes works the answer out in its reply; fixed fields leave no room for that.
REPLY_FORMAT = {"type": "json_schema", "schema": {"type": "object",
                                                  "properties": {"latex": {"type": "string"}, "box": BOX},
                                                  "required": ["latex", "box"], "additionalProperties": False}}


def load_env(path):
    """KEY=value lines of a .env file into os.environ, without overriding the shell."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Demo:
    def __init__(self, args):
        self.device = args.device
        self.model = args.model

        rec_ckpt = torch.load(args.recognizer, map_location=self.device, weights_only=False)
        self.recognizer = Recognizer(**rec_ckpt["config"]).to(self.device)
        self.recognizer.load_state_dict(rec_ckpt["model"])
        self.recognizer.eval()
        self.rec_charset = rec_ckpt["charset"]
        # Drawn ink has to be resampled as the recognizer's training ink was.
        self.spacing = rec_ckpt["meta"].get("spacing", 0.1)

        self.student, student_ckpt = load_model(args.student, self.device)
        self.student_token = char_tokens(student_ckpt["charset"])
        self.tokenizer = student_ckpt["meta"].get("tokenizer", "char")
        self.mu = np.array(student_ckpt["meta"]["mu"])
        self.std = np.array(student_ckpt["meta"]["std"])
        self.client = None

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda"))

    @torch.no_grad()
    def read(self, strokes):
        offsets = to_offsets([np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in strokes], self.spacing)
        if offsets is None:
            raise ValueError("no ink to read")
        offsets = torch.from_numpy(offsets).unsqueeze(0).to(self.device)
        ink_len = torch.tensor([offsets.shape[1]], device=self.device)
        # to_offsets gives raw offsets, so ink_positions has nothing to denormalize.
        ink = ink_positions(offsets, ink_len, torch.zeros(3, device=self.device), torch.ones(3, device=self.device))
        with self.autocast():
            ids = self.recognizer.read(ink, ink_len)[0]
        return "".join(self.rec_charset[t - 2] for t in ids)

    def ask_claude(self, latex, instruction, box, board):
        """The reply's LaTeX and the box Claude puts it in. With no instruction, Claude suggests the
        right-hand side of LaTeX that ends in =, and may reply with nothing."""
        import anthropic

        if instruction is None and not latex.rstrip().endswith("="):
            return {"latex": "", "box": None}
        if self.client is None:
            self.client = anthropic.Anthropic()
        prompt = AUTOCOMPLETE if instruction is None else INSTRUCTION
        system = prompt.format(max_text=MAX_TEXT, vocabulary="\n".join(self.student_token))
        content = (f"Board: {board['width']:.0f} x {board['height']:.0f}\n"
                   f"Writing's box: left {box['left']:.0f}, top {box['top']:.0f}, "
                   f"right {box['right']:.0f}, bottom {box['bottom']:.0f}\n\nLaTeX: {latex}")
        if instruction is not None:
            content = f"Instruction: {instruction}\n\n{content}"
        # Haiku has no server-side refusal fallback.
        fallback = {} if self.model.startswith("claude-haiku") else {
            "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            **fallback,
            output_config={"effort": "low", "format": REPLY_FORMAT},
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason == "refusal":
            raise ValueError("Claude declined to answer")
        reply = json.loads("".join(b.text for b in response.content if b.type == "text"))
        reply["latex"] = re.sub(r"^\$+|\$+$", "", reply["latex"].strip().strip("`").strip()).strip()
        return reply

    @torch.no_grad()
    def write(self, latex):
        tokens = split_text(latex, self.tokenizer)
        unknown = sorted({t for t in tokens if t not in self.student_token})
        if unknown:
            raise ValueError(f"the student cannot write {' '.join(unknown)}")
        if not tokens or len(tokens) > MAX_TEXT:
            raise ValueError(f"the student writes 1 to {MAX_TEXT} tokens, not {len(tokens)}")
        text = torch.tensor([[self.student_token[t] for t in tokens]], device=self.device)
        text_len = torch.tensor([len(tokens)], device=self.device)
        with self.autocast():
            offsets, _, _ = self.student.sample(text, text_len, 0.0)[0]
        offsets = offsets.float().numpy()
        xy = np.cumsum(offsets[:, :2] * self.std[:2] + self.mu[:2], axis=0)
        ends = np.flatnonzero(offsets[:, 2] > 0.5) + 1
        return [stroke.round(4).tolist() for stroke in np.split(xy, ends) if len(stroke)]


def handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                if self.path == "/api/read":
                    self.send(200, {"latex": demo.read(request["strokes"])})
                elif self.path == "/api/claude":
                    self.send(200, demo.ask_claude(request["latex"], request.get("instruction"),
                                                   request["box"], request["board"]))
                elif self.path == "/api/write":
                    self.send(200, {"strokes": demo.write(request["latex"])})
                else:
                    self.send(404, {"error": "not found"})
            except Exception as e:
                self.send(400, {"error": f"{type(e).__name__}: {e}"})

    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--student", required=True, help="text-to-ink student checkpoint (default kind: spline flow)")
    ap.add_argument("--recognizer", required=True, help="ink-to-LaTeX recognizer checkpoint")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="claude-opus-5-5", help="Claude model for /api/claude")
    args = ap.parse_args()
    load_env(ENV_FILE)
    demo = Demo(args)
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), handler(demo)).serve_forever()
