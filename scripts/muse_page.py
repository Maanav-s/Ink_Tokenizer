"""Have Meta's Muse Spark write a synthetic page's content.json, then lay it out and render it as InkML.

With --variant whiteboard the model gets docs/content_prompt.md's prompt with a topic and --mix (an
engineer's whiteboard page). The other variants are a student's worked solutions to --assignment:
docs/content_prompts/assignment.md plus that subject's rules (math, logic, circuits or writing). Both
get content_prompt.md's schema, characters and LaTeX subset. The content is checked with validate_page.check_content; failures are sent back for another attempt rather
than fixed here, so the checks keep the ground truth right. The accepted page goes through
layout_engine (seed 0) and render_inkml (clean futural ink), on a portrait paper page (US Letter by
default, as in an iPad note app; --paper whiteboard for the board used by the other pages).

The default model is Contributor mode: much cheaper, but Meta may train on the prompts and replies.
Pass --model muse-spark-1.3 for Standard mode.

Usage: uv run --extra render python scripts/muse_page.py artifacts/page_0008 --variant math \\
           --topic "probability problem set worked solutions" \\
           --assignment docs/prompts/probability_assignment.txt
Needs MUSE_API_KEY in .env or the shell.
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request

import layout_engine
import render_inkml
from validate_page import check_content, content_warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_URL = "https://api.meta.ai/v1/chat/completions"
DEFAULT_MIX = "a table, several equations or short derivations with checks, and a numbered list"
SCHEMA = "ink_tokenizer.synthetic_page.content/v1"
VARIANTS = ("whiteboard", "math", "logic", "circuits", "writing", "physics")

# Per-page style draws, so pages on the same assignment differ in notation and habits as students'
# do. Each knob is one line of the prompt; how things are drawn (resistor symbol, aromatic rings,
# stroke order) is varied by the layout seed instead. Every option must still pass the validator.
COMMON_STYLE = {
    "headings": ["number problems as 1. 2. 3.", "number problems as 1) 2) 3)", "number problems as Q1, Q2, Q3",
                 "number problems as #1, #2, #3", "number problems as P1., P2., P3."],
    "answers": ["box each final answer with \\boxed{...} in its math line",
                "underline each final answer (an underline annotation on its element)",
                "mark final answers with a short 'ans:' note or arrow",
                "do not mark final answers specially"],
    "verbosity": ["very terse: almost no words between steps", "a few words of explanation per problem",
                  "a short note naming the method before each problem's working"],
    "sanity": ["add a one-line sanity check ('check: units OK', 'sign OK') on one or two problems",
               "no sanity checks"],
}
SUBJECT_STYLE = {
    "math": {
        "derivatives": ["prime notation f'(x)", "Leibniz notation \\frac{dy}{dx}"],
        "numbers": ["keep exact fractions and radicals", "convert to decimals at the end (3 s.f.)"],
        "products": ["write products by juxtaposition (2x)", "write products with \\cdot"],
        "step size": ["one algebraic operation per line", "combine two easy operations per line"],
    },
    "logic": {
        "row order": ['truth tables in binary counting order (row_order "binary_counting_msb_first")',
                      'truth tables in Gray code order (row_order "gray_code": 00, 01, 11, 10)'],
        "not": ["NOT written as a prime in latex (A')", "NOT written as an overbar in latex (\\overline{A})"],
        "and": ["AND written by juxtaposition in latex (AB)", "AND written with \\cdot in latex (A \\cdot B)"],
        "names": ["inputs named A, B, C", "inputs named x, y, z", "inputs named a, b, c"],
    },
    "circuits": {
        "names": ["components named R1, R2, C1", "components named R_a, R_b, C"],
        "units": ["values in kΩ and mA", "values in Ω and A with powers of ten"],
        "parallel": ["write parallel combinations as R_2 \\parallel R_3 first", "go straight to product over sum"],
        "method": ["prefer nodal analysis where there is a choice", "prefer mesh / loop analysis where there is a choice"],
    },
    "physics": {
        "g": ["g = 9.8 m/s^2", "g = 9.81 m/s^2"],
        "givens": ["list givens first as a list", "write givens as note lines"],
        "symbols": ["Δx and v_0 notation", "d and u notation (v^2 = u^2 + 2ad)"],
    },
    "writing": {
        "outline": ["outline with roman numerals I. II. III.", "outline with 1. 2. 3. and a. b.",
                    "outline with dashes only"],
        "abbreviations": ["heavy abbreviations (w/, b/c, →, +)", "mostly full words"],
        "draft": ["draft prose in full sentences", "draft prose as fragments"],
    },
}


def draw_style(variant, seed):
    rng = random.Random(seed)
    knobs = {**COMMON_STYLE, **SUBJECT_STYLE.get(variant, {})} if variant != "whiteboard" else {}
    return {k: rng.choice(v) for k, v in knobs.items()}


def load_env(path=os.path.join(ROOT, ".env")):
    """KEY=value lines of a .env file into os.environ, without overriding the shell."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_prompt(variant, topic, mix, assignment, style=None):
    doc = open(os.path.join(ROOT, "docs", "content_prompt.md")).read()
    reference = "## Schema" + doc.split("## Schema", 1)[1]
    if variant == "whiteboard":
        prompt_section = doc.split("## Prompt", 1)[1].split("## Schema", 1)[0]
        prompt = "\n".join(line[2:] if line.startswith("> ") else line.lstrip(">")
                           for line in prompt_section.strip().splitlines())
        prompt = prompt.replace("{topic}", topic).replace("{mix}", mix)
        if assignment:
            prompt += "\n\nThe page works through this assignment, with each problem's solution:\n\n" + assignment
        return prompt + "\n\n" + reference
    # The whiteboard advice on what makes good content (truth tables, diagrams) would pull every
    # assignment page towards them, so it is left out.
    reference = reference.split("## What makes good content", 1)[0]
    prompts = os.path.join(ROOT, "docs", "content_prompts")
    prompt = open(os.path.join(prompts, "assignment.md")).read().replace("{topic}", topic)
    rules = open(os.path.join(prompts, f"{variant}.md")).read()
    if style:
        rules += "\nStyle for this page (follow it throughout):\n" + "".join(f"- {v}\n" for v in style.values())
    return f"{prompt}\n{rules}\nThe assignment:\n\n{assignment}\n\n{reference}"


def call_muse(messages, model, effort, api_key):
    body = {"model": model, "messages": messages, "reasoning_effort": effort,
            "max_completion_tokens": 32000}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"Muse API error {e.code}: {e.read().decode(errors='replace')[:2000]}")
    usage = reply.get("usage", {})
    print(f"{model}: {usage.get('prompt_tokens')} prompt + {usage.get('completion_tokens')} completion tokens")
    return reply["choices"][0]["message"]["content"]


def parse_json(text):
    # Models sometimes wrap the object in a ```json fence despite being told not to.
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return json.loads(m.group(1) if m else text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("page_dir")
    ap.add_argument("--variant", choices=VARIANTS, default="whiteboard")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--mix", default=DEFAULT_MIX)
    ap.add_argument("--assignment", help="text file with problems the page works through")
    ap.add_argument("--model", default="muse-spark-1.3-contributor")
    ap.add_argument("--effort", default="medium", help="minimal, low, medium, high or xhigh")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--style-seed", help="seed for the page's style draws (default: the page id)")
    ap.add_argument("--paper", default="letter", choices=[*sorted(layout_engine.PAPER_MM), "whiteboard"])
    a = ap.parse_args()

    load_env()
    api_key = os.environ.get("MUSE_API_KEY")
    if not api_key:
        sys.exit("MUSE_API_KEY is not set (in .env or the shell)")

    assignment = open(a.assignment).read() if a.assignment else None
    if a.variant != "whiteboard" and not assignment:
        sys.exit(f"--variant {a.variant} needs --assignment")
    page_id = os.path.basename(os.path.normpath(a.page_dir))
    style = draw_style(a.variant, a.style_seed or page_id)
    for k, v in style.items():
        print(f"style {k}: {v}")
    messages = [{"role": "user", "content": build_prompt(a.variant, a.topic, a.mix, assignment, style)}]
    os.makedirs(a.page_dir, exist_ok=True)

    for attempt in range(1, a.attempts + 1):
        reply = call_muse(messages, a.model, a.effort, api_key)
        with open(os.path.join(a.page_dir, f"muse_reply_{attempt}.txt"), "w") as f:
            f.write(reply)
        try:
            content = parse_json(reply)
            # Boilerplate the model often leaves out; filling it in saves a whole retry.
            content["schema"] = SCHEMA
            content["page_id"] = page_id
            content["provenance"] = {"generator": f"{a.model} / content_prompt v1 ({a.variant})",
                                     "automated": True, "style": style}
            errors = check_content(content)
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            errors = [f"reply is not a single JSON object: {e}"]
        if not errors:
            break
        print(f"attempt {attempt}: {len(errors)} problem(s)")
        for e in errors[:20]:
            print(f"  - {e}")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "The validator rejected this page:\n- " + "\n- ".join(errors)
                      + "\n\nFix every problem and output the whole corrected JSON object, and nothing else."}]
    else:
        sys.exit(f"no valid content after {a.attempts} attempts; replies are in {a.page_dir}")

    for w in content_warnings(content):
        print(f"warning: {w}")
    with open(os.path.join(a.page_dir, "content.json"), "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
    layout_engine.write_layout(a.page_dir, 0, paper=None if a.paper == "whiteboard" else a.paper)
    render_inkml.render(a.page_dir, ["futural"], 0.0, 0)


if __name__ == "__main__":
    main()
