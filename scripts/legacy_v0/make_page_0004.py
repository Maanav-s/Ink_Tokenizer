"""Hand-authored synthetic page 0004 (RC low-pass filter, math-heavy).

Emits artifacts/page_0004/content.json and layout.json. Standard library only.

The page is an engineer's whiteboard derivation for an RC low-pass filter:
KCL at the output node, the capacitor voltage as an integral of the current,
the step response with tau = RC, the transfer function by voltage division,
its magnitude and the -3 dB cutoff frequency, plus a small schematic.

Two-dimensional math is built from generic sketch items (artifacts/README.md,
"sketch"): text items on a baseline, smaller raised or lowered text items for
superscripts and subscripts, polylines for fraction bars and radical
overbars, and a rect for the boxed result. Each equation is one sketch
element in layout.json keyed by the math element's id; its items are named
with the equation id as prefix (eq_3.sup1, eq_6.f2.bar, ...). content.json
keeps each equation semantically (latex, a plain checkable form, defines, and
the list of layout items that draw it).

Before writing, the script asserts:
- the math: every equation is evaluated numerically on several samples, each
  continuation line is checked against the line it continues, and the step
  and frequency responses are checked against an RK4 simulation of the KCL
  differential equation;
- the layout: items exist, writing_order lists every leaf once, fraction bars
  cover their numerator and denominator, superscripts and subscripts sit at
  the right height, '=' signs are aligned, nothing overlaps unintentionally,
  everything is on the page, and only renderer-supported characters are used.
"""
import cmath
import json
import math
import os
import random
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "page_0004")
W = "writer_0"
PAGE_W, PAGE_H = 1200, 800
CHAR_W = 0.55  # README's nominal estimate, recorded in text_metrics

# Glyph advance per character as a fraction of text_height: the maximum over
# the Hershey fonts futural, cursive, scripts and timesi, as measured from
# scripts/render_inkml.py's TextRenderer.width(). Boxes sized with these are
# never narrower than the ink in any of those fonts (0.55/char is too narrow
# for capitals, digits and operators).
ADV = {' ': 0.57, "'": 0.36, '(': 0.54, ')': 0.54, '+': 0.93, ',': 0.39, '-': 0.93, '.': 0.39,
       '/': 0.79, ':': 0.39, '=': 0.93, '|': 0.29, '·': 0.30, '→': 0.75, '∫': 0.45, '√': 0.55,
       'Ω': 0.71, 'π': 0.79, 'τ': 0.71, 'ω': 0.82, 'α': 0.75, 'μ': 0.75,
       'A': 0.71, 'B': 0.86, 'C': 0.75, 'D': 0.82, 'E': 0.82, 'F': 0.79, 'G': 0.79, 'H': 0.93,
       'I': 0.52, 'J': 0.64, 'K': 0.82, 'L': 0.71, 'M': 1.0, 'N': 0.89, 'O': 0.79, 'P': 0.82,
       'Q': 0.79, 'R': 0.86, 'S': 0.82, 'T': 0.75, 'U': 0.89, 'V': 0.71, 'W': 0.93, 'X': 0.79,
       'Y': 0.75, 'Z': 0.79, 'a': 0.75, 'b': 0.68, 'c': 0.64, 'd': 0.75, 'e': 0.64, 'f': 0.54,
       'g': 0.71, 'h': 0.75, 'i': 0.46, 'j': 0.46, 'k': 0.71, 'l': 0.43, 'm': 1.18, 'n': 0.82,
       'o': 0.68, 'p': 0.75, 'q': 0.71, 'r': 0.61, 's': 0.61, 't': 0.5, 'u': 0.82, 'v': 0.71,
       'w': 1.04, 'x': 0.71, 'y': 0.75, 'z': 0.71}
ADV.update({d: 0.75 for d in "0123456789"})

# Vertical glyph geometry, as fractions of text_height measured from the
# rendered ink. A text item's box spans [top, top + h]; the cap line is the
# top and the baseline is 0.25 h above the bottom (futural; cursive's
# baseline is about 0.1 h higher). Parentheses and bars reach 0.15 h above
# the cap line.
BASE_FROM_TOP = 0.75
# Ink extents of the primitive-drawn ∫ and √, as fractions of their
# text_height relative to the box top, averaged over futural and cursive
# (futural / cursive: ∫ 0.02..1.05 / -0.10..0.94, √ 0.03..0.80 / -0.08..0.69;
# the cursive baseline sits 0.11 h higher in the box).
INT_INK = (-0.04, 1.0, 0.04, 0.41)    # (top, bottom, left, right) for ∫
SQRT_INK = (-0.025, 0.745, 0.0, 0.55)  # for √; the top-right corner ends the sign

SUP_SCALE, SUP_RAISE = 0.6, 0.45   # superscript height, baseline raise (x base height)
SUB_SCALE, SUB_DROP = 0.6, 0.20    # subscript height, baseline drop
AXIS = 0.30                        # math axis (fraction bar) above the baseline
SUPPORTED = set(" !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
                "abcdefghijklmnopqrstuvwxyz{|}~") | \
    set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ") | set("⊕→≤≥≠±·×∞∫√")


def tw(s, h):
    return sum(ADV.get(c, 0.8) for c in s) * h


def r1(v):
    return round(v, 1)


# ------------------------------------------------------------- typesetting
# A small box model. Every node knows its width, ascent and descent (mm,
# relative to its baseline) and emits sketch items, in writing order, at a
# given x and baseline B. Leaf names are relative; the equation id is
# prefixed on emit.

class Ctx:
    def __init__(self, prefix):
        self.prefix, self.items, self.order, self.meta, self.n = prefix, {}, [], {}, {}

    def name(self, stem):
        self.n[stem] = self.n.get(stem, 0) + 1
        return f"{self.prefix}.{stem}{self.n[stem]}"

    def add(self, iid, item, **meta):
        assert iid not in self.items, iid
        item["writer"] = W
        self.items[iid] = item
        self.order.append(iid)
        self.meta[iid] = meta
        return iid


class T:
    """A run of text at the row's height (or scaled)."""
    role = "text"

    def __init__(self, s, name=None, scale=1.0):
        self.s, self.nm, self.scale = s, name, scale

    def h(self, h):
        return h * self.scale

    def measure(self, h):
        hh = self.h(h)
        asc = hh * (BASE_FROM_TOP + (0.15 if set(self.s) & set("()|") else 0))
        return tw(self.s, hh), asc, hh * (1 - BASE_FROM_TOP)

    def emit(self, ctx, x, B, h, prev=None):
        hh = self.h(h)
        top = B - BASE_FROM_TOP * hh
        iid = f"{ctx.prefix}.{self.nm}" if self.nm else ctx.name("t")
        ctx.add(iid, {"kind": "text", "role": self.role, "text": self.s,
                      "bbox": [r1(x), r1(top), r1(x + tw(self.s, hh)), r1(top + hh)],
                      "text_height": r1(hh)}, baseline=B, h=hh, base=prev)
        return iid


class Sup(T):
    role = "superscript"

    def measure(self, h):
        w, a, d = T.measure(self, SUP_SCALE * h)
        return w, a + SUP_RAISE * h, max(0.0, d - SUP_RAISE * h)

    def emit(self, ctx, x, B, h, prev=None):
        if not self.nm:
            self.nm = ctx.name("sup")[len(ctx.prefix) + 1:]
        return T.emit(T(self.s, self.nm), ctx, x, B - SUP_RAISE * h, SUP_SCALE * h, prev) and \
            self._tag(ctx, prev, B, h)

    def _tag(self, ctx, prev, B, h):
        iid = ctx.order[-1]
        ctx.items[iid]["role"] = self.role
        ctx.meta[iid].update(base=prev, base_baseline=B, base_h=h)
        return iid


class Sub(Sup):
    role = "subscript"

    def measure(self, h):
        w, a, d = T.measure(self, SUB_SCALE * h)
        return w, max(0.0, a - SUB_DROP * h), d + SUB_DROP * h

    def emit(self, ctx, x, B, h, prev=None):
        if not self.nm:
            self.nm = ctx.name("sub")[len(ctx.prefix) + 1:]
        T.emit(T(self.s, self.nm), ctx, x, B + SUB_DROP * h, SUB_SCALE * h, prev)
        return self._tag(ctx, prev, B, h)


class Row:
    def __init__(self, *kids, gap=0.12):
        self.kids, self.gap = kids, gap

    def _gaps(self, h):
        # no gap before a sub/superscript: it attaches to the preceding run;
        # binary operators and relations get more room
        op = lambda k: type(k) is T and k.s in ("=", "+", "-", "·")
        return [0 if i == 0 or isinstance(k, Sup) else
                (0.22 if op(k) or op(self.kids[i - 1]) else self.gap) * h
                for i, k in enumerate(self.kids)]

    def measure(self, h):
        ms = [k.measure(h) for k in self.kids]
        return (sum(m[0] for m in ms) + sum(self._gaps(h)),
                max(m[1] for m in ms), max(m[2] for m in ms))

    def emit(self, ctx, x, B, h, prev=None):
        last = prev
        for k, g in zip(self.kids, self._gaps(h)):
            x += g
            last = k.emit(ctx, x, B, h, last if isinstance(k, Sup) else None) or last
            x += k.measure(h)[0]
        return last


class Frac:
    """Numerator, bar, denominator; the bar sits on the math axis."""

    def __init__(self, num, den, name, scale=0.92):
        self.num, self.den, self.nm, self.scale = num, den, name, scale

    def parts(self, h):
        hf = self.scale * h
        wn, an, dn = self.num.measure(hf)
        wd, ad, dd = self.den.measure(hf)
        pad, gap = 0.15 * h, 0.14 * h
        width = max(wn, wd) + 2 * pad
        return hf, wn, an, dn, wd, ad, dd, width, gap

    def measure(self, h):
        hf, wn, an, dn, wd, ad, dd, width, gap = self.parts(h)
        return width, AXIS * h + gap + dn + an, gap + ad + dd - AXIS * h

    def emit(self, ctx, x, B, h, prev=None):
        hf, wn, an, dn, wd, ad, dd, width, gap = self.parts(h)
        bar = B - AXIS * h
        p = ctx.prefix
        sub = Ctx(f"{p}.{self.nm}.num")
        self.num.emit(sub, x + (width - wn) / 2, bar - gap - dn, hf)
        num_ids = list(sub.order)
        sub.prefix = f"{p}.{self.nm}.den"
        sub.n = {}
        n_before = len(sub.order)
        bar_id = f"{p}.{self.nm}.bar"
        # writing order: numerator, bar, denominator
        for iid in num_ids:
            ctx.add(iid, sub.items[iid], **sub.meta[iid])
        ctx.add(bar_id, {"kind": "polyline", "role": "fraction_bar",
                         "points": [[r1(x), r1(bar)], [r1(x + width), r1(bar)]]},
                num=num_ids, den=None)
        self.den.emit(sub, x + (width - wd) / 2, bar + gap + ad, hf)
        den_ids = sub.order[n_before:]
        for iid in den_ids:
            ctx.add(iid, sub.items[iid], **sub.meta[iid])
        ctx.meta[bar_id]["den"] = den_ids
        return None


class Sqrt:
    """√ sign (a text item), overbar polyline, then the radicand."""

    def __init__(self, body, name):
        self.body, self.nm = body, name

    def parts(self, h):
        wb, ab, db = self.body.measure(h)
        top = ab + 0.12 * h                 # overbar height above the baseline
        bottom = -0.15 * h                  # sign's lowest point (below baseline)
        hr = (top - bottom) / (SQRT_INK[1] - SQRT_INK[0])
        sign_w = SQRT_INK[3] * hr
        return wb, ab, db, top, bottom, hr, sign_w

    def measure(self, h):
        wb, ab, db, top, bottom, hr, sign_w = self.parts(h)
        return sign_w + 0.1 * h + wb + 0.12 * h, top + 0.05 * h, max(db, -bottom)

    def emit(self, ctx, x, B, h, prev=None):
        wb, ab, db, top, bottom, hr, sign_w = self.parts(h)
        y0 = B - top - SQRT_INK[0] * hr     # box top so the ink top lands on the bar
        sid = f"{ctx.prefix}.{self.nm}.sign"
        ctx.add(sid, {"kind": "text", "role": "radical_sign", "text": "√",
                      "bbox": [r1(x), r1(y0), r1(x + sign_w), r1(y0 + hr)], "text_height": r1(hr)},
                ink=[x, y0 + SQRT_INK[0] * hr, x + sign_w, y0 + SQRT_INK[1] * hr])
        bx = x + sign_w + 0.1 * h
        bid = f"{ctx.prefix}.{self.nm}.bar"
        ctx.add(bid, {"kind": "polyline", "role": "radical_bar",
                      "points": [[r1(x + sign_w), r1(B - top)], [r1(bx + wb + 0.12 * h), r1(B - top)]]},
                sign=sid)
        sub = Ctx(f"{ctx.prefix}.{self.nm}.body")
        self.body.emit(sub, bx, B, h)
        for iid in sub.order:
            ctx.add(iid, sub.items[iid], **sub.meta[iid])
        ctx.meta[bid]["body"] = list(sub.order)
        return None


class Integral:
    """∫ with lower and upper limits written to its right."""

    def __init__(self, lo, hi, name, scale=1.6):
        self.lo, self.hi, self.nm, self.scale = lo, hi, name, scale

    def measure(self, h):
        hi = self.scale * h
        hl = 0.55 * h
        w = INT_INK[3] * hi + 0.12 * h + max(tw(self.hi, hl), tw(self.lo, hl))
        span = (INT_INK[1] - INT_INK[0]) * hi
        return w, AXIS * h + span / 2, span / 2 - AXIS * h + 0.05 * h

    def emit(self, ctx, x, B, h, prev=None):
        hi, hl = self.scale * h, 0.55 * h
        span = (INT_INK[1] - INT_INK[0]) * hi
        ink_top = B - AXIS * h - span / 2
        y0 = ink_top - INT_INK[0] * hi
        sid = f"{ctx.prefix}.{self.nm}.sign"
        ctx.add(sid, {"kind": "text", "role": "integral_sign", "text": "∫",
                      "bbox": [r1(x), r1(y0), r1(x + ADV['∫'] * hi), r1(y0 + hi)], "text_height": r1(hi)},
                ink=[x + INT_INK[2] * hi, ink_top, x + INT_INK[3] * hi, ink_top + span])
        # limits: lower first (as most people write them), then upper
        lo_B = ink_top + span
        hi_B = ink_top + BASE_FROM_TOP * hl + 0.02 * h
        for nm, s, xx, BB in (("lo", self.lo, x + INT_INK[3] * hi + 0.08 * h, lo_B),
                              ("hi", self.hi, x + INT_INK[3] * hi + 0.12 * h, hi_B)):
            top = BB - BASE_FROM_TOP * hl
            ctx.add(f"{ctx.prefix}.{self.nm}.{nm}",
                    {"kind": "text", "role": f"integral_{'lower' if nm == 'lo' else 'upper'}_limit",
                     "text": s, "bbox": [r1(xx), r1(top), r1(xx + tw(s, hl)), r1(top + hl)],
                     "text_height": r1(hl)}, baseline=BB, h=hl)
        return None


class Eq:
    """One math line: optional LHS, '=' at an x shared by its aligned group, RHS."""

    def __init__(self, eid, lhs, rhs, h=26):
        self.eid, self.lhs, self.rhs, self.h = eid, lhs, rhs, h

    def lhs_w(self):
        return self.lhs.measure(self.h)[0] if self.lhs else 0.0

    def build(self, x_left, x_eq, B):
        ctx = Ctx(self.eid)
        if self.lhs:
            self.lhs.emit(ctx, x_eq - 0.22 * self.h - self.lhs_w(), B, self.h)
        ctx.add(f"{self.eid}.eq", {"kind": "text", "role": "relation", "text": "=",
                                    "bbox": [r1(x_eq), r1(B - BASE_FROM_TOP * self.h),
                                             r1(x_eq + tw("=", self.h)), r1(B + 0.25 * self.h)],
                                    "text_height": self.h}, baseline=B, h=self.h)
        self.rhs.emit(ctx, x_eq + tw("=", self.h) + 0.22 * self.h, B, self.h)
        return ctx


def aligned(eqs, x_left, baselines):
    """Lay out equations whose first '=' share one column."""
    x_eq = x_left + max(e.lhs_w() for e in eqs) + 0.22 * eqs[0].h
    return [e.build(x_left, x_eq, B) for e, B in zip(eqs, baselines)]


# ------------------------------------------------------------- content
R_VAL, C_VAL, V_STEP = 10e3, 100e-9, 5.0

title = {"id": "title", "type": "text", "role": "title", "text": "RC low-pass - step + freq response"}
givens = {"id": "givens", "type": "text", "role": "bullet",
          "text": "given: R = 10 kΩ, C = 100 nF, 5 V step in at t = 0, cap starts at 0 V"}
sec_1 = {"id": "sec_1", "type": "text", "role": "bullet", "text": "KCL at out node:"}
sec_2 = {"id": "sec_2", "type": "text", "role": "bullet", "text": "sine in, phasors:"}

V = lambda s: [T("V"), Sub(s)]
math_src = [
    # id, latex, plain checkable form, defines, continues, Eq
    ("eq_1", r"i = \frac{V_{in} - V_{out}}{R} = C \frac{dV_{out}}{dt}",
     "i = (Vin - Vout)/R = C*dVout/dt", "i", None,
     Eq("eq_1", T("i"), Row(Frac(Row(T("("), *V("in"), T("-"), *V("out"), T(")")), T("R"), "f1"),
                             T("="), T("C"), Frac(Row(T("dV"), Sub("out")), T("dt"), "f2")))),
    ("eq_2", r"V_{out}(t) = \frac{1}{C} \int_0^t i \, dt",
     "Vout(t) = (1/C)*integral(i, 0, t)", "Vout", None,
     Eq("eq_2", Row(*V("out"), T("(t)")), Row(Frac(T("1"), T("C"), "f1"), Integral("0", "t", "int1"),
                                              T("i dt")))),
    ("eq_3", r"= V_{in} \left(1 - e^{-t/\tau}\right)",
     "= Vin*(1 - exp(-t/tau))", "Vout", "eq_2",
     Eq("eq_3", None, Row(*V("in"), T("(1"), T("-"), T("e"), Sup("-t/τ"), T(")")))),
    ("eq_4", r"\tau = RC = 10\,\mathrm{k\Omega} \cdot 100\,\mathrm{nF} = 1\,\mathrm{ms}",
     "tau = R*C = 10e3*100e-9 = 1e-3", "tau", None,
     Eq("eq_4", T("τ"), Row(T("RC"), T("="), T("10 kΩ"), T("·"), T("100 nF"), T("="), T("1 ms")))),
    ("eq_5", r"Z_C = \frac{1}{j \omega C}",
     "ZC = 1/(j*w*C)", "ZC", None,
     Eq("eq_5", Row(T("Z"), Sub("C")), Frac(T("1"), T("jωC"), "f1"))),
    ("eq_6", r"H(j\omega) = \frac{V_{out}}{V_{in}} = \frac{Z_C}{R + Z_C}",
     "H(j*w) = Vout/Vin = ZC/(R + ZC)", "H", None,
     Eq("eq_6", T("H(jω)"), Row(Frac(Row(*V("out")), Row(*V("in")), "f1"), T("="),
                                Frac(Row(T("Z"), Sub("C")), Row(T("R"), T("+"), T("Z"), Sub("C")), "f2")))),
    ("eq_7", r"= \frac{1}{1 + j \omega RC}",
     "= 1/(1 + j*w*R*C)", "H", "eq_6",
     Eq("eq_7", None, Frac(T("1"), Row(T("1"), T("+"), T("jωRC")), "f1"))),
    ("eq_8", r"|H(j\omega)| = \frac{1}{\sqrt{1 + (\omega RC)^2}}",
     "abs(H(j*w)) = 1/sqrt(1 + (w*R*C)**2)", "|H|", None,
     Eq("eq_8", T("|H(jω)|"), Frac(T("1"), Sqrt(Row(T("1"), T("+"), T("(ωRC)"), Sup("2")), "rad1"),
                                   "f1"))),
    ("eq_9", r"f_c = \frac{1}{2 \pi RC} = 159\,\mathrm{Hz}",
     "fc = 1/(2*pi*R*C) = 159", "fc", None,
     Eq("eq_9", Row(T("f"), Sub("c")), Row(Frac(T("1"), T("2πRC"), "f1"), T("="), T("159 Hz")))),
]

# ------------------------------------------------------------- math checks

def simpson(f, a, b, n=400):
    hh = (b - a) / n
    s = f(a) + f(b) + sum((4 if k % 2 else 2) * f(a + k * hh) for k in range(1, n))
    return s * hh / 3


class Sig(float):
    """A time signal: its value at the current t, and callable at any time."""

    def __new__(cls, f, t):
        o = float.__new__(cls, f(t))
        o.f = f
        return o

    def __call__(self, s):
        return self.f(s)


def to_python(expr):
    expr = re.sub(r"d(\w+)/dt", r"ddt(\1)", expr)          # derivative
    expr = re.sub(r"\bj\*", "1j*", expr)                   # imaginary unit
    return expr


def parts(text):
    return [p.strip() for p in text.split("=")]


def close(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def time_env(R, C, Vin, t):
    tau = R * C
    vout = lambda s: Vin * (1 - math.exp(-s / tau))   # the closed form of eq_3
    env = {"R": R, "C": C, "Vin": Vin, "tau": tau, "t": t, "exp": math.exp, "pi": math.pi}
    env["Vout"] = Sig(vout, t)
    env["i"] = Sig(lambda s: (Vin - vout(s)) / R, t)
    env["integral"] = lambda f, a, b: simpson(f, a, b)
    hstep = tau * 1e-4
    env["ddt"] = lambda sig: (sig(t + hstep) - sig(t - hstep)) / (2 * hstep)
    return env


def freq_env(R, C, Vin, w):
    # Phasors: KCL at the out node, (Vin - Vout)/R = jwC * Vout, solved for Vout.
    Vout = Vin / (1 + 1j * w * R * C)
    return {"R": R, "C": C, "Vin": Vin, "Vout": Vout, "w": w, "pi": math.pi, "sqrt": cmath.sqrt,
            "abs": abs, "H": lambda s: Vout / Vin}


def rk4_step_response(R, C, Vin, t_end, n=4000):
    """Integrate KCL, C dv/dt = (vin - v)/R, from v(0) = 0."""
    v, dt = 0.0, t_end / n
    f = lambda vv: (Vin - vv) / (R * C)
    for _ in range(n):
        k1 = f(v); k2 = f(v + dt * k1 / 2); k3 = f(v + dt * k2 / 2); k4 = f(v + dt * k3)
        v += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return v


def rk4_sine_gain(R, C, w, periods=40, steps_per_period=400):
    """Steady-state output amplitude for vin = sin(wt), simulated."""
    tau, dt = R * C, 2 * math.pi / w / steps_per_period
    f = lambda tt, vv: (math.sin(w * tt) - vv) / tau
    v, t, peak = 0.0, 0.0, 0.0
    for k in range(periods * steps_per_period):
        k1 = f(t, v); k2 = f(t + dt / 2, v + dt * k1 / 2)
        k3 = f(t + dt / 2, v + dt * k2 / 2); k4 = f(t + dt, v + dt * k3)
        v += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        if k >= (periods - 2) * steps_per_period:
            peak = max(peak, abs(v))
    return peak


def check_math():
    rng = random.Random(4)
    plain = {m[0]: m[2] for m in math_src}
    cont = {m[0]: m[4] for m in math_src}
    time_eqs, freq_eqs = ["eq_1", "eq_2", "eq_3"], ["eq_5", "eq_6", "eq_7", "eq_8"]
    n_evals = 0
    for _ in range(6):
        R, C, Vin = rng.uniform(100, 1e5), rng.uniform(1e-9, 1e-5), rng.uniform(0.5, 12)
        for t in (0.3 * R * C, R * C, 2.5 * R * C):
            env = time_env(R, C, Vin, t)
            for eid in time_eqs:
                vals = [eval(to_python(p), {}, env) for p in parts(plain[eid]) if p]
                if cont[eid]:  # the continued line's last value
                    vals.insert(0, eval(to_python(parts(plain[cont[eid]])[-1]), {}, env))
                assert all(close(v, vals[0], 1e-5) for v in vals), (eid, vals)
                n_evals += 1
            # the closed form against an independent RK4 integration of KCL
            assert close(rk4_step_response(R, C, Vin, t), env["Vout"], 1e-6)
        for w in (0.2 / (R * C), 1 / (R * C), 7 / (R * C)):
            env = freq_env(R, C, complex(rng.uniform(0.5, 5), rng.uniform(-1, 1)), w)
            for eid in freq_eqs:
                if eid == "eq_5":  # the definition of ZC
                    env["ZC"] = eval(to_python(parts(plain[eid])[1]), {}, env)
                vals = [eval(to_python(p), {}, env) for p in parts(plain[eid]) if p]
                if cont[eid]:
                    vals.insert(0, eval(to_python(parts(plain[cont[eid]])[-1]), {}, env))
                if eid == "eq_5":  # ZC is defined here; check it against the KCL phasor current
                    i = (env["Vin"] - env["Vout"]) / R
                    vals.append(env["Vout"] / i)
                assert all(close(complex(v), complex(vals[0]), 1e-9) for v in vals), (eid, vals)
                n_evals += 1
            # |H| against a simulated sine response
            assert close(rk4_sine_gain(R, C, w), abs(env["H"](1j * w)), 2e-3)
    # numeric lines, at the given component values
    env = {"R": R_VAL, "C": C_VAL, "pi": math.pi}
    for eid, tol in (("eq_4", 1e-9), ("eq_9", 2e-3)):   # eq_9 rounds 159.15 Hz to 159 Hz
        lhs, *rhs = parts(plain[eid])
        vals = [eval(p, {}, env) for p in rhs]
        assert all(close(v, vals[0], tol) for v in vals), (eid, vals)
        env[lhs] = vals[0]
        n_evals += 1
    # fc is the -3 dB point: |H(j 2 pi fc)| = 1/sqrt(2), also in simulation
    w_c = 2 * math.pi * env["fc"]
    assert close(abs(freq_env(R_VAL, C_VAL, 1, w_c)["H"](0)), 2 ** -0.5, 1e-9)
    assert close(rk4_sine_gain(R_VAL, C_VAL, w_c), 2 ** -0.5, 2e-3)
    assert close(env["tau"], R_VAL * C_VAL)
    return n_evals


# ------------------------------------------------------------- layout
el = {}


def text_el(eid, s, x, y, h):
    el[eid] = {"bbox": [x, y, r1(x + tw(s, h)), y + h], "text_height": h, "writer": W}


text_el("title", title["text"], 40, 25, 30)
text_el("givens", givens["text"], 40, 72, 22)
text_el("sec_1", sec_1["text"], 40, 118, 22)

E = {m[0]: m[5] for m in math_src}
X_LEFT, X_RIGHT = 40, 620
left = aligned([E["eq_1"], E["eq_2"], E["eq_3"], E["eq_4"]], X_LEFT, [205, 300, 375, 440])
text_el("sec_2", sec_2["text"], 40, 490, 22)
left2 = aligned([E["eq_5"]], X_LEFT, [575])
right = aligned([E["eq_6"], E["eq_7"], E["eq_8"]], X_RIGHT, [400, 495, 595])
right2 = aligned([E["eq_9"]], X_RIGHT, [715])
ctxs = {c.prefix: c for c in left + left2 + right + right2}

# box around the final result
c9 = ctxs["eq_9"]


def ink_box(iid, item, meta):
    """Approximate ink extent of an item (mm), for overlap and page checks."""
    if "ink" in meta:
        return list(meta["ink"])
    k = item["kind"]
    if k == "text":
        x0, y0, x1, y1 = item["bbox"]
        h = item["text_height"]
        top = y0 - (0.15 * h if set(item["text"]) & set("()|") else 0)
        bottom = y1 if set(item["text"]) & set("gjpqy()|,") else y1 - 0.25 * h
        return [x0, top, x1, bottom]
    if k in ("polyline", "arrow"):
        xs = [p[0] for p in item["points"]]
        ys = [p[1] for p in item["points"]]
        return [min(xs), min(ys), max(xs), max(ys)]
    return list(item["bbox"])


bx = [ink_box(i, c9.items[i], c9.meta[i]) for i in c9.order]
box9 = [r1(min(b[0] for b in bx) - 12), r1(min(b[1] for b in bx) - 10),
        r1(max(b[2] for b in bx) + 12), r1(max(b[3] for b in bx) + 10)]
c9.add("eq_9.box", {"kind": "rect", "role": "result_box", "bbox": box9, "corner_radius": 4})

for c in ctxs.values():
    boxes = [ink_box(i, c.items[i], c.meta[i]) for i in c.order]
    el[c.prefix] = {"bbox": [r1(min(b[0] for b in boxes)), r1(min(b[1] for b in boxes)),
                             r1(max(b[2] for b in boxes)), r1(max(b[3] for b in boxes))],
                    "writer": W, "items": c.items}

# ---- schematic (small): step source, series R, shunt C, ground, out terminal
fig = Ctx("fig_1")
ellipse = lambda cx, cy, r: [cx - r, cy - r, cx + r, cy + r]
pl = lambda *pts: {"kind": "polyline", "points": [list(p) for p in pts]}
SX, SY, SR = 690, 215, 22          # source centre and radius
TOPW, BOTW = 150, 290              # top and bottom wire y
RX0, RX1, NODE, CX = 765, 845, 905, 905
fig.add("fig_1.src", {"kind": "ellipse", "bbox": ellipse(SX, SY, SR)})
fig.add("fig_1.src_plus", {"kind": "text", "text": "+", "text_height": 14,
                           "bbox": [SX - 6, SY - 20, r1(SX - 6 + tw("+", 14)), SY - 6]})
fig.add("fig_1.src_minus", {"kind": "text", "text": "-", "text_height": 14,
                            "bbox": [SX - 6, SY + 2, r1(SX - 6 + tw("-", 14)), SY + 16]})
lab = Ctx("fig_1.lbl_vin")
Row(*V("in")).emit(lab, SX - SR - 12 - Row(*V("in")).measure(22)[0], SY + 8, 22)
for i in lab.order:
    fig.add(i, lab.items[i], **lab.meta[i])
fig.add("fig_1.w_in", pl((SX, SY - SR), (SX, TOPW), (RX0, TOPW)))
zig = [(RX0, TOPW)] + [(RX0 + 6 + 13.6 * k, TOPW - 9 if k % 2 == 0 else TOPW + 9) for k in range(6)] + [(RX1, TOPW)]
fig.add("fig_1.res", {"kind": "polyline", "points": [[r1(x), y] for x, y in zig]})
fig.add("fig_1.lbl_r", {"kind": "text", "text": "R", "text_height": 22,
                        "bbox": [796, 108, r1(796 + tw("R", 22)), 130]})
fig.add("fig_1.w_mid", pl((RX1, TOPW), (NODE, TOPW)))
fig.add("fig_1.w_cap_top", pl((CX, TOPW), (CX, 210)))
fig.add("fig_1.cap_plate_1", pl((CX - 20, 210), (CX + 20, 210)))
fig.add("fig_1.cap_plate_2", pl((CX - 20, 222), (CX + 20, 222)))
fig.add("fig_1.lbl_c", {"kind": "text", "text": "C", "text_height": 22,
                        "bbox": [CX + 28, 199, r1(CX + 28 + tw("C", 22)), 221]})
fig.add("fig_1.w_cap_bot", pl((CX, 222), (CX, BOTW)))
fig.add("fig_1.w_gnd", pl((SX, SY + SR), (SX, BOTW), (CX, BOTW)))
GX = 800
fig.add("fig_1.gnd", pl((GX, BOTW), (GX, BOTW + 10)))
fig.add("fig_1.gnd_1", pl((GX - 14, BOTW + 10), (GX + 14, BOTW + 10)))
fig.add("fig_1.gnd_2", pl((GX - 9, BOTW + 16), (GX + 9, BOTW + 16)))
fig.add("fig_1.gnd_3", pl((GX - 4, BOTW + 22), (GX + 4, BOTW + 22)))
fig.add("fig_1.node", {"kind": "ellipse", "bbox": ellipse(NODE, TOPW, 2.5)})
fig.add("fig_1.w_out", pl((NODE, TOPW), (965, TOPW)))
fig.add("fig_1.term", {"kind": "ellipse", "bbox": ellipse(970, TOPW, 5)})
lab = Ctx("fig_1.lbl_vout")
Row(*V("out")).emit(lab, 984, TOPW + 8, 22)
for i in lab.order:
    fig.add(i, lab.items[i], **lab.meta[i])
fig.add("fig_1.arrow_i", {"kind": "arrow", "points": [[712, 136], [742, 136]], "head": 7})
fig.add("fig_1.lbl_i", {"kind": "text", "text": "i", "text_height": 20,
                        "bbox": [722, 108, r1(722 + tw("i", 20)), 128]})
for it in fig.items.values():
    it.setdefault("writer", W)
fb = [ink_box(i, fig.items[i], fig.meta[i]) for i in fig.order]
el["fig_1"] = {"bbox": [r1(min(b[0] for b in fb) - 5), r1(min(b[1] for b in fb) - 5),
                        r1(max(b[2] for b in fb) + 5), r1(max(b[3] for b in fb) + 5)],
               "writer": W, "items": fig.items}

schematic = {
    "id": "fig_1", "type": "schematic",
    "components": [
        {"id": "V1", "type": "voltage_source", "label": "V_in", "value": "5 V step at t = 0",
         "ports": ["p", "n"], "drawn_by": ["fig_1.src", "fig_1.src_plus", "fig_1.src_minus"]},
        {"id": "R1", "type": "resistor", "label": "R", "value": "10 kΩ", "ports": ["a", "b"],
         "drawn_by": ["fig_1.res"]},
        {"id": "C1", "type": "capacitor", "label": "C", "value": "100 nF", "ports": ["a", "b"],
         "drawn_by": ["fig_1.cap_plate_1", "fig_1.cap_plate_2"]},
        {"id": "GND", "type": "ground", "ports": ["g"],
         "drawn_by": ["fig_1.gnd", "fig_1.gnd_1", "fig_1.gnd_2", "fig_1.gnd_3"]},
        {"id": "OUT", "type": "terminal", "label": "V_out", "ports": ["t"], "drawn_by": ["fig_1.term"]},
    ],
    "nets": [
        {"id": "in", "connects": ["V1.p", "R1.a"], "drawn_by": ["fig_1.w_in"]},
        {"id": "out", "connects": ["R1.b", "C1.a", "OUT.t"],
         "drawn_by": ["fig_1.w_mid", "fig_1.w_cap_top", "fig_1.node", "fig_1.w_out"]},
        {"id": "gnd", "connects": ["V1.n", "C1.b", "GND.g"], "drawn_by": ["fig_1.w_cap_bot", "fig_1.w_gnd"]},
    ],
    "labels": [
        {"id": "lbl_vin", "text": "V_in", "attached_to": "V1", "drawn_by": [i for i in fig.order if ".lbl_vin." in i]},
        {"id": "lbl_r", "text": "R", "attached_to": "R1", "drawn_by": ["fig_1.lbl_r"]},
        {"id": "lbl_c", "text": "C", "attached_to": "C1", "drawn_by": ["fig_1.lbl_c"]},
        {"id": "lbl_vout", "text": "V_out", "attached_to": "net:out", "drawn_by": [i for i in fig.order if ".lbl_vout." in i]},
        {"id": "lbl_i", "text": "i", "attached_to": "net:in", "meaning": "loop current, arrow shows direction",
         "drawn_by": ["fig_1.arrow_i", "fig_1.lbl_i"]},
    ],
}

maths = []
for eid, latex, plain, defines, cont, _ in math_src:
    m = {"id": eid, "type": "math", "latex": latex, "text": plain, "defines": defines,
         "layout": "2d_sketch", "items": list(ctxs[eid].order)}
    if cont:
        m["continues"] = cont
    if eid == "eq_9":
        m["boxed"] = True
        m["numeric_tolerance"] = {"relative": 2e-3, "note": "159.15 Hz written as 159 Hz"}
    maths.append(m)

content = {
    "schema": "ink_tokenizer.synthetic_page.content/v0",
    "page_id": "page_0004",
    "topic": "circuits: RC low-pass filter, step response and cutoff frequency",
    "provenance": {"generator": "hand-produced by LLM agent (Claude), scripts/make_page_0004.py",
                   "automated": False},
    "elements": [title, givens, schematic, sec_1, *maths[:4], sec_2, *maths[4:]],
}

order = ["title", "givens"] + fig.order + ["sec_1"]
for eid in ("eq_1", "eq_2", "eq_3", "eq_4"):
    order += ctxs[eid].order
order += ["sec_2"]
for eid in ("eq_5", "eq_6", "eq_7", "eq_8", "eq_9"):
    order += ctxs[eid].order

layout = {
    "schema": "ink_tokenizer.synthetic_page.layout/v0",
    "page_id": "page_0004",
    "unit": "mm",
    "page": {"width": PAGE_W, "height": PAGE_H, "origin": "top-left", "y_axis": "down",
             "surface": "whiteboard"},
    "writers": [{"id": W, "style": None}],
    "text_metrics": {"char_advance_per_height": CHAR_W,
                     "note": "text boxes here are sized with per-character advances measured from "
                             "the Hershey fonts (max over futural, cursive, scripts, timesi), not 0.55"},
    "math_metrics": {"superscript_scale": SUP_SCALE, "superscript_raise": SUP_RAISE,
                     "subscript_scale": SUB_SCALE, "subscript_drop": SUB_DROP,
                     "fraction_bar_above_baseline": AXIS,
                     "note": "fractions of the base text_height; text box bottom is the descender "
                             "line, baseline is 0.25 text_height above it"},
    "elements": el,
    "writing_order": order,
}

# ------------------------------------------------------------- layout checks


def overlap(a, b, gap=0.0):
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


def check_layout():
    meta = {}
    for c in list(ctxs.values()) + [fig]:
        meta.update(c.meta)
    items = {}
    for e in el.values():
        items.update(e.get("items", {}))
    # coverage: math items exist, writing order lists every leaf exactly once
    for m in maths:
        assert m["items"] and all(i in el[m["id"]]["items"] for i in m["items"]), m["id"]
        assert set(m["items"]) == set(el[m["id"]]["items"]), m["id"]
        assert all(i.startswith(m["id"] + ".") for i in m["items"])
    for c in schematic["components"] + schematic["nets"] + schematic["labels"]:
        assert all(i in el["fig_1"]["items"] for i in c["drawn_by"]), c["id"]
    drawn = {i for c in schematic["components"] + schematic["nets"] + schematic["labels"] for i in c["drawn_by"]}
    assert drawn == set(el["fig_1"]["items"]), set(el["fig_1"]["items"]) ^ drawn
    leaves = {k for k, e in el.items() if "items" not in e} | set(items)
    assert len(order) == len(set(order)) and set(order) == leaves, set(order) ^ leaves
    content_ids = {e["id"] for e in content["elements"]}
    assert set(el) == content_ids, set(el) ^ content_ids

    # characters
    texts = [content_el["text"] for content_el in (title, givens, sec_1, sec_2)]
    texts += [it["text"] for it in items.values() if it["kind"] == "text"]
    bad = {ch for s in texts for ch in s if ch not in SUPPORTED}
    assert not bad, bad

    # fraction bars span numerator and denominator; radical bars span radicands
    n_frac = n_sup = n_sub = 0
    for iid, it in items.items():
        mt = meta.get(iid, {})
        if it.get("role") == "fraction_bar":
            (x0, y), (x1, _) = it["points"]
            for part, below in ((mt["num"], False), (mt["den"], True)):
                assert part
                for p in part:
                    ib = ink_box(p, items[p], meta[p])
                    assert x0 <= ib[0] and ib[2] <= x1, (iid, p)
                    assert (ib[1] > y + 1) if below else (ib[3] < y - 1), (iid, p, ib, y)
            n_frac += 1
        if it.get("role") == "radical_bar":
            (x0, y), (x1, _) = it["points"]
            sign = meta[mt["sign"]]["ink"]
            assert abs(sign[2] - x0) < 0.5 and abs(sign[1] - y) < 0.5, iid
            for p in mt["body"]:
                b = ink_box(p, items[p], meta[p])
                assert x0 <= b[0] and b[2] <= x1 and b[1] > y + 1, (iid, p)
        # superscripts above the base's middle, subscripts below its baseline
        if it.get("role") in ("superscript", "subscript"):
            base = items[mt["base"]]
            bB, bh = mt["base_baseline"], mt["base_h"]
            assert abs((base["bbox"][3] - 0.25 * base["text_height"]) - bB) < 0.2, iid
            assert abs(it["bbox"][0] - base["bbox"][2]) < 0.2, iid  # attached to its base
            if it["role"] == "superscript":
                assert mt["baseline"] < bB - 0.375 * bh, iid
                assert it["text_height"] < 0.7 * bh
                n_sup += 1
            else:
                assert mt["baseline"] > bB and it["text_height"] < 0.7 * bh, iid
                assert it["bbox"][1] > bB - 0.75 * bh, iid  # top below the base's cap line
                n_sub += 1
    assert n_frac >= 8 and n_sup >= 2 and n_sub >= 10, (n_frac, n_sup, n_sub)

    # '=' columns line up within each aligned group
    for group in (left, right):
        xs = {c.items[f"{c.prefix}.eq"]["bbox"][0] for c in group}
        assert len(xs) == 1, xs

    # page bounds and overlaps, on approximate ink boxes
    boxes = {}
    for k, e in el.items():
        if "items" not in e:
            b = list(e["bbox"])
            boxes[k] = b
        for iid, it in e.get("items", {}).items():
            boxes[iid] = ink_box(iid, it, meta.get(iid, {}))
    for k, b in boxes.items():
        assert 15 <= b[0] and b[2] <= PAGE_W - 15 and 15 <= b[1] and b[3] <= PAGE_H - 15, (k, b)
    # the result box encloses eq_9 and nothing else
    rb = items["eq_9.box"]["bbox"]
    for k, b in boxes.items():
        inside = rb[0] < b[0] and b[2] < rb[2] and rb[1] < b[1] and b[3] < rb[3]
        assert inside == (k.startswith("eq_9.") and k != "eq_9.box"), k
    group = lambda k: k.split(".")[0]
    is_text = lambda k: k not in items or items[k]["kind"] == "text"
    keys = [k for k in boxes if k != "eq_9.box"]
    allowed = {frozenset(("fig_1.src", "fig_1.src_plus")), frozenset(("fig_1.src", "fig_1.src_minus"))}
    for iid, mt in meta.items():
        if items.get(iid, {}).get("role") == "radical_bar":
            allowed.add(frozenset((iid, mt["sign"])))
    n_pairs = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if frozenset((a, b)) in allowed:
                continue
            if group(a) == group(b) == "fig_1" and not (is_text(a) or is_text(b)):
                continue  # schematic wires meet their components by design
            gap = 2.0 if group(a) != group(b) else 0.0
            if group(a) != group(b) and not (is_text(a) and is_text(b)):
                gap = 4.0
            assert not overlap(boxes[a], boxes[b], gap), (a, boxes[a], b, boxes[b])
            n_pairs += 1
    # different elements keep clear of each other (element boxes, not items)
    tops = list(el)
    for i, a in enumerate(tops):
        for b in tops[i + 1:]:
            assert not overlap(el[a]["bbox"], el[b]["bbox"]), (a, b)
    return len(items), n_frac, n_sup, n_sub, n_pairs


if __name__ == "__main__":
    n_math = check_math()
    n_items, n_frac, n_sup, n_sub, n_pairs = check_layout()
    os.makedirs(OUT, exist_ok=True)
    for name, obj in (("content.json", content), ("layout.json", layout)):
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
    print(f"math OK ({n_math} equation evaluations + RK4 cross-checks); layout OK "
          f"({n_items} sketch items, {n_frac} fraction bars, {n_sup} superscripts, "
          f"{n_sub} subscripts, {n_pairs} item pairs overlap-checked); written to {OUT}")
