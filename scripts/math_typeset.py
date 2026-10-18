"""A small TeX-like typesetter for handwritten 2-D math.

typeset(latex, h, font, options) lays out a LaTeX subset with the metrics of a
fonts.TextRenderer or fonts.PoolMetrics and returns an Equation: width,
ascent and descent (mm, about the baseline) and the atoms to draw, in the
order a person writes them by hand. Coordinates are mm, x right from the
equation's left edge, y up from its baseline:

- {"type": "char", "ch", "x", "y", "size", "src"}: one font character, its
  pen position x, its baseline y and its text height; src is the offset of
  its token in the LaTeX string.
- {"type": "strokes", "part", "strokes", "src"}: clean polylines for
  geometry (frac_bar, radical, radical_bar, overline, underline, vec_arrow,
  hat, dot, paren_left/right, bracket_left/right, vbar_left/right,
  brace_left/right, integral, sum, prod, box). The renderer adds the noise.

Supported subset: {} groups; ^ and _ (single token or group, nested,
stacked when both); \\frac \\dfrac \\tfrac; \\sqrt[n]{x}; the big operators
in BIG_OPS with limits; \\left \\right with ( ) [ ] | \\| . \\{ \\} \\langle
\\rangle \\lfloor \\rfloor \\lceil \\rceil; \\begin{matrix} ... & ... \\\\ ...
\\end{matrix}; \\boxed, \\overline, \\underline, \\bar, \\vec, \\hat, \\dot,
\\tilde; \\not; \\mathbb (capitals); \\mathrm \\text \\textrm \\operatorname
(and \\mathbf \\mathit \\mathsf, drawn plain); function names (\\sin, \\lim,
...); Greek letters; the relation, operator and misc symbols in SYMBOLS;
spacing \\, \\: \\; \\! \\quad \\qquad \\  and ~. Spacing between atoms
follows TeX's atom classes (relations, binary operators, unary minus,
punctuation). Anything else, and any character the font lacks, raises
ValueError, so this doubles as the validator for generated LaTeX.

Plain delimiters (no \\left/\\right) around a fraction, matrix, big operator
or radical of one are drawn as tall as what they enclose, as people write
them; MathWriting's labels never use \\left and \\right.

Options: frac_order "num_bar_den" (default) or "bar_num_den"; radical_bar
"joined" (default: the radical sign and its overbar are one stroke written
before the radicand) or "after" (sign, radicand, then the bar).

Vertical metrics are per character class rather than per font, so a layout
measured with PoolMetrics bounds the ink of every font in the pool.
"""
import math

SCRIPT_SCALE, SCRIPT_MIN = 0.62, 0.45  # script size: x parent, min x h
FRAC_SCALE, FRAC_NESTED, FRAC_MIN = 0.85, 0.7, 0.5
SUP_RAISE, SUB_DROP = 0.42, 0.18       # x parent size
THIN, MED, THICK, PUNCT = 0.12, 0.2, 0.28, 0.15  # inter-atom spaces, x size

GREEK_NAMES = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π", "rho": "ρ",
    "varrho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω", "varpi": "ϖ", "varsigma": "ς"}
SYMBOLS = {
    "cdot": "·", "times": "×", "pm": "±", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
    "ne": "≠", "neq": "≠", "approx": "≈", "infty": "∞", "to": "→", "rightarrow": "→",
    "leftarrow": "←", "gets": "←", "Rightarrow": "⇒", "implies": "⇒", "leftrightarrow": "↔",
    "oplus": "⊕", "in": "∈", "partial": "∂", "circ": "°", "degree": "°", "%": "%", "{": "{",
    "}": "}", "#": "#", "&": "&", "$": "$", "_": "_", "sim": "~", "ast": "*", "prime": "'",
    "colon": ":", "mid": "|", "vert": "|", "lt": "<", "gt": ">", "neg": "¬", "lnot": "¬",
    "equiv": "≡", "notin": "∉", "ni": "∋", "subset": "⊂", "supset": "⊃", "subseteq": "⊆",
    "supseteq": "⊇", "subsetneq": "⊊", "sqsubseteq": "⊑", "mapsto": "↦", "longrightarrow": "⟶",
    "Leftrightarrow": "⇔", "iff": "⟺", "hookrightarrow": "↪", "rightleftharpoons": "⇌",
    "cong": "≅", "simeq": "≃", "triangleq": "≜", "propto": "∝", "perp": "⊥", "top": "⊤",
    "vdash": "⊢", "Vdash": "⊩", "models": "⊨", "ll": "≪", "gg": "≫", "otimes": "⊗", "odot": "⊙",
    "ominus": "⊖", "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨", "cap": "∩", "cup": "∪",
    "mp": "∓", "div": "÷", "bullet": "∙", "dagger": "†", "triangleleft": "◁", "triangle": "△",
    "bigcirc": "◯", "angle": "∠", "nabla": "∇", "forall": "∀", "exists": "∃", "emptyset": "∅",
    "varnothing": "∅", "aleph": "ℵ", "hbar": "ℏ", "vdots": "⋮", "backslash": "\\", "|": "‖",
    "Vert": "‖", "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈",
    "rceil": "⌉"}
DOTS = {"ldots": ".", "dots": ".", "cdots": "·"}
BIG_OPS = {"int": "integral", "oint": "oint", "iint": "iint", "sum": "sum", "prod": "prod",
           "bigcup": "bigcup", "bigcap": "bigcap", "bigoplus": "bigoplus", "bigwedge": "bigwedge",
           "bigvee": "bigvee"}
INTEGRALS = {"integral", "oint", "iint"}  # limits at the side by default
MATHBB = {c: {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}.get(c, chr(0x1D538 + i))
          for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
SPACES = {",": 3 / 18, ":": 4 / 18, ";": 5 / 18, "!": -3 / 18, " ": 1 / 3, "quad": 1.0, "qquad": 2.0}
FUNCTIONS = {"sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "arcsin", "arccos",
             "arctan", "log", "ln", "lg", "exp", "arg", "det", "deg", "dim", "ker", "gcd", "Pr"}
LIMIT_FUNCTIONS = {"lim", "max", "min", "sup", "inf"}
ACCENTS = {"overline": "overline", "bar": "overline", "underline": "underline", "vec": "vec_arrow",
           "overrightarrow": "vec_arrow", "hat": "hat", "widehat": "hat", "dot": "dot", "tilde": "tilde",
           "widetilde": "tilde"}
TEXT_CMDS = {"text", "textrm", "mbox"}
ROMAN_CMDS = {"mathrm", "mathbf", "mathit", "mathsf", "boldsymbol", "operatorname"}
NOOPS = {"displaystyle", "textstyle", "limits", "nolimits"}  # handled where they matter
UNICODE = {"−": "-", "⋅": "·", "∙": "·", "µ": "μ", "Ω": "Ω", "∆": "Δ", "…": "...", "⋯": "···",
           " ": " "}
DELIMS = {"(": "paren", ")": "paren", "[": "bracket", "]": "bracket", "|": "vbar", "{": "brace",
          "}": "brace", "‖": "dvbar", "⟨": "angle", "⟩": "angle", "⌊": "floor", "⌋": "floor",
          "⌈": "ceil", "⌉": "ceil", ".": None}
PAIRS = {"(": ")", "[": "]", "{": "}", "⟨": "⟩", "⌊": "⌋", "⌈": "⌉", "|": "|", "‖": "‖"}

REL = set("=<>≤≥≠≈→←⇒↔∈~:≡∉∋⊂⊃⊆⊇⊊⊑↦⟶⇔⟺↪⇌≅≃≜∝⊥⊢⊩⊨≪≫")
BIN = set("+-±×·⊕*⊗⊙⊖∧∨∩∪∓÷∙◁")
OPEN, CLOSE, PUNCT_CH = set("([{⟨⌊⌈"), set(")]}!⟩⌋⌉"), set(",;")
# Vertical extents by character class (x size): the ink of every pool font
# stays within these, plus a little noise.
NO_DESC = set("!\"%&'*+-.0123456789:<=>?ABCDEFIMOPSTVW^`abcehilmnorstuvw~αδεθικλνοπστυω"
              "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ⊕→≤≥≠±·×∞√≈∑∂∆°←↔⇒∈¬"
              "≡∉∋⊂⊃⊆⊇⊊⊑↦⟶⇔⟺↪⇌≅≃≜∝⊥⊤⊢⊩⊨≪≫⊗⊙⊖∧∨∩∪∓÷∙◁△◯∠∇∀∃ℵ⋮ϖ") | set(MATHBB.values())
DEEP = set("fgjpqyzGJYZ")
X_LOW = set("acegmnopqrsuvwxyz")
GREEK_LOW = set("αγεηικμνοπρστυφχως")
TALL = set("()[]{}|/\\#$‖⟨⟩⌊⌋⌈⌉")
ASCENDER = set("bdfhklβδζθλξψ\"*^`ℏ")
MID = {"+": 0.68, "±": 0.72, "-": 0.6, "=": 0.6, "~": 0.6, "≈": 0.6, "·": 0.5, "×": 0.6, "→": 0.6,
       "←": 0.6, "↔": 0.6, "⇒": 0.62, ":": 0.6, ";": 0.6, ",": 0.22, ".": 0.22, "_": 0.1, "¬": 0.6,
       "≡": 0.6, "∉": 0.72, "∋": 0.6, "⊂": 0.6, "⊃": 0.6, "⊆": 0.62, "⊇": 0.62, "⊊": 0.62, "⊑": 0.62,
       "↦": 0.6, "⟶": 0.6, "⇔": 0.62, "⟺": 0.62, "↪": 0.62, "⇌": 0.62, "≅": 0.62, "≃": 0.6,
       "≜": 0.75, "∝": 0.6, "≪": 0.6, "≫": 0.6, "∧": 0.62, "∨": 0.62, "∩": 0.62, "∪": 0.62,
       "∓": 0.6, "÷": 0.6, "∙": 0.5, "◁": 0.62, "ϖ": 0.62}


class Equation:
    """A typeset equation: width, ascent, descent (mm) and atoms in writing order."""

    def __init__(self, width, ascent, descent, atoms):
        self.width, self.ascent, self.descent, self.atoms = width, ascent, descent, atoms

    def __repr__(self):
        return f"Equation(width={self.width:.2f}, ascent={self.ascent:.2f}, descent={self.descent:.2f}, " \
               f"{len(self.atoms)} atoms)"


def typeset(latex, h, font, options=None):
    """Typeset `latex` at base text height h (mm) with `font`'s metrics."""
    return _Typesetter(latex, h, font, options or {}).run()


# ------------------------------------------------------------------ parse
# Nodes are tuples: (kind, ...). src fields are offsets into the LaTeX string.

class _Parser:
    def __init__(self, s):
        self.s, self.i = s, 0

    def error(self, msg, at=None):
        at = self.i if at is None else at
        return ValueError(f"{msg} at offset {at} in {self.s!r}")

    def skip(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\n":
            self.i += 1

    def peek(self):
        self.skip()
        return self.s[self.i] if self.i < len(self.s) else None

    def command(self):
        """Read the command name after a backslash at self.i."""
        j = self.i + 1
        if j >= len(self.s):
            raise self.error("trailing backslash")
        if self.s[j].isalpha():
            k = j
            while k < len(self.s) and self.s[k].isalpha():
                k += 1
        else:
            k = j + 1
        self.i = k
        return self.s[j:k]

    def parse(self):
        nodes = self.list(None)
        if self.i < len(self.s):
            raise self.error(f"unexpected {self.s[self.i]!r}")
        return nodes

    def at_command(self, name):
        n = len(name) + 1
        return self.s.startswith("\\" + name, self.i) and not self.s[self.i + n:self.i + n + 1].isalpha()

    def list(self, end):
        """Nodes up to `end`: '}', ']', 'right', 'matrix' (the end of a matrix
        cell: '&', '\\\\' or '\\end') or None (end of input)."""
        nodes = []
        while True:
            c = self.peek()
            if c is None:
                if end is not None:
                    raise self.error(f"missing {({'right': '\\right', 'matrix': '\\end{matrix}'}).get(end, repr(end))}")
                return _tall_delims(nodes)
            if c == end:
                return _tall_delims(nodes)
            if c == "}":
                raise self.error("unbalanced '}'")
            if end == "right" and self.at_command("right"):
                return _tall_delims(nodes)
            if end == "matrix" and (c == "&" or self.s.startswith("\\\\", self.i) or self.at_command("end")):
                return _tall_delims(nodes)
            if c in "^_":
                base = nodes.pop() if nodes and nodes[-1][0] != "space" else ("group", [])
                nodes.append(self.scripts(base))
                continue
            node = self.atom()
            if node is not None:
                nodes.append(node)

    def scripts(self, base):
        sup = sub = None
        order = []
        while self.peek() in ("^", "_"):
            c, at = self.s[self.i], self.i
            self.i += 1
            arg = self.arg()
            if c == "^":
                if sup is not None:
                    raise self.error("double superscript", at)
                sup = arg
            else:
                if sub is not None:
                    raise self.error("double subscript", at)
                sub = arg
            order.append(c)
        if base[0] == "scripts":
            raise self.error("scripts on a scripted base; use braces")
        return ("scripts", base, sup, sub, order)

    def arg(self):
        """A command or script argument: a group or a single token."""
        c = self.peek()
        if c is None:
            raise self.error("missing argument")
        if c == "{":
            self.i += 1
            nodes = self.list("}")
            self.i += 1
            return ("group", nodes)
        if c in "}^_":
            raise self.error(f"missing argument before {c!r}")
        node = self.atom()
        return node if node is not None else ("group", [])

    def atom(self):
        """One node starting at self.i (spaces already skipped); None for no-ops."""
        s, at = self.s, self.i
        c = s[at]
        if c == "{":
            self.i += 1
            nodes = self.list("}")
            self.i += 1
            return ("group", nodes)
        if c == "\\":
            return self.cmd(at)
        self.i += 1
        if c == "~":
            return ("space", SPACES[" "])
        if c in "$&#%":
            raise self.error(f"unsupported character {c!r} (escape it or remove it)", at)
        c = UNICODE.get(c, c)
        if c == "∫":
            return ("bigop", "integral", at, False)
        if c == "∑":
            return ("bigop", "sum", at, True)
        if len(c) > 1:
            return ("group", [("char", ch, at) for ch in c])
        return ("char", c, at)

    def cmd(self, at):
        name = self.command()
        if name in SPACES:
            return ("space", SPACES[name])
        if name in GREEK_NAMES:
            return ("char", GREEK_NAMES[name], at)
        if name in SYMBOLS:
            return ("char", SYMBOLS[name], at)
        if name in DOTS:
            return ("group", [("char", DOTS[name], at)] * 3, "inner")
        if name in ("frac", "dfrac", "tfrac"):
            return ("frac", self.arg(), self.arg(), at)
        if name == "sqrt":
            index = None
            if self.peek() == "[":
                self.i += 1
                index = ("group", self.list("]"))
                self.i += 1
            return ("sqrt", self.arg(), index, at)
        if name in BIG_OPS:
            kind = BIG_OPS[name]
            limits = kind not in INTEGRALS
            while self.s.startswith("\\limits", self.i) or self.s.startswith("\\nolimits", self.i):
                limits = self.command() == "limits"
                self.skip()
            return ("bigop", kind, at, limits)
        if name in FUNCTIONS or name in LIMIT_FUNCTIONS:
            limits = name in LIMIT_FUNCTIONS
            self.skip()
            while self.s.startswith("\\limits", self.i) or self.s.startswith("\\nolimits", self.i):
                limits = self.command() == "limits"
                self.skip()
            return ("opname", [(ch, at) for ch in name], limits)
        if name == "operatorname":
            body = self.arg()
            return ("opname", [(n[1], n[2]) for n in _flatten_chars(body, self)], False)
        if name == "mathbb":
            return ("group", [("char", MATHBB.get(n[1], n[1]), n[2]) for n in _flatten_chars(self.arg(), self)])
        if name in ROMAN_CMDS:
            return ("group", [self.arg()])
        if name == "not":
            return ("not", self.arg(), at)
        if name == "begin":
            return self.matrix(at)
        if name in TEXT_CMDS:
            return ("text", self.text_arg())
        if name in ACCENTS:
            return ("accent", ACCENTS[name], self.arg(), at)
        if name == "boxed":
            return ("boxed", self.arg(), at)
        if name == "left":
            ld, lsrc = self.delim()
            body = self.list("right")
            self.command()  # \right
            rd, rsrc = self.delim()
            return ("leftright", ld, lsrc, ("group", body), rd, rsrc)
        if name == "right":
            raise self.error("\\right without \\left", at)
        if name in NOOPS:
            return None
        raise self.error(f"unsupported command \\{name}", at)

    def delim(self):
        c, at = self.peek(), self.i
        if c is None:
            raise self.error("missing delimiter")
        if c == "\\":
            name = self.command()
            ch = {"{": "{", "}": "}", "|": "‖", "Vert": "‖", "vert": "|", "lbrace": "{", "rbrace": "}",
                  "lbrack": "[", "rbrack": "]", "langle": "⟨", "rangle": "⟩", "lfloor": "⌊",
                  "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉"}.get(name)
            if ch is None:
                raise self.error(f"unsupported delimiter \\{name}", at)
            return ch, at
        self.i += 1
        if c not in DELIMS or c in "{}":
            raise self.error(f"unsupported delimiter {c!r}", at)
        return c, at

    def env_name(self):
        if self.peek() != "{":
            raise self.error("missing environment name")
        j = self.s.find("}", self.i)
        if j < 0:
            raise self.error("unterminated environment name")
        name, self.i = self.s[self.i + 1:j], j + 1
        return name

    def matrix(self, at):
        """\\begin{matrix} (already read up to \\begin) through \\end{matrix}:
        ("matrix", rows of cell groups, src)."""
        name = self.env_name()
        if name != "matrix":
            raise self.error(f"unsupported environment {name}", at)
        rows, row = [], []
        while True:
            row.append(("group", self.list("matrix")))
            if self.s.startswith("&", self.i):
                self.i += 1
                continue
            rows.append(row)
            row = []
            if self.s.startswith("\\\\", self.i):
                self.i += 2
                continue
            self.command()  # \end
            if self.env_name() != "matrix":
                raise self.error("\\begin{matrix} ended by another environment", at)
            break
        if len(rows) > 1 and rows[-1] == [("group", [])]:  # a trailing \\
            rows.pop()
        return ("matrix", rows, at)

    def text_arg(self):
        """Raw characters of a {...} text argument, spaces kept."""
        if self.peek() != "{":
            raise self.error("\\text needs a {...} argument")
        self.i += 1
        out, depth = [], 0
        while True:
            if self.i >= len(self.s):
                raise self.error("unterminated \\text{")
            c, at = self.s[self.i], self.i
            if c == "}" and depth == 0:
                self.i += 1
                return out
            if c == "\\":
                name = self.command()
                if name in SYMBOLS or name in GREEK_NAMES:
                    out.append((SYMBOLS.get(name) or GREEK_NAMES[name], at))
                elif name in (" ", ",", ";", ":"):
                    out.append((" ", at))
                else:
                    raise self.error(f"unsupported command \\{name} in text", at)
                continue
            self.i += 1
            if c in "{}":
                depth += 1 if c == "{" else -1
                continue
            if c in "$&#%^_":
                raise self.error(f"unsupported character {c!r} in text", at)
            out.append((" " if c in "~\t\n" else UNICODE.get(c, c), at))


def _flatten_chars(node, parser):
    """The char nodes of a plain group (for \\operatorname)."""
    if node[0] == "char":
        return [node]
    if node[0] == "group":
        return [c for n in node[1] for c in _flatten_chars(n, parser)]
    if node[0] == "space":
        return []
    raise parser.error("\\operatorname and \\mathbb take plain characters")


def _tall(node):
    """Whether a node is drawn taller than a line of text."""
    kind = node[0]
    if kind in ("frac", "bigop"):
        return True
    if kind == "matrix":
        return len(node[1]) > 1
    if kind == "group":
        return any(_tall(n) for n in node[1])
    if kind in ("scripts", "sqrt", "not", "boxed"):
        return _tall(node[1])
    if kind == "accent":
        return _tall(node[2])
    if kind == "leftright":
        return _tall(node[3])
    return False


def _tall_delims(nodes):
    """Pair plain delimiters in one list, and turn each pair that encloses
    something tall into a \\left ... \\right node. A closing delimiter may
    carry scripts, as in (\\frac{a}{b})^{2}."""
    out, stack = [], []  # stack: (index in out, opening character)
    for n in nodes:
        closer = n if n[0] == "char" else n[1] if n[0] == "scripts" and n[1][0] == "char" else None
        opened = None
        if closer is not None:
            opened = next((k for k in reversed(range(len(stack))) if PAIRS[stack[k][1]] == closer[1]), None)
        if opened is not None:
            j, ch = stack[opened]
            del stack[opened:]
            body = out[j + 1:]
            if any(_tall(b) for b in body):
                pair = ("leftright", ch, out[j][2], ("group", body), closer[1], closer[2])
                out[j:] = [pair if n is closer else ("scripts", pair) + n[2:]]
                continue
        elif n[0] == "char" and n[1] in PAIRS:
            stack.append((len(out), n[1]))
        out.append(n)
    return out


# ----------------------------------------------------------------- boxes

class Box:
    """Laid-out material: width, ascent, descent (mm) and atoms relative to
    its left edge and baseline. cls is its TeX atom class for spacing."""

    def __init__(self, w=0.0, a=0.0, d=0.0, atoms=None, cls="ord", char=None, limits=None):
        self.w, self.a, self.d, self.atoms, self.cls = w, a, d, atoms or [], cls
        self.char = char      # the character, if this is a single char
        self.limits = limits  # "under" (sum, lim), "side" (integral) or None
        self.op = None        # the big operator, if this is a group holding just one


def _moved(atoms, dx, dy):
    out = []
    for at in atoms:
        if at["type"] == "char":
            out.append({**at, "x": at["x"] + dx, "y": at["y"] + dy})
        else:
            out.append({**at, "strokes": [[(x + dx, y + dy) for x, y in st] for st in at["strokes"]]})
    return out


def _strokes(part, strokes, src):
    return {"type": "strokes", "part": part, "strokes": strokes, "src": src}


def _bezier(p0, p1, p2, n=10):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / n for i in range(n + 1))]


def _cubic(p0, p1, p2, p3, n=10):
    return [tuple((1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t * t * c + t ** 3 * d
                  for a, b, c, d in zip(p0, p1, p2, p3)) for t in (i / n for i in range(n + 1))]


def _circle(cx, cy, r, n=24):
    return [(cx - r * math.sin(2 * math.pi * i / n), cy + r * math.cos(2 * math.pi * i / n)) for i in range(n + 1)]


def _spacing(left, right):
    """Inter-atom space (x size) between atom classes, after TeX's table."""
    if left is None or right is None:
        return 0.0
    if left == "punct":
        return PUNCT
    if left == "rel" and right == "rel":
        return 0.0
    if left == "rel" or right == "rel":
        return THICK if left != "open" and right not in ("close", "punct") else 0.0
    if left == "bin" or right == "bin":
        return MED
    if left == "open" or right in ("close", "punct"):
        return 0.0
    if left == "op" or right == "op":
        return THIN if right != "open" else 0.0
    if left == "inner" or right == "inner":
        return THIN
    return 0.0


class _Typesetter:
    def __init__(self, latex, h, font, options):
        unknown = set(options) - {"frac_order", "radical_bar"}
        if unknown:
            raise ValueError(f"unknown math options {sorted(unknown)}")
        self.frac_order = options.get("frac_order", "num_bar_den")
        self.radical_bar = options.get("radical_bar", "joined")
        if self.frac_order not in ("num_bar_den", "bar_num_den"):
            raise ValueError(f"unknown frac_order {self.frac_order!r}")
        if self.radical_bar not in ("joined", "after"):
            raise ValueError(f"unknown radical_bar {self.radical_bar!r}")
        if not h > 0:
            raise ValueError(f"text height must be positive, got {h!r}")
        self.latex, self.h, self.font = latex, h, font

    def run(self):
        box = self.hlist(_Parser(self.latex).parse(), self.h, False, False)
        atoms = box.atoms
        pts = [(at["x"], at["y"]) for at in atoms if at["type"] == "char"] + \
              [p for at in atoms if at["type"] == "strokes" for st in at["strokes"] for p in st]
        x0 = min([0.0] + [p[0] for p in pts])
        atoms = _moved(atoms, -x0, 0.0)
        width = box.w - x0
        sx = [p for at in atoms if at["type"] == "strokes" for st in at["strokes"] for p in st]
        width = max([width] + [x for x, _ in sx])
        asc = max([box.a] + [y for _, y in sx])
        desc = max([box.d] + [-y for _, y in sx])
        return Equation(width, asc, desc, atoms)

    # ---- sizes

    def script_size(self, s):
        return max(SCRIPT_SCALE * s, SCRIPT_MIN * self.h)

    def frac_size(self, s, in_frac):
        return min(s, max((FRAC_NESTED if in_frac else FRAC_SCALE) * s, FRAC_MIN * self.h))

    # ---- leaves

    def char(self, ch, s, src, cls=None):
        if not self.font.has(ch):
            raise ValueError(f"the font has no glyph for {ch!r} (offset {src} in {self.latex!r})")
        if ch in MID:
            a = MID[ch] * s
        elif ch in X_LOW:
            a = (self.font.x_height + 0.03) * s
        elif ch in GREEK_LOW:
            a = 0.53 * s
        elif ch in TALL:
            a = 0.93 * s
        elif ch in ASCENDER:
            a = 0.86 * s
        else:
            a = (1 - self.font.base) * s
        if ch in NO_DESC:
            d = 0.02 * s if ch not in ",;" else 0.18 * s
        elif ch in DEEP:
            d = 0.46 * s
        else:
            d = self.font.base * s
        if cls is None:
            cls = "rel" if ch in REL else "bin" if ch in BIN else "open" if ch in OPEN else \
                "close" if ch in CLOSE else "punct" if ch in PUNCT_CH else "ord"
        atom = {"type": "char", "ch": ch, "x": 0.0, "y": 0.0, "size": s, "src": src}
        return Box(self.font.advance(ch, s), a, d, [atom], cls, char=ch)

    def chars(self, pairs, s, cls="ord", keep_spaces=False):
        boxes = []
        for ch, src in pairs:
            if ch == " ":
                if keep_spaces:
                    boxes.append(Box(SPACES[" "] * s, cls=None))
                continue
            boxes.append(self.char(ch, s, src, "ord"))
        box = self.concat(boxes)
        box.cls = cls
        if len(boxes) == 1:
            box.char = boxes[0].char
        return box

    # ---- lists

    def concat(self, boxes):
        x, a, d, atoms = 0.0, 0.0, 0.0, []
        for b in boxes:
            atoms += _moved(b.atoms, x, 0.0)
            x += b.w
            a, d = max(a, b.a), max(d, b.d)
        return Box(x, a, d, atoms)

    def hlist(self, nodes, s, script, in_frac, cls_override=None):
        boxes = [self.node(n, s, script, in_frac) for n in nodes]
        # TeX's binary-operator rule: a + or - with nothing to bind on its left
        # (or right) is unary, i.e. an ordinary symbol.
        prev = None
        for b in boxes:
            if b.cls is None:
                continue
            if b.cls == "bin" and prev in (None, "bin", "op", "rel", "open", "punct"):
                b.cls = "ord"
            if b.cls in ("rel", "close", "punct") and prev == "bin":
                last = next(x for x in reversed(boxes[:boxes.index(b)]) if x.cls is not None)
                last.cls = "ord"
            prev = b.cls
        real = [b for b in boxes if b.cls is not None]
        if real and real[-1].cls == "bin":
            real[-1].cls = "ord"
        out, prev = [], None
        k = 0.6 if script else 1.0
        for b in boxes:
            if b.cls is not None:
                sp = _spacing(prev, b.cls)
                if sp:
                    out.append(Box((sp * k if sp in (MED, THICK) else sp) * s, cls=None))
                prev = b.cls
            out.append(b)
        box = self.concat(out)
        box.cls = cls_override or "ord"
        if len(real) == 1:
            box.cls = cls_override or real[0].cls
            box.char, box.limits = real[0].char, real[0].limits
            if real[0].limits and len(boxes) == 1:
                box.op = real[0]
        return box

    def node(self, n, s, script, in_frac):
        kind = n[0]
        if kind == "char":
            return self.char(n[1], s, n[2])
        if kind == "group":
            return self.hlist(n[1], s, script, in_frac, n[2] if len(n) > 2 else None)
        if kind == "space":
            return Box(n[1] * s, cls=None)
        if kind == "text":
            return self.chars(n[1], s, keep_spaces=True)
        if kind == "opname":
            box = self.chars(n[1], s, cls="op")
            box.char, box.limits = None, "under" if n[2] else None
            return box
        if kind == "scripts":
            return self.scripts(n, s, script, in_frac)
        if kind == "frac":
            return self.frac(n, s, in_frac)
        if kind == "sqrt":
            return self.sqrt(n, s, script, in_frac)
        if kind == "bigop":
            return self.bigop(n, s)
        if kind == "leftright":
            return self.leftright(n, s, script, in_frac)
        if kind == "accent":
            return self.accent(n, s, script, in_frac)
        if kind == "boxed":
            return self.boxed(n, s, script, in_frac)
        if kind == "matrix":
            return self.matrix(n, s, script, in_frac)
        if kind == "not":
            return self.negated(n, s, script, in_frac)
        raise ValueError(f"internal: unknown node {kind!r}")

    # ---- scripts and limits

    def scripts(self, n, s, script, in_frac):
        _, base_n, sup_n, sub_n, order = n
        # x^\circ is a degree sign at full size, not a raised ring.
        if sub_n is None and sup_n[0] == "char" and sup_n[1] == "°":
            return self.hlist([base_n, sup_n], s, script, in_frac)
        base = self.node(base_n, s, script, in_frac)
        ss = self.script_size(s)
        sup = self.node(sup_n, ss, True, in_frac) if sup_n is not None else None
        sub = self.node(sub_n, ss, True, in_frac) if sub_n is not None else None
        op = base.op or (base if base.limits else None)
        if op is not None and op.limits == "under":
            return self.limits_under(base, sup, sub, order, s)
        if op is not None and op.limits == "side":
            return self.limits_side(base, op, sup, sub, order, s)
        tall = base.char is None and base.atoms
        raise_ = max(SUP_RAISE * s, base.a - 0.5 * ss if tall else 0.0)
        drop = max(SUB_DROP * s, base.d - 0.4 * ss if tall else 0.0)
        if sup and sub:
            # keep a clear gap between the superscript's ink and the subscript's
            gap = (raise_ - sup.d) - (sub.a - drop)
            if gap < 0.12 * s:
                drop += 0.12 * s - gap
        if sup:
            raise_ = max(raise_, sup.d + 0.25 * s)
        kern = 0.06 * s
        x = base.w + kern
        atoms, w, a, d = list(base.atoms), base.w, base.a, base.d
        for c in order:
            b, dy = (sup, raise_) if c == "^" else (sub, -drop)
            atoms += _moved(b.atoms, x, dy)
            w = max(w, x + b.w)
            a, d = max(a, b.a + dy), max(d, b.d - dy)
        return Box(w, a, d, atoms, base.cls)

    def limits_under(self, base, sup, sub, order, s):
        """Limits centred below and above (sum, prod, lim)."""
        gap = 0.12 * s
        w = max([base.w] + [b.w for b in (sup, sub) if b])
        atoms = _moved(base.atoms, (w - base.w) / 2, 0.0)
        a, d = base.a, base.d
        for c in sorted(order, key=lambda c: c != "_"):  # lower limit first
            if c == "_":
                y = -base.d - gap - sub.a
                atoms += _moved(sub.atoms, (w - sub.w) / 2, y)
                d = max(d, sub.d - y)
            else:
                y = base.a + gap + sup.d
                atoms += _moved(sup.atoms, (w - sup.w) / 2, y)
                a = max(a, sup.a + y)
        return Box(w, a, d, atoms, "op")

    def limits_side(self, base, op, sup, sub, order, s):
        """Integral limits at the top right and bottom right of the sign."""
        top, bot, right = op.ext
        atoms, w, a, d = list(base.atoms), base.w, base.a, base.d
        for c in sorted(order, key=lambda c: c != "_"):  # lower limit first
            if c == "_":
                x, y = op.lower_x, bot
                atoms += _moved(sub.atoms, x, y)
                w, d = max(w, x + sub.w), max(d, sub.d - y)
            else:
                x, y = right + 0.06 * s, top - sup.a
                atoms += _moved(sup.atoms, x, y)
                w, a = max(w, x + sup.w), max(a, sup.a + y)
        return Box(w + 0.04 * s, a, d, atoms, "op")

    # ---- fractions, radicals

    def frac(self, n, s, in_frac):
        _, num_n, den_n, src = n
        fs = self.frac_size(s, in_frac)
        num = self.node(num_n, fs, False, True)
        den = self.node(den_n, fs, False, True)
        axis, gap, over, margin = self.font.axis * s, 0.15 * s, 0.1 * s, 0.06 * s
        bar_w = max(num.w, den.w) + 2 * over
        w = bar_w + 2 * margin
        yn = axis + gap + num.d
        yd = axis - gap - den.a
        num_atoms = _moved(num.atoms, (w - num.w) / 2, yn)
        den_atoms = _moved(den.atoms, (w - den.w) / 2, yd)
        bar = _strokes("frac_bar", [[(margin, axis), (margin + bar_w, axis)]], src)
        atoms = num_atoms + [bar] + den_atoms if self.frac_order == "num_bar_den" else \
            [bar] + num_atoms + den_atoms
        return Box(w, yn + num.a, den.d - yd, atoms, "inner")

    def sqrt(self, n, s, script, in_frac):
        _, body_n, index_n, src = n
        body = self.node(body_n, s, script, in_frac)
        top = max(body.a, 0.5 * s) + 0.14 * s
        bot = -(max(body.d, 0.0) + 0.08 * s)
        H = top - bot
        k = min(H, 1.2 * s)
        p0 = (0.0, bot + 0.32 * k)
        p1 = (0.1 * s, bot + 0.4 * k)
        p2 = (0.26 * s, bot)
        p3 = (p2[0] + 0.1 * s + 0.12 * H, top)
        index, shift = None, 0.0
        if index_n is not None:
            index = self.node(index_n, max(0.5 * s, SCRIPT_MIN * self.h), True, in_frac)
            room = p2[0] - 0.04 * s
            shift = max(0.0, index.w - room)
        p0, p1, p2, p3 = [(x + shift, y) for x, y in (p0, p1, p2, p3)]
        bx = p3[0] + 0.14 * s
        p4 = (bx + body.w + 0.1 * s, top)
        atoms = []
        if self.radical_bar == "joined":
            atoms.append(_strokes("radical", [[p0, p1, p2, p3, p4]], src))
        else:
            atoms.append(_strokes("radical", [[p0, p1, p2, p3]], src))
        a = top
        if index is not None:
            iy = p1[1] + 0.1 * s + index.d
            atoms += _moved(index.atoms, p2[0] - 0.04 * s - index.w, iy)
            a = max(a, iy + index.a)
        atoms += _moved(body.atoms, bx, 0.0)
        if self.radical_bar == "after":
            atoms.append(_strokes("radical_bar", [[p3, p4]], src))
        return Box(p4[0] + 0.04 * s, a, -bot, atoms, "ord")

    # ---- big operators

    def bigop(self, n, s):
        kind, src = n[1], n[2]
        ax = self.font.axis * s
        if kind in INTEGRALS:
            H, r = 2.0 * s, 0.13 * s
            top, bot = ax + H / 2, ax - H / 2
            xb = 2 * r + 0.06 * s          # stem bottom; the lower hook ends left of it
            xa = xb + 0.08 * H             # stem top
            ux, uy = (xb - xa) / H, -1.0   # stem direction (down), roughly unit
            A, B = (xa, top - 1.3 * r), (xb, bot + 1.3 * r)
            k = 1.2 * r
            pts = (_cubic((xa + 2 * r, top - 0.8 * r), (xa + 2 * r, top + 0.1 * r), (A[0] - k * ux, A[1] - k * uy), A)
                   + _cubic(B, (B[0] + k * ux, B[1] + k * uy), (xb - 2 * r, bot - 0.1 * r), (xb - 2 * r, bot + 0.8 * r))[1:])
            strokes, right = [pts], xa + 2 * r
            if kind == "iint":
                gap = 0.4 * s
                strokes.append([(x + gap, y) for x, y in pts])
                right, xb = right + gap, xb + gap
            elif kind == "oint":
                strokes.append(_circle((xa + xb) / 2, ax, 0.24 * s))
            box = Box(right + 0.06 * s, top + 0.03 * s, -bot + 0.03 * s, [_strokes(kind, strokes, src)],
                      "op", limits="under" if n[3] else "side")
            box.ext, box.lower_x = (top, bot, right), xb + 0.16 * s
            return box
        H, W = 1.5 * s, 1.0 * s
        top, bot = ax + H / 2, ax - H / 2
        m = 0.06 * s
        if kind == "sum":
            strokes = [[(m + W, top - 0.06 * s), (m + W, top), (m, top), (m + 0.48 * W, ax), (m, bot),
                        (m + W, bot), (m + W, bot + 0.06 * s)]]
        elif kind == "prod":
            strokes = [[(m, top), (m + W, top)], [(m + 0.18 * W, top), (m + 0.18 * W, bot)],
                       [(m + 0.82 * W, top), (m + 0.82 * W, bot)]]
        elif kind in ("bigcup", "bigcap"):
            r, cx = 0.42 * W, m + W / 2
            cup = [(cx - r, top)] + [(cx - r * math.cos(t), bot + r - r * math.sin(t))
                                     for t in (math.pi * i / 12 for i in range(13))] + [(cx + r, top)]
            strokes = [cup if kind == "bigcup" else [(x, top + bot - y) for x, y in cup]]
        elif kind == "bigoplus":
            r, cx = W / 2, m + W / 2
            strokes = [_circle(cx, ax, r), [(cx - r, ax), (cx + r, ax)], [(cx, ax + r), (cx, ax - r)]]
        else:  # bigwedge, bigvee
            wedge = [(m, bot), (m + W / 2, top), (m + W, bot)]
            strokes = [wedge if kind == "bigwedge" else [(x, top + bot - y) for x, y in wedge]]
        limits = "under" if n[3] else "side"
        box = Box(W + 2 * m, top, -bot, [_strokes(kind, strokes, src)], "op", limits=limits)
        box.ext, box.lower_x = (top, bot, W + m), W + 2 * m
        return box

    # ---- delimiters

    def leftright(self, n, s, script, in_frac):
        _, ld, lsrc, body_n, rd, rsrc = n
        body = self.node(body_n, s, script, in_frac)
        ax = self.font.axis * s
        small = body.a <= 1.0 * s and body.d <= 0.3 * s
        half = max(body.a - ax, body.d + ax) + 0.08 * s
        parts = []
        for ch, src, side in ((ld, lsrc, "left"), (rd, rsrc, "right")):
            if ch == ".":
                parts.append(Box(0.06 * s, cls=None))
            elif small:
                parts.append(self.char(ch, s, src))
            else:
                parts.append(self.big_delim(ch, side, ax + half, ax - half, s, src))
        box = self.concat([parts[0], body, parts[1]])
        box.cls = "inner"
        return box

    def big_delim(self, ch, side, top, bot, s, src):
        H, pad = top - bot, 0.08 * s
        kind = DELIMS[ch]
        if kind == "paren":
            bulge = 0.12 * s + 0.05 * H
            w = bulge + 2 * pad
            xa, xb = (w - pad, pad) if side == "left" else (pad, w - pad)
            ctrl = 2 * xb - xa  # the quadratic's control point, so the curve reaches xb
            strokes = [_bezier((xa, top), (ctrl * 0.5 + xb * 0.5, (top + bot) / 2), (xa, bot), 16)]
        elif kind == "bracket":
            arm = 0.18 * s
            w = arm + 2 * pad
            xa, xb = (w - pad, pad) if side == "left" else (pad, w - pad)
            strokes = [[(xa, top), (xb, top), (xb, bot), (xa, bot)]]
        elif kind in ("floor", "ceil"):
            arm = 0.18 * s
            w = arm + 2 * pad
            xa, xb = (w - pad, pad) if side == "left" else (pad, w - pad)
            strokes = [[(xb, top), (xb, bot), (xa, bot)] if kind == "floor" else [(xa, top), (xb, top), (xb, bot)]]
        elif kind == "angle":
            w = 0.16 * s + 0.08 * H + 2 * pad
            xa, xb = (w - pad, pad) if side == "left" else (pad, w - pad)
            strokes = [[(xa, top), (xb, (top + bot) / 2), (xa, bot)]]
        elif kind == "vbar":
            w = 0.1 * s + 2 * pad
            strokes = [[(w / 2, top), (w / 2, bot)]]
        elif kind == "dvbar":
            w = 0.22 * s + 2 * pad
            strokes = [[(x, top), (x, bot)] for x in (w / 2 - 0.07 * s, w / 2 + 0.07 * s)]
        else:  # brace
            dep = 0.22 * s
            w = dep + 2 * pad
            xa, xm, xb = (w - pad, pad + dep / 2, pad) if side == "left" else (pad, pad + dep / 2, w - pad)
            c = (top + bot) / 2
            q = 0.08 * H
            strokes = [_bezier((xa, top), (xm, top), (xm, top - q), 4)
                       + [(xm, c + q)] + _bezier((xm, c + q), (xm, c), (xb, c), 4)[1:]
                       + _bezier((xb, c), (xm, c), (xm, c - q), 4)[1:]
                       + [(xm, bot + q)] + _bezier((xm, bot + q), (xm, bot), (xa, bot), 4)[1:]]
        return Box(w, top, -bot, [_strokes(f"{kind}_{side}", strokes, src)],
                   "open" if side == "left" else "close")

    # ---- accents and boxes

    def accent(self, n, s, script, in_frac):
        _, kind, body_n, src = n
        body = self.node(body_n, s, script, in_frac)
        cx = body.w / 2
        if body.char is not None:
            # a single letter: centre on its glyph rather than its whole box
            w = 0.7 * body.w
        else:
            w = body.w - 0.06 * s
        y = body.a + 0.1 * s
        a, d = body.a, body.d
        if kind == "overline":
            strokes = [[(cx - w / 2, y), (cx + w / 2, y)]]
            a = y
        elif kind == "underline":
            y = -body.d - 0.08 * s
            strokes = [[(cx - w / 2, y), (cx + w / 2, y)]]
            d = -y
        elif kind == "vec_arrow":
            w = max(w, 0.45 * s)
            y += 0.04 * s
            tip = cx + w / 2
            strokes = [[(cx - w / 2, y), (tip, y)],
                       [(tip - 0.13 * s, y + 0.07 * s), (tip, y), (tip - 0.13 * s, y - 0.07 * s)]]
            a = y + 0.07 * s
        elif kind == "hat":
            hw = min(max(w, 0.3 * s), 1.2 * s) / 2
            strokes = [[(cx - hw, y), (cx, y + 0.14 * s), (cx + hw, y)]]
            a = y + 0.14 * s
        elif kind == "tilde":
            hw = min(max(w, 0.3 * s), 1.2 * s) / 2
            y += 0.06 * s
            strokes = [[(cx - hw + 2 * hw * i / 8, y + 0.05 * s * math.sin(2 * math.pi * i / 8)) for i in range(9)]]
            a = y + 0.06 * s
        else:  # dot
            r = 0.035 * s
            cy = y + r + 0.02 * s
            strokes = [[(cx + r * math.cos(2 * math.pi * i / 8), cy + r * math.sin(2 * math.pi * i / 8))
                        for i in range(9)]]
            a = cy + r
        w_box = max(body.w, max(x for st in strokes for x, _ in st))
        x0 = min(0.0, min(x for st in strokes for x, _ in st))
        atoms = _moved(body.atoms + [_strokes(kind, strokes, src)], -x0, 0.0)
        return Box(w_box - x0, a, d, atoms, body.cls if body.cls != "inner" else "ord")

    def negated(self, n, s, script, in_frac):
        """\\not: the symbol with a slash through it."""
        _, body_n, src = n
        body = self.node(body_n, s, script, in_frac)
        cx, cy = body.w / 2, self.font.axis * s
        slash = _strokes("not", [[(cx + 0.14 * s, cy + 0.34 * s), (cx - 0.14 * s, cy - 0.34 * s)]], src)
        return Box(body.w, max(body.a, cy + 0.34 * s), max(body.d, 0.0), body.atoms + [slash], body.cls)

    def matrix(self, n, s, script, in_frac):
        """Cells centred in their columns, rows written top to bottom, the
        whole block centred on the math axis."""
        rows = n[1]
        cells = [[self.node(c, s, script, in_frac) for c in row] for row in rows]
        ncol = max(len(row) for row in cells)
        colw = [max(row[k].w for row in cells if k < len(row)) for k in range(ncol)]
        sep, margin = 0.9 * s, 0.12 * s
        # Each row is at least as tall as a line of text (TeX's strut).
        asc = [max([0.9 * s] + [c.a for c in row]) for row in cells]
        desc = [max([0.35 * s] + [c.d for c in row]) for row in cells]
        base = [0.0]
        for k in range(1, len(cells)):
            base.append(base[-1] - (desc[k - 1] + 0.1 * s + asc[k]))
        dy = self.font.axis * s - (asc[0] + base[-1] - desc[-1]) / 2
        atoms = []
        for row, y in zip(cells, base):
            x = margin
            for k, c in enumerate(row):
                atoms += _moved(c.atoms, x + (colw[k] - c.w) / 2, y + dy)
                x += colw[k] + sep
        w = 2 * margin + sum(colw) + sep * (ncol - 1)
        return Box(w, asc[0] + dy, desc[-1] - base[-1] - dy, atoms, "ord")

    def boxed(self, n, s, script, in_frac):
        _, body_n, src = n
        body = self.node(body_n, s, script, in_frac)
        pad, m = 0.25 * s, 0.05 * s
        x0, x1 = m, m + body.w + 2 * pad
        y0, y1 = -body.d - pad, body.a + pad
        rect = [(x0, y1), (x1, y1), (x1, y0), (x0, y0), (x0, y1)]
        atoms = _moved(body.atoms, m + pad, 0.0) + [_strokes("box", [rect], src)]
        return Box(x1 + m, y1, -y0, atoms, "ord")
