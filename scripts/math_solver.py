"""Evaluate a SymPy expression written by a language model and format the value
as LaTeX the student can write, so the model translates the math and SymPy
does it.

The expression is Python syntax over a whitelist of SymPy names (SOLVER_NAMES);
any other name becomes a symbol, and any other call, attribute access or
subscript is refused before it is parsed. A value that still holds an
unevaluated integral, sum, limit or derivative is refused too, so a reply is
either exact or nothing.

demo_server.py runs this file as a worker (--serve) and talks to it through
Solver, so an evaluation that hangs can be killed without killing the server.

Usage: python scripts/math_solver.py "Integral(x**3, (x, 0, 2))"
"""
import ast
import json
import os
import re
import select
import subprocess
import sys

import sympy
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

SOLVER_NAMES = [
    "Integral", "integrate", "Derivative", "diff", "Sum", "summation", "Product", "Limit", "limit",
    "Rational", "sqrt", "root", "exp", "log", "sin", "cos", "tan", "sec", "csc", "cot",
    "asin", "acos", "atan", "sinh", "cosh", "tanh", "factorial", "binomial", "Abs", "floor", "ceiling",
    "gcd", "lcm", "Mod", "Eq", "solve", "simplify", "expand", "factor", "cancel", "apart", "together",
    "pi", "E", "I", "oo",
]
NAMESPACE = {"__builtins__": {}, **{name: getattr(sympy, name) for name in SOLVER_NAMES}}
# The parser's transformations rewrite names and numbers into calls to these.
PARSER_NAMES = {name: getattr(sympy, name) for name in ("Symbol", "Integer", "Float")}
TRANSFORMATIONS = standard_transformations + (convert_xor,)
UNEVALUATED = (sympy.Integral, sympy.Sum, sympy.Product, sympy.Limit, sympy.Derivative)
ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
                 ast.Tuple, ast.List, ast.keyword, ast.operator, ast.unaryop)

# MathWriting's normalized labels, which the student learned from, write function names as plain
# letters and brackets without \left and \right.
BOARD_LATEX = [
    (re.compile(r"\\(left|right)(?=[()\[\]|.]|\\[{}|])"), ""),
    (re.compile(r"\\operatorname\{(\w+)\}"), r"\1"),
    (re.compile(r"\\(log|ln|exp|sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|gcd|lcm|bmod)(?![A-Za-z])"), r"\1"),
    (re.compile(r"\\(arcsin|arccos|arctan)(?![A-Za-z])"), r"\1"),
    (re.compile(r"\\[,;:! ]"), ""),
]


def check(expr):
    """Refuse anything but arithmetic, names and calls to whitelisted functions."""
    for node in ast.walk(ast.parse(expr, mode="eval")):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"name {node.id} is not allowed")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in NAMESPACE):
            raise ValueError(f"{ast.unparse(node.func)} is not a solver function")
        if isinstance(node, ast.keyword) and (node.arg is None or node.arg.startswith("_")):
            raise ValueError("that keyword argument is not allowed")


def evaluate(expr):
    check(expr)
    value = parse_expr(expr, global_dict={**NAMESPACE, **PARSER_NAMES}, transformations=TRANSFORMATIONS)
    if isinstance(value, (list, tuple, set)):
        values = [v.doit() for v in value]
    else:
        values = [value.doit()]
    for v in values:
        if not isinstance(v, sympy.Basic):
            raise ValueError(f"{type(v).__name__} is not a math value")
        if v.has(*UNEVALUATED):
            raise ValueError("SymPy could not evaluate it in closed form")
        if v.has(sympy.nan, sympy.zoo):
            raise ValueError("the value is undefined")
    return values


def drop_loose_braces(latex):
    """Braces that group nothing for ^, _ or a command, like SymPy's around function arguments
    (cos{(x)}) and inside absolute values (|{x}|)."""
    out = list(latex)
    stack = []
    for i, c in enumerate(latex):
        if c not in "{}" or (i and latex[i - 1] == "\\"):
            continue
        if c == "{":
            before = latex[:i].rstrip()
            stack.append((i, before.endswith(("^", "_", "}", "]")) or re.search(r"\\[A-Za-z]+$", before)))
        elif stack:
            start, needed = stack.pop()
            if not needed:
                out[start] = out[i] = ""
    return "".join(out)


def board_latex(value):
    latex = sympy.latex(value)
    for pattern, replacement in BOARD_LATEX:
        latex = pattern.sub(replacement, latex)
    latex = drop_loose_braces(latex)
    # A space is kept only to end a command before a letter (\pi x) or before a function name
    # (x cos(x)); a product like x y z is written xyz.
    return re.sub(r" +", lambda m: " " if re.search(r"[A-Za-z]$", latex[:m.start()])
                  and (re.search(r"\\[A-Za-z]+$", latex[:m.start()]) or re.match(r"[A-Za-z]{2,}\(", latex[m.end():]))
                  else "", latex)


def solve_to_latex(expr):
    return ",".join(board_latex(v) for v in evaluate(expr))


def serve():
    """One JSON request per line on stdin, one reply per line on stdout."""
    for line in sys.stdin:
        try:
            reply = {"latex": solve_to_latex(json.loads(line)["expr"])}
        except Exception as e:
            reply = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(reply), flush=True)


class Solver:
    """solve_to_latex in a worker process, which is killed and restarted when it runs past the
    timeout; SymPy's integrate and simplify can run for minutes, and big integer powers can't be
    interrupted from inside the process."""

    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self.worker = None

    def start(self):
        self.worker = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--serve"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def __call__(self, expr):
        if self.worker is None or self.worker.poll() is not None:
            self.start()
        self.worker.stdin.write(json.dumps({"expr": expr}) + "\n")
        self.worker.stdin.flush()
        ready, _, _ = select.select([self.worker.stdout], [], [], self.timeout)
        if not ready:
            self.worker.kill()
            self.worker.wait()
            self.worker = None
            raise TimeoutError(f"SymPy took longer than {self.timeout:g} s")
        reply = json.loads(self.worker.stdout.readline())
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply["latex"]


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        print(solve_to_latex(sys.argv[1]))
