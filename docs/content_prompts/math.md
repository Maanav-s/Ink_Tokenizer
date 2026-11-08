Subject rules (math: algebra, calculus, probability):

- Show every step. A problem's working is a `math` line followed by 2–5 continuation lines, one
  algebraic step each, ending with the result. A continuation sets `continues` to the line just
  above it, and its `latex` and `text` start with "=". A new result such as evaluating at a point
  (f'(0) = 0) is a new `math` line, not a continuation.
- `text` is Python arithmetic with explicit `*` (`2*x*sin(x)`, never `2x sin x`), using `sqrt`,
  `exp`, `log`, `sin`, `cos`, `tan`, `pi` and `**` (or `^`).
- A `check` compares every side of a line, plus (for a continuation) the last side of the line
  above, and skips a side that is just the `defines` name. So a first line like `fp = 2*x*sin(x)`
  has nothing to compare: give it no `check`, and put the check on its continuation lines instead.
  For a function of x, check at a sample point: `"check": {"vars": {"x": 0.7}}`; continuations of a
  line inherit vars given anywhere above them in the chain.
- When the left side cannot be computed (a derivative, an integral, a limit), name it with `defines`
  and use that name as the left side of `text`: latex `f'(x) = 2x\sin x + x^2\cos x`, text
  `fp = 2*x*sin(x) + x**2*cos(x)`, defines `fp` (no check on this line). For an integral, `text` is the evaluated
  antiderivative: latex `\int_0^2 (3x^2-4x+1)\,dx = \left[x^3-2x^2+x\right]_0^2`, text
  `I = (2**3-2*2**2+2) - 0`, defines `I`.
- Short `note` text lines may name the rule or substitution used ("product rule", "u = x,
  dv = e^x dx"). A small table only if a problem asks for one (a PMF, say). No diagrams, circuits or
  truth tables.
