Subject rules (physics: mechanics, energy, waves, E&M):

- Start a problem with its givens as a short `list` or `note` lines ("m = 2.0 kg, v0 = 0, h = 5 m"),
  then the governing equation and its solution as `math` lines: the formula, then continuation
  lines substituting values and giving the result with units in the latex.
- `text` is Python arithmetic with explicit `*` and no units, in SI throughout. A `check` needs two
  sides besides the `defines` name, and a continuation counts the line above's last side: the formula
  line (latex `v = \sqrt{2gh}`, text `v = sqrt(2*g*h)`, defines `v`) gets no check, and its
  continuation (`= sqrt(2*9.8*5) = 9.90`) gets `check {"vars": {"g": 9.8, "h": 5}}`.
- Vectors with `\vec{F}`, components with subscripts (`F_x`, `v_{0y}`).
- A small table only when a problem asks for several values (e.g. energy at points A, B, C). No
  circuits, schematics, truth tables or flowcharts.
