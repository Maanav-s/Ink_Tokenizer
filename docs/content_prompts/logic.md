Subject rules (digital logic):

- Use truth tables, Boolean algebra and gate-level circuits as the problems need them.
- A truth table's header names its columns (single letters or short names like Cout), and every
  Boolean `math` line that defines an output sets `defines` to that column's name exactly.
- The `text` of a Boolean line uses the Boolean form only: `+` for or, juxtaposition for and, `'`
  for not, `xor`, and the truth table's column names, e.g. `M = AB + AC + BC` or `S = A'B + AB'`.
  Never `*`, `1-A`, `or`, `and` or Python syntax. Boolean lines have no `check`.
- A Boolean line whose left side is not a truth-table column (an intermediate term like A'B) is
  checked only if it is a column; otherwise make it a `note` text line instead of a `math` line.
- Circuit input and output pin labels match the truth table's column names, so the circuit is
  simulated against the table.
- Number conversions and arithmetic are `math` lines with Python arithmetic in `text` and a `check`
  (e.g. text `182 = 128 + 32 + 16 + 4 + 2`, check `{"vars": {}}`).
