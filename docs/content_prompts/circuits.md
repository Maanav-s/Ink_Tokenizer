Subject rules (circuits):

- Draw a `schematic` when a problem gives or asks for a circuit, with labels and values on its
  components ("R1", "2k").
- Show each calculation as `math` lines: the formula, then continuation lines substituting values
  and giving the result. A continuation sets `continues` to the line above, and starts with "=".
- `text` is Python arithmetic with explicit `*` and no units: keep each line in one unit (ohms,
  volts, amps, seconds, or kΩ / mA throughout) and give it a `check` with the values as vars, e.g.
  latex `R_{eq} = R_1 + \frac{R_2 R_3}{R_2 + R_3}`, text `Req = R1 + R2*R3/(R2 + R3)`, defines
  `Req`, and no check (its only side besides Req is the formula); its continuation `= 2 + 4*4/(4 + 4)
  = 4` carries `check {"vars": {"R1": 2, "R2": 4, "R3": 4}}` and is compared with the formula.
  A line needs at least two sides other than its `defines` name, counting the line above for a
  continuation, to have a check.
- Units go in the latex (`= 4\,k\Omega`) and in text lines, not in `text`.
- A small table of node voltages or branch currents if a problem asks for several values. No truth
  tables or flowcharts.
