# Fonts

`svg/` holds single-line (single-stroke) SVG fonts from Inkscape's Hershey
Text extension (https://gitlab.com/inkscape/extensions, `svg_fonts/`).
`scripts/render_inkml.py --font <name>` uses `svg/<name>.svg` when it
exists, and otherwise treats the name as a Hershey font from the
`HersheyFonts` package (for example `futural` or `cursive`).

The EMS fonts are single-line derivatives of OFL-licensed fonts, converted by
Windell H. Oskay (Evil Mad Scientist). Each file's `<metadata>` names the
original font and designer. They are licensed under the SIL Open Font License
1.1; the license text is in `svg/OFL.txt`.

The renderer scales each SVG font so its cap height and baseline match the
Hershey `futural` font's. Text then has the same capital size for a given
`text_height` in any font.

| Font | Style | Used in batch renders |
|---|---|---|
| EMSReadability, EMSReadabilityItalic | clean print | yes |
| EMSTech | technical print | yes |
| EMSNixish | rounded print, wide | no (did not look human) |
| EMSAllure | connected script | yes |
| EMSFelix | casual print | no (did not look human) |
| EMSNixishItalic | rounded print, italic | no (similar to EMSNixish) |
| EMSElfin | narrow | no (digits are tiny) |
| EMSOsmotron | display | no (tall lines collide) |
