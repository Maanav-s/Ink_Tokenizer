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

`scripts/fonts.py` normalizes every font, Hershey and SVG alike, to one
frame: the baseline is at 0.25 of the text height above the descender line,
and capitals (H, E) reach the full text height. Text then has the same
capital size for a given `text_height` in any font, while descenders keep
each font's proportions (cursive and EMSAllure reach about 0.2 h below the
descender line). The batch pool is `fonts.FONT_POOL`: futural, cursive,
EMSReadability, EMSReadabilityItalic, EMSTech and EMSAllure.

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
