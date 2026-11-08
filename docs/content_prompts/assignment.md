You are writing the content of one student's handwritten page of worked solutions to the assignment
below, written with a stylus in a note-taking app on a US Letter page. The topic is **{topic}**.
Output a single JSON object following the schema below, and nothing else. Set "schema" to
"ink_tokenizer.synthetic_page.content/v1".

- Work through the problems in order. Start each with a short heading ("1. Product rule",
  "2) Bayes"), then its solution, before going on to the next problem.
- Write like a student: terse notes, abbreviations, arrows (→), symbols and units. Not prose,
  unless the subject rules below say otherwise.
- The page must fit on one sheet: about 25–40 written lines in total, and text lines of at most
  about 45 characters.
- Include only what the problems need and the subject rules allow: no tables, diagrams or truth
  tables that no problem asks for.
- Everything you write must be correct.
- Use only the characters listed under "Characters". LaTeX only in `math` elements and in `$...$`
  labels.
- Give no positions, sizes or styling: the layout is generated separately.
