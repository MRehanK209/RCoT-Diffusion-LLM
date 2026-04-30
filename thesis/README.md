# Goettingen Thesis Draft Project

This project is an Overleaf-ready thesis draft based on the University of Goettingen-inspired `mhellmeier/LaTeX-Thesis-Template` structure. It preserves the root-file style of that template (`thesis.tex`, `front-page.tex`, `packages.tex`, `definitions.tex`, `acronyms.tex`, `literature.bib`, and `images/`) while storing the main body in `chapters/body.tex`.

## Main file

Compile `thesis.tex`.

Recommended Overleaf settings:

- Compiler: `pdfLaTeX`
- Bibliography: currently manual `thebibliography` in `thesis.tex` (no `biber` needed yet)
- Main document: `thesis.tex`

Local build, if `latexmk` is installed:

```bash
make
```

Alternatively, run `pdflatex thesis.tex` twice.

## Structure

- `thesis.tex` - main document entry point
- `front-page.tex` - title page
- `packages.tex` - packages and bibliography configuration
- `definitions.tex` - reusable commands and layout definitions
- `acronyms.tex` - abbreviation list
- `literature.bib` - optional BibTeX/BibLaTeX source file for later migration; current draft uses manual references in `thesis.tex`
- `chapters/body.tex` - thesis body, results tables, discussion, and retained milestone appendix
- `images/` - shared evaluation pipeline image
- `source_material/` - uploaded source material used to create the draft

## Draft status

This is a first thesis draft. It is numerically grounded in the supplied Milestone 2 material, but it still needs final proofreading, supervisor feedback, final figure styling, and possibly additional statistical tests before submission.
