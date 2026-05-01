# Thesis draft (LaTeX)

This folder is a University of Göttingen–style thesis draft (structure inspired by `mhellmeier/LaTeX-Thesis-Template`). The **only file you must compile** is `thesis.tex`; it pulls in `packages.tex`, `definitions.tex`, `front-page.tex`, `acronyms.tex`, and `chapters/body.tex`.

The current draft includes generated evaluation figures in `images/`. To rebuild them from the repository result JSONs, run this from the repository root:

```bash
.venv/bin/python thesis/scripts/build_thesis_figures.py
```

The plotting script is dependency-light and writes PDF figures directly into `thesis/images/`.

The strengthened draft also adds analysis helpers for parser sensitivity, diversity, and oracle complementarity:

```bash
.venv/bin/python thesis/scripts/analyze_parser_failures.py
.venv/bin/python thesis/scripts/analyze_diversity_and_ensemble.py
.venv/bin/python thesis/scripts/build_thesis_figures.py
```

The first two scripts write CSV summaries into `thesis/tables/`; the figure builder reads those CSVs when present.

---

## 1. Install a LaTeX toolchain

Pick **one** path: **TeX Live + latexmk** (typical for local editing) or **Tectonic** (single binary, good for minimal setups).

### Linux (Debian / Ubuntu)

Update package lists, then install TeX extras and `latexmk`:

```bash
sudo apt update
sudo apt install -y texlive-latex-extra latexmk
```

If the compiler still misses fonts or KOMA classes, add:

```bash
sudo apt install -y texlive-fonts-recommended
```

**Check that TeX is on your PATH** (then restart Cursor/VS Code if you use LaTeX Workshop):

```bash
latexmk -v
kpsewhich scrartcl.cls
```

If `apt` reports **404 / unable to fetch**, refresh mirrors (`sudo apt update`) or fix `/etc/apt/sources.list`, then retry. On shared clusters without `sudo`, use the **Tectonic** option below or ask your admin for a TeX module.

### macOS

Install **MacTeX** or **BasicTeX**, ensure binaries are on `PATH` (often `/Library/TeX/texbin`), install `latexmk` if missing, then verify:

```bash
export PATH="/Library/TeX/texbin:$PATH"
latexmk -v
kpsewhich scrartcl.cls
```

### Windows

Install **MiKTeX** or **TeX Live**, install the **latexmk** package if the installer did not, and ensure the TeX `bin` directory is on your user **PATH**. Restart Cursor/VS Code after changing PATH.

---

## 2. Compile on your machine

Always run commands from **this directory** (`thesis/`), i.e. the folder that contains `thesis.tex` and `Makefile`.

### Option A — Makefile (uses `latexmk`)

```bash
cd thesis
make
```

This runs `latexmk -pdf` on `thesis.tex`. The PDF is written next to the sources as `thesis/thesis.pdf`.

### Option B — `latexmk` directly

```bash
cd thesis
latexmk -pdf -interaction=nonstopmode thesis.tex
```

### Option C — `pdflatex` only (minimal; fewer passes than `latexmk`)

```bash
cd thesis
pdflatex -interaction=nonstopmode thesis.tex
pdflatex -interaction=nonstopmode thesis.tex
```

Run `pdflatex` **at least twice** so the table of contents and lists settle.

### Option D — Tectonic (no full TeX Live install)

Install [Tectonic](https://tectonic-typesetting.github.io/), then:

```bash
cd thesis
tectonic thesis.tex
```

This repository may also ship a vendored Tectonic under `../.tools/tectonic/` for Cursor builds; that path is optional and mainly for environments without `apt`.

If the vendored binary is present, this works from `thesis/`:

```bash
../.tools/tectonic/tectonic thesis.tex
```

---

## 3. Cursor / VS Code (LaTeX Workshop)

1. Install the extension **LaTeX Workshop** (`James-Yu.latex-workshop`).
2. Open the **repository root** (recommended) or the `thesis/` folder.
3. Use **LaTeX Workshop: Build LaTeX project** (or the recipe picker).

Workspace settings live in `../.vscode/settings.json` (repo root) and `.vscode/settings.json` (if you opened `thesis/` alone). The default recipe may use vendored **Tectonic** when system TeX is missing; after installing `latexmk`, choose the **latexmk (pdf)** recipe if you prefer TeX Live.

Do **not** add `% !TEX program = latexmk` unless TeX is installed; without TeX, that magic line can break LaTeX Workshop’s build.

---

## 4. Overleaf

- **Main document:** `thesis.tex`
- **Compiler:** pdfLaTeX
- **Bibliography:** this draft uses a manual `thebibliography` in `thesis.tex` (no `biber` step required yet).

---

## 5. Project layout

| Path | Role |
|------|------|
| `thesis.tex` | Main entry; compile this |
| `front-page.tex` | Title page |
| `packages.tex` | `\usepackage` list |
| `definitions.tex` | Layout helpers and macros |
| `acronyms.tex` | Abbreviation table |
| `chapters/body.tex` | Lightweight orchestrator that inputs the chapter files |
| `chapters/01_*.tex`–`08_*.tex` | Main thesis sections split for easier editing |
| `chapters/05*_results_*.tex` | Results subsections split out because they contain most figures/tables |
| `chapters/appendix_*.tex` | Appendix sections |
| `images/` | Figures used by the draft |
| `scripts/build_thesis_figures.py` | Rebuilds thesis result figures from JSON artifacts |
| `source_material/` | Source uploads used to build the draft |
| `Makefile` | `make` → `latexmk -pdf thesis.tex` |
| `requirements.txt` | Machine-readable notes (not `pip`) |

---

## 6. Draft status

This is a strengthened thesis draft: it has real result figures, parser documentation, and updated methodology/results prose. It still needs proofreading, supervisor feedback, final styling, and possible extra analysis before submission.
