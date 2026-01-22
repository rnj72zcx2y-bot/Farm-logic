# Farm Logic (Daily Puzzle Pipeline)

A daily logic puzzle game backend pipeline that generates **global (UTC) Daily puzzles** for **Easy / Medium / Hard** and publishes them as **public static JSON**.

This repo is designed to:
- Generate puzzles **30 days ahead**
- Ensure puzzles are **uniquely solvable** (Stop-at-2 solver) — *once the CSP solver is implemented*
- Maintain **steady difficulty** using a hardness score band — *temporarily disabled until solver is real*
- Host output via **static hosting** (e.g., Cloudflare Pages) from the `public/` folder

---

## Key Game Rules (V1)

### Board sizes (fixed)
- **Easy:** 4×4 grid, **12 active cells**, **6 cards**
- **Medium:** 6×6 grid, **30 active cells**, **15 cards**
- **Hard:** 8×8 grid, **48 active cells**, **24 cards**

Unused cells are blockers (tree / house / tractor).

### Cards
- Each card covers **exactly 2 orthogonally adjacent cells**.
- Each card has **one fence color**: `white | brown | black` (hint-only, **not** used in rules).
- Each half is either:
  - one of **8 animals**, or
  - **empty grass** (`null`)

### Animals (8)
- Chicken (2 legs)
- Cow (4)
- Horse (4)
- Dog (4)
- Cat (4)
- Bee (6)
- Spider (8)
- **Snail (0)** ← full animal, counts as animal species

### Rule evaluation
- Rules are evaluated **per CELL** (not per card).
- Cards may cross region boundaries.
- **Structured rules are included in the puzzle JSON** (no “badge string” logic).

### UX decisions
- **Snap-to-valid:** cards can only be placed on valid domino slots (active + free cells).
- **Check button only enabled when all cards are placed.**

---

## Public Endpoints (generated output)

After a successful run, the generator writes:

- `public/api/today.json`
- `public/api/latest.json`
- `public/api/daily/YYYY-MM-DD/`
  - `easy.json`
  - `medium.json`
  - `hard.json`
  - `meta.json` (**public**, no solutions)

### today.json
Pointer to the current UTC puzzle day.

### latest.json
Shows how far ahead puzzles are generated (default: +30 days).

### meta.json (public)
Contains:
- generator version + git commit
- per-difficulty seed
- uniqueness check results (Stop-at-2)
- solver stats
- hardness score (0..1) and target bands
- hash of each puzzle JSON

> ⚠️ **No solutions** are published.

---

## Repository Structure


.github/workflows/ main.yml                # GitHub Actions workflow
generator/ init.py requirements.txt config.py difficulty_score.py export_json.py generate_one_day.py generate_range.py solver_unique.py        # CSP solver hook (Stop-at-2)
public/ api/ today.json latest.json daily/                # generated day folders

---

## How It Works (Pipeline)

1. GitHub Actions runs daily at **00:10 UTC** (and can be triggered manually).
2. It generates puzzles for the next **30 days**.
3. It writes JSON into `public/api/`.
4. It commits & pushes changes back to the repo.
5. Cloudflare Pages (or any static host) serves `public/` as the website root.

---

## Setup (Step-by-step)

### 1) Repository Secrets
Add a GitHub Actions secret:

- **Name:** `GLOBAL_SALT`
- **Value:** a long random string (40+ chars)

Path:
`Repo → Settings → Secrets and variables → Actions → New repository secret`

### 2) GitHub Actions permissions (required for pushing)
In your workflow file (`.github/workflows/main.yml`) ensure:

```yaml
permissions:
  contents: write

Also enable write permissions in GitHub UI: Repo → Settings → Actions → General → Workflow permissions → Read and write
3) Run the workflow manually (first run)
• Go to Actions
• Select workflow Generate Daily Puzzles (UTC, +30)
• Click Run workflow
4) Hosting (Cloudflare Pages recommended)
• Connect repository
• Set output directory to public
• Deploy

---

Puzzle JSON Schema (V1)
Each puzzle file (easy/medium/hard) contains:
• grid.activeCellsCoords: list of playable cells
• grid.blocked: list of blocker objects
• regions[]: each with cells + a structured rule object
• cards[]: list of cards with a and b halves (animal name or null)
Rule object types (V1)
• Legs sum:
    • { "type":"legs", "op":"=", "value": N }
    • { "type":"legs", "op":"<", "value": N }
    • { "type":"legs", "op":">", "value": N }
• Animal count (non-empty only):
    • { "type":"animals", "op":"=", "value": N }
    • { "type":"animals", "op":"<", "value": N }
    • { "type":"animals", "op":">", "value": N }
• Unique species:
    • { "type":"uniqueSpecies" }  (empty ignored)
• Only one species:
    • { "type":"onlySpecies", "species":"Cow" } (empty not allowed)
