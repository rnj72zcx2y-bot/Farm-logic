# Farm-logic Daily Demo (MVP)

This folder contains a minimal playable demo that loads today's puzzle from your generated JSON API.

## How it works
- Loads `api/today.json` (relative to the current site URL)
- Then loads the selected difficulty JSON via `today.paths.easy|medium|hard`

## Hosting
Place these files in your web root (e.g., `public/demo/`) and open `index.html`.
For GitHub Pages or any static hosting, ensure your generated JSON is available at `.../api/...`.

## Gameplay
- Click a card
- Click two adjacent active cells to place it
- Use **Flip** (or press `F`) if you want to swap `a` and `b`
- Press **Check** to validate region rules
- Press **Reset** to clear placements

## Notes
- This demo validates rules client-side:
  - legs (=,<,>)
  - animals (=,<,>)
  - uniqueSpecies
  - onlySpecies
- Leg mapping is hardcoded for the 8 demo animals:
  - Chicken=2, Bee=6, Spider=8, Snail=0, others=4
