# Farm-logic Daily Demo (MVP+)

This demo is optimized for iPad/Safari and GitHub Pages.

## Features
- Loads `api/today.json` and then the selected difficulty via `today.paths`.
- Drag & drop cards: drag onto the first cell, then tap a highlighted adjacent cell to complete placement.
- Undo: reverts the last placement.
- Remove: tap an occupied cell to remove its card.
- Check: validates region rules client-side.

## Install
Copy the folder into your repo under `public/Demo/` (or `public/demo/`) and keep the API under `public/api/`.

## Notes
- The demo computes rule validation locally (legs/animals/uniqueSpecies/onlySpecies).
- Leg mapping is hardcoded for the 8 animals.
