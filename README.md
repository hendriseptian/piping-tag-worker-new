# Piping Tag Extractor Frontend V1

Static GitHub Pages frontend for:

https://piping-tag-worker-new.side-gs78.workers.dev

## Files

- `index.html` — UI
- `style.css` — dark responsive UI
- `script.js` — PDF.js rendering, overlapping tiles, Worker API call, editable result table, Excel export

## Processing

1. Select a PDF.
2. The first page is rendered in the browser with PDF.js.
3. The page is divided into 4 × 2 = 8 overlapping tiles.
4. Tiles are JPEG-compressed to stay within Worker payload limits.
5. A reduced full-page overview is sent for P&ID number detection.
6. Tiles are sent to `/api/extract`.
7. Results are editable.
8. Excel contains exactly:
   - Tag No.
   - P&ID No.
   - From
   - To
   - NPS (in)

`From` and `To` are intentionally exported blank.

## GitHub Pages

Upload the three frontend files to a GitHub Pages repository and publish from the selected branch/folder.

No backend secret is stored in the frontend.
