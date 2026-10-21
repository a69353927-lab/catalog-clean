# catalog-clean

Turns messy supplier product data into clean, structured catalog data with Claude, and refuses to trust it blindly.

```
MENS TSHIRT CREW NECK BLK XL 100% COTT.   →   title     Men's Crew Neck T-Shirt, Black, XL
col: BLK / sz: XL                             category  Apparel & Accessories > Clothing > Shirts & Tops
                                              color     Black    (from "BLK")
                                              size      XL       (from "XL")
                                              material  Cotton   (from "100% COTT.")
```

- **Clean titles** in the language you choose, no ALL CAPS, codes or promo words
- **Your taxonomy only**: the category list is enforced by a JSON schema; the model cannot invent one
- **No invented specs**: every brand and attribute comes with the exact text it was read from, and that text is checked against the supplier row. Values that can't be found there are flagged, not published
- **Review queue**: unclear products, unmatched categories, unverifiable values and model doubts are marked `needs_review` with the reason
- **Know the cost first**: run 10 rows with `-limit 10` and the report shows the real cost per row and the estimate for the whole file
- **Safe on big files**: parallel requests with backoff on rate limits; finished rows are cached, so an interrupted run resumes without paying twice; a bad API key stops the run at once
- **Any CSV**: send whatever columns the supplier has; output keeps them and adds `clean_*` columns

## Usage

```sh
export ANTHROPIC_API_KEY=sk-ant-...

# 1. sample run: 10 rows, see quality and cost (numbers below are illustrative)
catalog-clean -in products.csv -taxonomy taxonomy.txt -attrs color,size,material -limit 10
# cleaned 10 of 1000 rows (10 this run) · 2 need review · 0 failed
# cost this run $0.08 · $0.0081 per row · all 1000 rows ≈ $8.10

# 2. full run (the 10 sampled rows come from the cache)
catalog-clean -in products.csv -taxonomy taxonomy.txt -attrs color,size,material
open report.html
```

| Flag | Default | |
|---|---|---|
| `-attrs` | `color,size,material` | attributes to extract |
| `-lang` | `English` | language for titles |
| `-model` | `claude-opus-5-5` | `claude-sonnet-5-5` / `claude-haiku-5-5` for cheaper bulk runs |
| `-effort` | `low` | raise for hard catalogs |
| `-workers` | `4` | parallel requests |
| `-csv-delim` | `,` | |

If the model declines a request, the API re-runs it on its recommended fallback model (server-side fallbacks); a row that is still declined goes to review.

## Tests

`go test ./...` runs the whole pipeline against a fake Messages API: schema and fallback settings in the request, the invented-value check, refusals, cost math, resume from cache, and stopping on an auth error.
