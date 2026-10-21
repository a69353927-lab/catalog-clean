// catalog-clean turns messy supplier product rows into clean titles, a category from your
// taxonomy and structured attributes, using Claude. Every value is checked against the source text.
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html/template"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

func main() {
	in := flag.String("in", "", "product CSV (first row is the header)")
	delim := flag.String("csv-delim", ",", "CSV delimiter")
	taxonomy := flag.String("taxonomy", "", "category list, one per line")
	attrs := flag.String("attrs", "color,size,material", "attributes to extract")
	lang := flag.String("lang", "English", "language for clean titles")
	model := flag.String("model", "claude-opus-5-5", "Claude model")
	effort := flag.String("effort", "low", "low | medium | high | xhigh | max")
	workers := flag.Int("workers", 4, "parallel requests")
	limit := flag.Int("limit", 0, "clean only the first N rows, e.g. a cheap sample run to see the cost (0 = all)")
	out := flag.String("out", "cleaned.csv", "output CSV; finished rows are cached in <out>.jsonl")
	report := flag.String("report", "report.html", "before/after report")
	flag.Parse()
	if *in == "" || *taxonomy == "" {
		log.Fatal("-in and -taxonomy are required")
	}

	head, rows, err := readCSV(*in, []rune(*delim)[0])
	if err != nil {
		log.Fatal(err)
	}
	cats, err := readLines(*taxonomy)
	if err != nil {
		log.Fatal(err)
	}
	client := anthropic.NewClient(option.WithMaxRetries(5)) // backs off on 429 / 5xx
	c := NewCleaner(client, *model, *effort, *lang, cats, split(*attrs))

	s, err := run(context.Background(), c, head, rows, *limit, *workers, *out)
	if err != nil {
		log.Fatal(err)
	}
	if err := writeCSV(*out, s, c.Attrs); err != nil {
		log.Fatal(err)
	}
	if err := writeReport(*report, s, c); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("cleaned %d of %d rows (%d this run) · %d need review · %d failed\n",
		len(s.Results), s.Total, s.New, s.NeedReview(), len(s.Failed))
	for _, f := range s.Failed {
		fmt.Println("  failed:", f)
	}
	if cost, perRow, ok := s.Cost(c.Model); ok {
		fmt.Printf("cost this run $%.4f · $%.4f per row · all %d rows ≈ $%.2f\n", cost, perRow, s.Total, perRow*float64(s.Total))
	}
	fmt.Printf("wrote %s and %s\n", *out, *report)
}

type Summary struct {
	Head    []string
	Rows    [][]string // raw rows, same order as Results
	Results []Result
	Total   int // rows in the input file
	New     int // rows cleaned by this run (the rest came from the cache)
	Failed  []string
	spent   Usage // tokens used by this run
}

func (s *Summary) NeedReview() (n int) {
	for _, r := range s.Results {
		if len(r.Review) > 0 {
			n++
		}
	}
	return n
}

// Cost returns this run's spend and the average per row, for projecting the full file.
func (s *Summary) Cost(model string) (run, perRow float64, ok bool) {
	var all Usage
	for _, r := range s.Results {
		all.In, all.Out, all.CacheWrite, all.CacheRead = all.In+r.Usage.In, all.Out+r.Usage.Out, all.CacheWrite+r.Usage.CacheWrite, all.CacheRead+r.Usage.CacheRead
	}
	run, ok = s.spent.Cost(model)
	total, _ := all.Cost(model)
	if len(s.Results) > 0 {
		perRow = total / float64(len(s.Results))
	}
	return run, perRow, ok
}

// run cleans rows in parallel. Each finished row is appended to <out>.jsonl at once,
// so a crash or Ctrl-C never pays for the same row twice.
func run(ctx context.Context, c *Cleaner, head []string, rows [][]string, limit, workers int, out string) (*Summary, error) {
	n := len(rows)
	if limit > 0 && limit < n {
		n = limit
	}
	cache := loadCache(out + ".jsonl")
	f, err := os.OpenFile(out+".jsonl", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	s := &Summary{Head: head, Total: len(rows)}
	results, done := make([]Result, n), make([]bool, n)
	var todo []int
	for i := range n {
		if r, ok := cache[i]; ok && r.Key == c.Key(rowJSON(record(head, rows[i]))) {
			results[i], done[i] = r, true
		} else {
			todo = append(todo, i)
		}
	}

	// A bad key, model name or request fails every row the same way: stop instead of paying 1000 times.
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	var mu sync.Mutex
	var wg sync.WaitGroup
	jobs := make(chan int)
	for range workers {
		wg.Go(func() {
			for i := range jobs {
				if ctx.Err() != nil {
					continue
				}
				r, err := c.Clean(ctx, i, record(head, rows[i]))
				mu.Lock()
				if err != nil {
					s.Failed = append(s.Failed, fmt.Sprintf("row %d: %v", i+2, err))
					var apiErr *anthropic.Error
					if errors.As(err, &apiErr) && apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 && apiErr.StatusCode != 429 {
						cancel(err)
					}
				} else {
					results[i], done[i] = r, true
					s.New++
					s.spent.In, s.spent.Out = s.spent.In+r.Usage.In, s.spent.Out+r.Usage.Out
					s.spent.CacheWrite, s.spent.CacheRead = s.spent.CacheWrite+r.Usage.CacheWrite, s.spent.CacheRead+r.Usage.CacheRead
					line, _ := json.Marshal(r)
					f.Write(append(line, '\n'))
					fmt.Fprintf(os.Stderr, "\r%d/%d", s.New, len(todo))
				}
				mu.Unlock()
			}
		})
	}
	for _, i := range todo {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if len(todo) > 0 {
		fmt.Fprintln(os.Stderr)
	}
	if err := context.Cause(ctx); err != nil {
		return nil, fmt.Errorf("stopped: %w", err)
	}

	for i := range n {
		if done[i] {
			s.Rows, s.Results = append(s.Rows, rows[i]), append(s.Results, results[i])
		}
	}
	return s, nil
}

func loadCache(path string) map[int]Result {
	m := map[int]Result{}
	f, err := os.Open(path)
	if err != nil {
		return m
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<20)
	for sc.Scan() {
		var r Result
		if json.Unmarshal(sc.Bytes(), &r) == nil {
			m[r.Row] = r
		}
	}
	return m
}

func record(head, row []string) map[string]string {
	m := map[string]string{}
	for i, h := range head {
		if i < len(row) && strings.TrimSpace(row[i]) != "" {
			m[h] = row[i]
		}
	}
	return m
}

func readCSV(path string, delim rune) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.Comma, r.FieldsPerRecord, r.LazyQuotes = delim, -1, true
	recs, err := r.ReadAll()
	if err != nil || len(recs) == 0 {
		return nil, nil, fmt.Errorf("%s: no rows (%v)", path, err)
	}
	recs[0][0] = strings.TrimPrefix(recs[0][0], "\uFEFF")
	return recs[0], recs[1:], nil
}

func readLines(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, l := range strings.Split(string(b), "\n") {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return out, nil
}

func split(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func writeCSV(path string, s *Summary, attrs []string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	// clean_ prefix: supplier files often already have brand/color/size columns
	head := append(append([]string{}, s.Head...), "clean_title", "clean_brand", "clean_category")
	for _, a := range attrs {
		head = append(head, "clean_"+a)
	}
	head = append(head, "needs_review", "review_reasons")
	w.Write(head)
	for i, r := range s.Results {
		row := append(append([]string{}, s.Rows[i]...), make([]string, len(s.Head)-min(len(s.Head), len(s.Rows[i])))...)
		row = append(row, r.Title, r.Brand.Value, r.Category)
		for _, a := range attrs {
			row = append(row, r.Attrs[a].Value)
		}
		row = append(row, fmt.Sprint(len(r.Review) > 0), strings.Join(r.Review, "; "))
		w.Write(row)
	}
	w.Flush()
	return w.Error()
}

func writeReport(path string, s *Summary, c *Cleaner) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	run, perRow, ok := s.Cost(c.Model)
	return reportTmpl.Execute(f, map[string]any{
		"S": s, "Attrs": c.Attrs, "Model": c.Model, "Priced": ok,
		"Run": run, "PerRow": perRow, "Projected": perRow * float64(s.Total),
		"Raw": func(i int) string { return strings.Join(nonEmpty(s.Rows[i]), " · ") },
	})
}

func nonEmpty(xs []string) (out []string) {
	for _, x := range xs {
		if strings.TrimSpace(x) != "" {
			out = append(out, x)
		}
	}
	return out
}

var reportTmpl = template.Must(template.New("r").Funcs(template.FuncMap{
	"attr": func(r Result, a string) Field { return r.Attrs[a] },
	"call": func(f func(int) string, i int) string { return f(i) },
	"line": func(row int) int { return row + 2 }, // CSV line number: header is line 1
}).Parse(`<!doctype html><meta charset="utf-8"><title>Catalog cleanup report</title>
<style>
body{font:14px/1.5 system-ui,sans-serif;margin:24px auto;max-width:1200px;padding:0 16px;color:#1f2328}
table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border-bottom:1px solid #d0d7de;padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f6f8fa}.raw{color:#57606a;font-size:12px;max-width:320px}.rev{background:#fff8c5}.why{color:#9a6700;font-size:12px}
.src{color:#57606a;font-size:11px}.cards{display:flex;flex-wrap:wrap;gap:8px}.cards div{border:1px solid #d0d7de;border-radius:6px;padding:8px 14px}.cards b{display:block;font-size:22px}
</style>
<h1>Catalog cleanup report</h1>
<p>{{.Model}} · every brand and attribute value is checked against the supplier's own text</p>
<div class="cards">
<div><b>{{len .S.Results}} / {{.S.Total}}</b>rows cleaned</div>
<div><b>{{.S.NeedReview}}</b>need review</div>
{{if .Priced}}<div><b>${{printf "%.4f" .PerRow}}</b>per row</div>
<div><b>${{printf "%.2f" .Projected}}</b>estimated for all {{.S.Total}} rows</div>{{end}}
</div>
{{$attrs := .Attrs}}{{$raw := .Raw}}
<table><tr><th>#</th><th>Before (supplier data)</th><th>Clean title</th><th>Brand</th><th>Category</th>{{range $attrs}}<th>{{.}}</th>{{end}}</tr>
{{range $i, $r := .S.Results}}<tr{{if $r.Review}} class="rev"{{end}}>
<td>{{line $r.Row}}</td><td class="raw">{{call $raw $i}}</td>
<td>{{$r.Title}}{{range $r.Review}}<div class="why">⚠ {{.}}</div>{{end}}</td>
<td>{{$r.Brand.Value}}{{if $r.Brand.Source}}<div class="src">“{{$r.Brand.Source}}”</div>{{end}}</td>
<td>{{$r.Category}}</td>
{{range $attrs}}{{$f := attr $r .}}<td>{{$f.Value}}{{if $f.Source}}<div class="src">“{{$f.Source}}”</div>{{end}}</td>{{end}}
</tr>{{end}}</table>
`))
