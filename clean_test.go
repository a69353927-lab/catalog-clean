package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// fakeClaude answers like the Messages API, keyed by the SKU in the row it is sent.
func fakeClaude(t *testing.T, calls *atomic.Int32) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var req struct {
			Model        string `json:"model"`
			Fallbacks    any    `json:"fallbacks"`
			OutputConfig struct {
				Effort string `json:"effort"`
				Format struct {
					Schema struct {
						Properties struct {
							Category struct{ Enum []string } `json:"category"`
						} `json:"properties"`
					} `json:"schema"`
				} `json:"format"`
			} `json:"output_config"`
			Messages []struct {
				Content []struct{ Text string } `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if req.Fallbacks != "default" || !strings.Contains(r.Header.Get("anthropic-beta"), "server-side-fallback-2026-07-01") ||
			req.OutputConfig.Effort != "low" || !slices.Contains(req.OutputConfig.Format.Schema.Properties.Category.Enum, unknown) {
			t.Errorf("unexpected request: %+v beta=%q", req, r.Header.Get("anthropic-beta"))
		}
		var row map[string]string
		json.Unmarshal([]byte(req.Messages[0].Content[0].Text), &row)

		f := func(v, s string) map[string]string { return map[string]string{"value": v, "source": s} }
		none := f("", "")
		ans := map[string]any{
			"title": "Clean " + row["sku"], "brand": none, "category": "Toys & Games > Toys",
			"attributes": map[string]any{"color": none, "size": none, "material": none}, "doubts": "",
		}
		stop := "end_turn"
		switch row["sku"] {
		case "TS-1001":
			ans["category"] = "Apparel & Accessories > Clothing > Shirts & Tops"
			ans["attributes"] = map[string]any{"color": f("Black", "BLK"), "size": f("XL", "XL"), "material": f("Cotton", "100%  cott.")}
		case "TS-1002": // material invented: "poly blend" is not in the row
			ans["brand"] = f("Fruit of the Loom", "FRUITLOOM")
			ans["attributes"] = map[string]any{"color": f("White", "wht"), "size": none, "material": f("Polyester", "poly blend")}
		case "XX-1601":
			ans["category"], ans["doubts"] = unknown, "not clear what this part is"
		case "CS-1501":
			stop = "refusal"
		}
		text, _ := json.Marshal(ans)
		content := []any{map[string]string{"type": "text", "text": string(text)}}
		if stop == "refusal" {
			content = []any{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "model": req.Model,
			"content": content, "stop_reason": stop,
			"stop_details": map[string]any{"type": "refusal", "category": "cyber", "explanation": "x"},
			"usage":        map[string]int{"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 500, "cache_creation_input_tokens": 0},
		})
	}))
}

func TestRun(t *testing.T) {
	var calls atomic.Int32
	srv := fakeClaude(t, &calls)
	defer srv.Close()
	client := anthropic.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("test"), option.WithMaxRetries(0))

	head, rows, err := readCSV("testdata/products.csv", ',')
	if err != nil {
		t.Fatal(err)
	}
	cats, _ := readLines("testdata/taxonomy.txt")
	c := NewCleaner(client, "claude-opus-5-5", "low", "English", cats, []string{"color", "size", "material"})
	out := filepath.Join(t.TempDir(), "cleaned.csv")

	s, err := run(context.Background(), c, head, rows, 0, 4, out)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Results) != 30 || calls.Load() != 30 || s.NeedReview() != 3 {
		t.Fatalf("results %d, calls %d, review %d, failed %v", len(s.Results), calls.Load(), s.NeedReview(), s.Failed[:min(2, len(s.Failed))])
	}
	by := map[string]Result{}
	for i, r := range s.Results {
		by[s.Rows[i][0]] = r
	}
	if r := by["TS-1001"]; len(r.Review) != 0 || r.Attrs["color"].Value != "Black" {
		t.Errorf("TS-1001 should pass (source match ignores case and spacing): %+v", r)
	}
	if r := by["TS-1002"]; len(r.Review) != 1 || !strings.Contains(r.Review[0], `material "Polyester"`) {
		t.Errorf("TS-1002 invented material not caught: %v", r.Review)
	}
	if r := by["XX-1601"]; len(r.Review) != 2 {
		t.Errorf("XX-1601 should flag category and doubts: %v", r.Review)
	}
	if r := by["CS-1501"]; len(r.Review) != 1 || r.Review[0] != "model declined: cyber" {
		t.Errorf("CS-1501 refusal: %v", r.Review)
	}
	// 30 rows × (1000 in × $4 + 200 out × $20 + 500 cache read × $0.20) / 1M
	if cost, perRow, _ := s.Cost("claude-opus-5-5"); abs(cost-0.243) > 1e-9 || abs(perRow-0.0081) > 1e-9 {
		t.Errorf("cost %v per row %v", cost, perRow)
	}

	if err := writeCSV(out, s, c.Attrs); err != nil {
		t.Fatal(err)
	}
	f, _ := os.Open(out)
	recs, _ := csv.NewReader(f).ReadAll()
	f.Close()
	if len(recs) != 31 || recs[1][6] != "Clean TS-1001" || recs[0][7] != "clean_brand" || recs[1][9] != "Black" || recs[2][12] != "true" {
		t.Errorf("csv: %v / %v", recs[0], recs[1])
	}

	// Rerun: everything comes from the cache. Change one row: only that row is sent again.
	rows[5][1] += " (updated)"
	calls.Store(0)
	s, err = run(context.Background(), c, head, rows, 0, 4, out)
	if err != nil || calls.Load() != 1 || s.New != 1 || len(s.Results) != 30 {
		t.Fatalf("rerun: calls %d new %d err %v", calls.Load(), s.New, err)
	}
}

// A request every row fails the same way (bad key) must stop the run, not repeat 30 times.
func TestStopsOnAuthError(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(401)
		w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`))
	}))
	defer srv.Close()
	client := anthropic.NewClient(option.WithBaseURL(srv.URL), option.WithAPIKey("bad"), option.WithMaxRetries(0))
	head, rows, _ := readCSV("testdata/products.csv", ',')
	c := NewCleaner(client, "claude-opus-5-5", "low", "English", []string{"A"}, []string{"color"})

	_, err := run(context.Background(), c, head, rows, 0, 4, filepath.Join(t.TempDir(), "o.csv"))
	if err == nil || !strings.Contains(err.Error(), "401") || calls.Load() > 8 {
		t.Fatalf("err %v after %d calls", err, calls.Load())
	}
}

func abs(x float64) float64 { return max(x, -x) }
