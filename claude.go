package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
)

// Cleaner turns one messy product row into structured fields with Claude.
type Cleaner struct {
	Client anthropic.Client
	Model  string
	Effort string
	Attrs  []string
	system string
	schema map[string]any
}

// Field is a cleaned value plus the exact input text it was taken from,
// so every value can be checked against the source instead of trusted.
type Field struct {
	Value  string `json:"value"`
	Source string `json:"source"`
}

type Result struct {
	Row      int              `json:"row"`
	Key      string           `json:"key"` // hash of prompt + model + row; a changed key means re-clean
	Title    string           `json:"title"`
	Brand    Field            `json:"brand"`
	Category string           `json:"category"`
	Attrs    map[string]Field `json:"attributes"`
	Doubts   string           `json:"doubts"`
	Review   []string         `json:"review,omitempty"` // why a human should check this row
	Usage    Usage            `json:"usage"`
}

type Usage struct {
	In, Out, CacheWrite, CacheRead int64
}

const unknown = "UNKNOWN"

func NewCleaner(client anthropic.Client, model, effort, lang string, categories, attrs []string) *Cleaner {
	field := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []string{"value", "source"},
		"properties": map[string]any{
			"value":  map[string]any{"type": "string"},
			"source": map[string]any{"type": "string"},
		},
	}
	attrProps := map[string]any{}
	for _, a := range attrs {
		attrProps[a] = field
	}
	// ponytail: the whole taxonomy goes into one enum; split by top-level category if a client has thousands.
	schema := map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"required":             []string{"title", "brand", "category", "attributes", "doubts"},
		"properties": map[string]any{
			"title":    map[string]any{"type": "string"},
			"brand":    field,
			"category": map[string]any{"type": "string", "enum": append(append([]string{}, categories...), unknown)},
			"attributes": map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"required":             attrs,
				"properties":           attrProps,
			},
			"doubts": map[string]any{"type": "string"},
		},
	}

	system := fmt.Sprintf(`You clean product data for an online store. Each message is one product row from a supplier, as JSON.

Return:
- title: a clean, readable product title in %s. Brand first if known, then product type and key attributes. No ALL CAPS, no internal codes, no promo words.
- brand, and each attribute (%s): value is the normalized value (for example "BLK" becomes "Black", "100%% cott." becomes "Cotton"); source is the exact text from the row that states it, copied character for character. Fill a field only when the row states it. If it doesn't, leave value and source empty; do not guess from the product type or brand.
- category: the most specific category from the list that fits, or %s if none does.
- doubts: anything a person should double-check (conflicting data, unclear product), or empty.

Categories:
%s`, lang, strings.Join(attrs, ", "), unknown, strings.Join(categories, "\n"))

	return &Cleaner{Client: client, Model: model, Effort: effort, Attrs: attrs, system: system, schema: schema}
}

// rowJSON is what the model sees; map keys are sorted, so it is deterministic.
func rowJSON(rec map[string]string) string {
	b, _ := json.Marshal(rec)
	return string(b)
}

// Key identifies a cleaning job; cached results with the same key are reused.
func (c *Cleaner) Key(src string) string {
	h := sha256.Sum256([]byte(c.Model + "\x00" + c.Effort + "\x00" + c.system + "\x00" + src))
	return hex.EncodeToString(h[:8])
}

// Clean sends one row and validates the answer. Problems that need a person go into Result.Review;
// only transport/API failures return an error (those rows are retried on the next run).
func (c *Cleaner) Clean(ctx context.Context, row int, rec map[string]string) (Result, error) {
	src := rowJSON(rec)
	res := Result{Row: row, Key: c.Key(src)}

	msg, err := c.Client.Beta.Messages.New(ctx, anthropic.BetaMessageNewParams{
		Model:     c.Model,
		MaxTokens: 16000,
		// If the model declines, the API re-runs the request on its recommended fallback model.
		Betas:     []anthropic.AnthropicBeta{anthropic.AnthropicBetaServerSideFallback2026_07_01},
		Fallbacks: anthropic.BetaFallbacksParamUnion{OfDefault: constant.ValueOf[constant.Default]()},
		System:    []anthropic.BetaTextBlockParam{{Text: c.system, CacheControl: anthropic.NewBetaCacheControlEphemeralParam()}},
		OutputConfig: anthropic.BetaOutputConfigParam{
			Effort: anthropic.BetaOutputConfigEffort(c.Effort),
			Format: anthropic.BetaJSONOutputFormatParam{Schema: c.schema},
		},
		Messages: []anthropic.BetaMessageParam{anthropic.NewBetaUserMessage(anthropic.NewBetaTextBlock(src))},
	})
	if err != nil {
		return res, err
	}
	res.Usage = Usage{msg.Usage.InputTokens, msg.Usage.OutputTokens, msg.Usage.CacheCreationInputTokens, msg.Usage.CacheReadInputTokens}

	switch msg.StopReason {
	case anthropic.BetaStopReasonRefusal:
		res.Review = append(res.Review, "model declined: "+string(msg.StopDetails.Category))
		return res, nil
	case anthropic.BetaStopReasonMaxTokens:
		res.Review = append(res.Review, "answer cut off (max_tokens)")
		return res, nil
	}
	var text string
	for _, block := range msg.Content {
		if t, ok := block.AsAny().(anthropic.BetaTextBlock); ok {
			text += t.Text
		}
	}
	if err := json.Unmarshal([]byte(text), &res); err != nil {
		res.Review = append(res.Review, "unreadable answer: "+err.Error())
		return res, nil
	}
	res.Review = append(res.Review, check(res, rec, c.Attrs)...)
	return res, nil
}

// check flags values the row doesn't support. A value whose source text isn't in the row
// was invented, whatever the model says.
func check(r Result, rec map[string]string, attrs []string) []string {
	var hay []string
	for _, v := range rec {
		hay = append(hay, v)
	}
	raw := squash(strings.Join(hay, " "))

	var why []string
	fields := map[string]Field{"brand": r.Brand}
	names := append([]string{"brand"}, attrs...)
	for _, a := range attrs {
		fields[a] = r.Attrs[a]
	}
	for _, name := range names {
		f := fields[name]
		if f.Value != "" && (f.Source == "" || !strings.Contains(raw, squash(f.Source))) {
			why = append(why, fmt.Sprintf("%s %q not found in source", name, f.Value))
		}
	}
	if r.Category == unknown {
		why = append(why, "no matching category")
	}
	if r.Doubts != "" {
		why = append(why, "model: "+r.Doubts)
	}
	return why
}

// squash lowercases and collapses whitespace so "Cotton  Blend" matches "cotton blend".
func squash(s string) string { return strings.Join(strings.Fields(strings.ToLower(s)), " ") }

// $ per million tokens: input, output, cache write, cache read.
// ponytail: a request served by the fallback model is billed at that model's rates; estimates use the main model's.
var prices = map[string][4]float64{
	"claude-opus-5-5":   {4, 20, 5, 0.20},
	"claude-sonnet-5-5": {2, 10, 2.5, 0.20},
	"claude-haiku-5-5":  {0.10, 0.50, 0.125, 0.01},
}

func (u Usage) Cost(model string) (float64, bool) {
	p, ok := prices[model]
	return (float64(u.In)*p[0] + float64(u.Out)*p[1] + float64(u.CacheWrite)*p[2] + float64(u.CacheRead)*p[3]) / 1e6, ok
}
