# Validation contract

Every run directory must contain exact argv/environment, image and source IDs,
full server log, requests/results, GPU and host telemetry, and an explicit
PASS/FAIL summary. `COMPLETE` is written only after every criterion for that
run passes.

## Frozen functional battery

- deterministic plain-text completion;
- arithmetic `17*23+5 = 396`;
- Vision OCR with all four expected lines exact;
- chart categories, values and maximum;
- spatial relation of three shapes;
- fine-detail grid answer `43`;
- two-image day/night comparison.

## FP8-KV checks

- actual cache dtype is FP8 E4M3;
- calibrated scale file is mounted and all 12 QSA layers log calibrated K/V
  scales;
- no fallback to scale 1.0;
- no NaN/Inf;
- deterministic BF16/FP8 A/B uses the same W4A16 model and prompts;
- record exact-match separately from semantic equivalence;
- if feasible, preserve top-logprob divergence or an explicitly approximate
  KLD measure. Do not label top-N KLD as full-vocabulary KLD.

## Long-context checks

An acceptance request must contain at least one real image plus text padding,
a late text needle and a final image-dependent question. Record configured
length, tokenizer-counted actual input and served output independently.

No eager mode, no language-model-only mode and no graph disablement are valid
capacity workarounds.

