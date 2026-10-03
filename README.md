# SynBioBench code

Reusable code for evaluating language models on quantitative synthetic-biology
tasks: response rendering and grading, symbolic equation comparison,
gene-circuit numerical solvers, metabolic-flux solvers, description verification,
and descriptive item analysis.

## Availability

This is a **code-only release**. The benchmark dataset cannot be publicly
released. Benchmark questions, answer keys, saved model responses, network
instances, and scripts containing embedded question banks are excluded.
The metabolic solver module contains the reusable functions only; its
item-specific reference cases are withheld. Small artificial records in the
grading tests exercise the software interface and are not a benchmark release.

The code can be used with separately supplied items, but it cannot independently
reproduce the manuscript's reported scores without the withheld dataset and
responses. The original dataset-dependent runner is omitted because its bundled
input paths are unavailable in this release. No claim of full experimental
reproducibility is made.

## Installation

Use Python 3.10 or later:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before installing (Windows PowerShell:
`.venv\Scripts\Activate.ps1`; macOS/Linux: `source .venv/bin/activate`).
The requirements include optional provider clients and the optional COBRA solver.
Local grading and the standard numerical solvers use NumPy, SciPy, and SymPy.
For API evaluation, copy `.env.example` to `.env` and set the appropriate key.
Credentials and outputs are ignored by Git.

## Evaluation using your own inputs

Each JSONL record supplies an `item_id`, `tier`, `stem`, `question`, `answer`,
and `grader`. A numeric grader specifies `kind: numeric`, `key: value`, and
`rel_tol`; its answer contains a numeric `value`. Categorical graders supply
`choices`, while equation graders also define their symbolic reference and
required answer components. The rendering and grading implementations in
`synbiobench/core/pipeline.py` define the accepted schema.

```sh
python -m synbiobench.core.pipeline run --items /path/to/your_items.jsonl --model mock --out results/preds.jsonl
python -m synbiobench.core.pipeline grade --items /path/to/your_items.jsonl --preds results/preds.jsonl --out results/scores.jsonl
python -m synbiobench.core.pipeline score --items /path/to/your_items.jsonl --results results/scores.jsonl
```

The mock adapter is an interface smoke check, not a model-performance experiment.
Real adapters require the corresponding credentials. Symbolic equation grading
is supported, but the manuscript's main experiment used final-value scoring for
equation-type items; enabling equation requirements defines a different metric.

## Tests

The included tests cover categorical extraction, equation equivalence, combined
equation/value requirements, and corroboration utilities without benchmark files
or API calls:

```sh
python -m pip install pytest
python -m pytest tests
```

## Source layout

- `synbiobench/core`: provider adapters, scoring, symbolic comparison,
  psychometrics, and independent-solver corroboration utilities.
- `synbiobench/gene_circuit`: reusable numerical solvers, grammar,
  paraphrase verification, and item utilities.
- `synbiobench/mfa/oracle.py`: linear algebra, reconciliation, FBA, and sensitivity.
- `tests`: software regression tests using artificial inputs.
