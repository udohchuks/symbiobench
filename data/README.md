# SynBioBench dataset

The primary collection contains 346 circuit questions and 59 metabolic-flux questions (405 total). Questions, reference answers, grading rules, and original task metadata are provided. The separate circuit hard set has 14 questions in each of two matched presentations.

The manuscript main evaluation uses `gene_circuit_main.jsonl` (250 questions) and `mfa_main.jsonl` (29 questions), 279 total. `evaluation_manifest.csv` records inclusion and exclusion. The original selection procedure for the archived 250-circuit pool has not been recovered; it is not presented as a random sample.

The full MFA file retains 18 questions referring to external network descriptions missing from the archived rendered prompts and 12 qualitative metabolic-deletion questions. These 30 questions were excluded from the main accuracy calculation. The release does not restore the missing external descriptions.

`circuit_hard_biological.jsonl` and `circuit_hard_equations.jsonl` share item IDs and answers. The three structural pairs used in the manuscript are contained in this 14-item set.

Experts were consulted during question development, as reported by the authors. Time constraints prevented a separate formal expert-validation study or human-solver evaluation. Consultation does not establish item-level correctness or uniqueness of interpretation.

The earlier flux question bank contains a mixture of material whose item-level outside-source attribution has not been fully recovered. Existing provenance fields are original metadata, not a certification of source ownership or complete attribution. No new dataset licence or assertion that all wording is original is made in this release.

Saved model responses are not included. These files enable inspection and new evaluations but do not alone reproduce the manuscript's archived model scores. See the repository scoring code for the accepted schema. `inventory.json` records source-file counts and checksums.
