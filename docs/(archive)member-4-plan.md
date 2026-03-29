# Member 4 Plan

## Role

Data Analysis and Technical Writing

This role is responsible for turning the theoretical and empirical results into a clear paper, beginning with an early outline and then using that outline to drive the figure set.

## Primary Goals

- produce an early paper outline
- use that outline to determine which figures and tables are necessary
- create the theoretical and empirical amortization visualizations
- assemble contributions from the rest of the team into a coherent report

## Work Plan

1. Draft the paper outline first.

- Produce an early outline before the figure set is locked.
- Define the main story of the paper so the team knows what evidence is required.
- Identify the core claims, what each section needs to argue, and which results are necessary to support those claims.

Suggested outline:

- Introduction
- Background on KV-cache fragmentation and amortization
- Theoretical model for waste and amortization
- System design and telemetry collection
- Experimental methodology
- Results for MHA and GQA baselines
- Results for MLA
- Discussion
- Limitations
- Conclusion

2. Use the outline to lock the figure set.

- Once the outline is in place, identify the exact figures and tables needed for each section.
- Make sure every major claim in the outline has a corresponding figure, table, derivation, or experiment.
- Remove low-value plots and prioritize the visuals that directly support the paper's argument.

3. Build the plotting workflow.

- Standardize the expected CSV inputs from telemetry and benchmark runs.
- Create reusable scripts or notebooks for plotting `% Waste` against sequence length.
- Keep labels, colors, line styles, legends, and axis ranges consistent across all plots.

4. Generate the core figures.

- Create theoretical amortization curves.
- Create empirical amortization curves.
- Create theory-vs-empirical overlays.
- Create cross-architecture comparisons for MHA, GQA, and MLA.
- Create summary tables or bar charts for key thresholds such as the sequence length where fragmentation drops below `2%`.

5. Coordinate with the rest of the team.

- Gather the mathematical derivations and baseline validation results from Member 3.
- Gather the MLA implementation details and experiment notes from Member 1.
- Gather the telemetry and benchmarking pipeline details from Member 2.
- Keep a running checklist of missing artifacts needed to complete the paper.

6. Write the results narrative.

- For each figure, write a short explanation of:
  - what is plotted
  - what trend matters
  - whether theory and experiment agree
  - why the architecture behaves that way
- Turn the figures into an argument, not just a gallery of plots.

7. Assemble the draft paper.

- Convert the outline into a working draft as soon as the first figures are available.
- Integrate theoretical results, experimental setup, plots, and interpretation into one document.
- Keep notation and terminology consistent across sections.

8. Final polish responsibilities.

- Check that every claim is supported by a derivation, result, or citation.
- Standardize notation for waste, fragmentation, amortization, sequence length, and architecture names.
- Ensure the final draft reads like one paper instead of several stitched-together sections.

## Deliverables

- an early paper outline
- a locked figure and table plan derived from the outline
- plotting scripts or notebooks
- the full figure set for theory and experiment
- a compiled draft integrating results from all group members

## Suggested First Milestones

- produce the first paper outline
- identify the minimum figure set required by that outline
- create one template plot for `% Waste` vs sequence length
- start a shared results inventory for the team

## Notes

- The outline should come before figure lock.
- The role is not only about making plots look good; it is about shaping the paper's argument and making sure the evidence matches that argument.
