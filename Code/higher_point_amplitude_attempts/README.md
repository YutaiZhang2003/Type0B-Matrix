# Higher-point amplitude attempts

This folder isolates exploratory computations for amplitudes beyond the
established lower-point checks. Results here must not be treated as frozen
or converged unless explicitly supported by their validation records.

The fixed complex-frequency production protocol permits a preliminary
matrix comparison before an extended convergence study; no epsilon-to-zero
limit is required. The separate `complex_frequency_comparison/` postprocessor
preserves the independent worldsheet inputs and labels comparisons preliminary.

The reusable conformal-block and Liouville machinery remains in
`Code/c_Recursion/`.  Each attempt subfolder contains its own integrand,
kinematic audits, numerical drivers, plots, and regression tests.
