# Conventions appendix verification

The existing local Ward suite was rebuilt and run at 40 digits on September 23, 2026. All 8,238 values passed, including physical and auxiliary NS/R residue identities, production-form comparisons, the Ramond odd-ground identity, and both embedded Virasoro Ward identities. Maximum scaled error: 3.0084e-31 in the embedded Virasoro check. Runtime excluding compilation: 8.13 seconds.

`local_results.json` contains the numerical results. `manifest.json` records the build command and source hashes. No block benchmark or momentum integration was rerun. Both paper drafts were left unchanged.

The appendix was subsequently shortened to reference material already in the long draft. Only documentation and its PDF were revised; the numerical suite was not rerun. The manifest records the previous and revised documentation hashes. The revised PDF compiles without undefined references or overfull boxes, and every revised appendix page was visually inspected.
