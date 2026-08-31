# Batched c-recursion evaluation

The failed 12-hour run had finished constructing its face tables. Later face
samples still spent roughly one hour evaluating scalar blocks, with about
152,544 SQLite loads and 338,016 block evictions per sample. Reducing the
coefficient precision does not address that repeated work.

`fivepoint_batch.py` separates compilation from evaluation. For a fixed
ordering and momentum grid it retains coefficient tensors, internal weights
and structure products. The per-point work constructs the geometry and PCO
terms once, evaluates both chiral polynomials in NumPy, and contracts the
nonchiral cocycle. The antiholomorphic coefficients and external weights are
NOT complex conjugated at analytically continued energies; the logarithmic
lifts are conjugated. Recursion remains the original c-recursion at the same
working precision and truncation.

The tensor cache has a byte budget independent of the small block-object LRU.
Final arbitrary-precision coefficients remain durable in SQLite. A conservative
rounding/cancellation indicator sends unsafe momentum rows to the scalar
implementation. This indicator is a numerical safeguard, not a rigorous bound
on the final integral. Momentum, moduli and truncation convergence remain
separate questions. An epsilon-to-zero study is needed only for a real-energy
boundary value, not for the requested fixed-complex-energy comparison.

Inside a collar the tensor evaluator partitions each chiral polynomial into
primary and excited states. The forest remainder sums the disjoint products
with at least one excited half on every subtracted edge. This avoids computing
the small remainder by cancelling F-P1-P2+P12. The path is enabled only for a
full rectangular cutoff, where it matches the existing projected forest.
A boundary test also exposed ordinary-complex arithmetic in the factorized
endpoint vertex; both scalar and tensor paths now convert those weights to
mpmath before forming the vertex.

`migrate_fivepoint_coefficients.py` reads the frozen old source tree, verifies
identical transitive recursion dependencies and identical compilation and
serialization methods, and copies the old database with an audit record.
The original database is unchanged. Old scalar checkpoints are deliberately
not migrated. New coefficient identities separate coefficient code from
integration code; scalar checkpoints still fingerprint all runtime sources.

Validation includes all 18 face orbits, both endpoint projections, corners,
negative-real branch lifts, multiple bulk moduli, finite parts with momentum
subtraction, and a 45-digit forest-subtraction reference. The initial small
face profile improved warm evaluation from 0.0713 s to 0.00118 s (about 60x),
with differences of order 1e-15 relative to the scalar result. This is not a
prediction of the complete cluster run time; the production-grid benchmark
records that separately.

The optimized calculation still computes the all-NS contribution
`integral d2z d2w I_NS`. Its literal diagram carries `(i/64) g_s^5 C_S2 delta(E)`.
No matrix-model amplitude is used as an input, normalization fit or convergence
target, and this runtime change does not establish equality of the other
NS/R diagrams. A separate postprocessor can compare the all-tachyon result
directly at the finite complex energies, without an epsilon-to-zero limit.
