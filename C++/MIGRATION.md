# Completed C++ migration

The two current production pipelines are native C++17: ordinary genus-two
NS–R–R theta Ramond blocks and the Theta v_(1/2) inserted recovery pipeline.
Both support machine and configurable higher precision. No production stage
invokes Python. Historical alternate methods and unrelated research scripts
under `Code/` remain Python references.

| C++ source | Responsibility / Python counterpart |
| --- | --- |
| `src/main.cpp` | CLI, precision selection, atomic JSON output |
| `include/ramond/number.hpp` | Machine/MPC scalar types and exact-input parsing |
| `include/ramond/storage.hpp` | Sparse series, partitions, parity-index storage |
| `include/ramond/linalg.hpp` | LAPACK rank selection and multiprecision residual refinement |
| `include/ramond/free_field.hpp` | `ramond_branching_recursion/compute_target.py` and `theta_fermion_ccy/action_optimization.py`: mode actions, embeddings, action solves, shared caches |
| `include/ramond/anchors.hpp` | Low physical/fermionic three-point forms and branching anchors used by `ramond_zero_mode_recovery` |
| `include/ramond/branching.hpp` | `theta_fermion_ccy/outer_branching.py` and `middle_branching.py`: Ward systems, reflected actions, middle recurrence |
| `include/ramond/ccy.hpp` | `theta_fermion_ccy/forward_ccy.py`: ordinary and punctured forward c-recursion, global factors, A and fusion caches |
| `include/ramond/fermion.hpp` | `theta_fermion_ccy/direct_fermion.py`, ordinary direct fermion sewing, parity star operations |
| `include/ramond/schottky.hpp` | `theta_fermion_ccy/schottky_vacuum.py`: exact primitive Schottky product |
| `include/ramond/pipeline.hpp` | Ordinary restricted-inverse assembly and inserted diagonal-target assembly, physical recovery, segment timings |

The abandoned C prototype has been removed. The executable is `C++/bin/ramond`;
Python implementations and manuscript were not changed by this migration.
Build instructions, conventions, measurements, and validation limits are in
[README.md](README.md).
