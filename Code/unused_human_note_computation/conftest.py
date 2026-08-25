"""Keep provenance-only experiments out of repository-wide pytest runs."""

collect_ignore_glob = [
    "legacy_direct_chi_anchor/*.py",
    "pbw_c_recursion_double_virasoro_crosscheck/*.py",
]
