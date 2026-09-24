"""Count level-20 PBW work and apply saved fits; never evaluate a PBW block."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "Code/theta_fermion_ccy/count_level_work.py"
BASELINE = HERE / "pbw_estimates.json"
spec = importlib.util.spec_from_file_location("integer_work_counter", SOURCE)
counter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counter)
baseline = json.loads(BASELINE.read_text())
work = counter.pbw_counts(20)
features = {
    "gram_entries": work["gram_entries"],
    "requested_tensor_entries": work["requested_three_point_tensor_entries"],
    "sum_dimension_cubed": work["gram_inverse_sum_dimension_cubed"],
    "dense_contraction_products": work["dense_contraction_complex_multiplications"],
    "json_record_writes_proxy": work["monomials"]*(work["monomials"]+1)//2,
    "monomials": work["monomials"],
}
report = {
    "level": 20,
    "scope": "Exact integer workload counts with previously saved level-10 cost fits",
    "new_numerical_pbw_gram_or_ward_evaluations": 0,
    "work": work,
    "predictions": {},
    "limitations": baseline["limitations"],
    "precision_comparison": baseline["precision_comparison"],
    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (SOURCE, BASELINE, Path(__file__))},
}
for backend, calibration in baseline["calibrations"].items():
    models = []
    for model in calibration["models"]:
        parts = {name: rate*features[name] for name, rate in
                 zip(model["features"], model["fitted_seconds_per_feature_unit"])}
        minus = sum(parts.values())
        plus = [minus-parts["requested_tensor_entries"]/2, minus]
        models.append({
            "name": "full_features" if "sum_dimension_cubed" in parts else "lean",
            "fitted_cost_contributions_seconds_not_measured_stages": parts,
            "eta_eta_prime_minus_seconds": minus,
            "eta_eta_prime_plus_scenarios_seconds": plus,
            "eta_eta_prime_minus_hours": minus/3600,
            "eta_eta_prime_plus_scenarios_hours": [x/3600 for x in plus],
        })
    report["predictions"][backend] = {"precision_bits": calibration["precision_bits"],
                                     "models": models}
output = HERE / "pbw_level20_estimate.json"
output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({"work": {key: value for key,value in work.items()
                           if not isinstance(value,list)},
                  "predictions": report["predictions"]}, indent=2))
