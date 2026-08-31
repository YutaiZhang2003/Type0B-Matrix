#!/usr/bin/env python3
"""Add color keys, a unity line and diagnostic caveats to the numerical plot.

Kept separate so the immutable numerical worker fingerprint is unchanged.
This generates a derived SVG without changing any numerical data.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def annotate(summary, source_svg, output_svg):
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    tree = ET.parse(source_svg)
    group = tree.getroot().find(f"{{{ns}}}g")
    if group is None:
        raise ValueError("plot group not found")
    for element in group.findall(f"{{{ns}}}text"):
        value = element.text or ""
        if value.startswith("All-NS c-recursion convergence:"):
            element.text = "All-NS c-recursion order test against the original NSRR baseline"
        elif value.startswith("NSRR fixed"):
            element.text = value.replace("NSRR fixed", "NSRR baseline")
            element.set("fill", "#ba432f")
        elif value.startswith("all-NS R="):
            r = int(value.split("=")[1])
            color = {8: "#777777", 12: "#d77a13", 16: "#176dad"}.get(r, "#84469b")
            element.set("fill", color)
        elif value.startswith("Same local-frame free normalization"):
            element.text = "Original NSRR baseline unchanged; its spin projection and edge-order map are under audit. Cutoff changes are not error bars."
    n = max(summary["config"]["quadrature_orders"])
    ys = [r["source_over_target"] for r in summary["rows"] if r["quadrature_order"] == n] + [1.]
    low, high = min(ys), max(ys)
    margin = .10*(high-low or max(abs(high), 1e-10))
    low, high = low-margin, high+margin
    y = 400+195*(high-1)/(high-low)
    ET.SubElement(group, f"{{{ns}}}path", {
        "d": f"M 125 {y} h 645", "stroke": "#666666", "stroke-dasharray": "4 5"})
    label = ET.SubElement(group, f"{{{ns}}}text", {
        "x": "820", "y": str(y+4), "font-size": "12", "fill": "#666666"})
    label.text = "modular equality = 1"
    tree.write(output_svg, encoding="unicode", xml_declaration=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-svg", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    args = parser.parse_args()
    annotate(json.loads(args.summary.read_text()), args.source_svg, args.output_svg)
