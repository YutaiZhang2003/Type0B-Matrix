#!/usr/bin/env python3
"""Render the sphere-crossing CSV as an inline D3 visualization fragment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
DEFAULT_INPUT = (
    WORKSPACE_DIR / "Data Set" / "crossing_equation" / "spin23_sphere_crossing.csv"
)


def read_data(paths: list[Path]) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    orders = sorted(
        {int(row["q_order"]) for row in rows if row["path"] == "real_axis"}
    )
    highest_order = max(orders)

    def select(path_name: str, order: int) -> list[dict[str, str]]:
        return [
            row
            for row in rows
            if row["path"] == path_name and int(row["q_order"]) == order
        ]

    highest_rows = select("real_axis", highest_order)
    complex_path = select("complex_x_plus_i_height", highest_order)
    return {
        "highest_order": highest_order,
        "x": [float(row["z_real"]) for row in highest_rows],
        "q": [float(row["q_s_abs"]) for row in highest_rows],
        "gs": [float(row["g_s_real"]) for row in highest_rows],
        "gt": [float(row["g_t_real"]) for row in highest_rows],
        "residuals": {
            str(order): [
                float(row["relative_crossing_residual"])
                for row in select("real_axis", order)
            ]
            for order in orders
        },
        "complex": [
            float(row["relative_crossing_residual"]) for row in complex_path
        ],
    }


def fragment(data: dict[str, object]) -> str:
    encoded = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    x_values = data["x"]
    x_min = min(x_values)
    x_max = max(x_values)
    return f"""
<div id="spin23-z-dependence-viz" class="spin23-crossing-viz">
  <style>
    .spin23-crossing-viz {{ --ink:var(--foreground); --muted:var(--foreground);
      --grid:var(--border); --blue:var(--viz-series-1); --orange:var(--viz-series-2);
      color:var(--ink); background:transparent; padding:4px;
      font:14px/1.4 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .spin23-crossing-viz * {{ box-sizing:border-box; }}
    .spin23-crossing-viz h2 {{ margin:0; font-size:20px; letter-spacing:-.02em; }}
    .spin23-crossing-viz .sub {{ color:var(--muted); margin:4px 0 12px; }}
    .spin23-crossing-viz .chart {{ position:relative; width:100%; min-height:620px; }}
    .spin23-crossing-viz svg {{ width:100%; height:620px; display:block; overflow:visible; }}
    .spin23-crossing-viz .axis text {{ fill:var(--muted); font-size:11px; }}
    .spin23-crossing-viz .axis path,.spin23-crossing-viz .axis line {{ stroke:var(--border); }}
    .spin23-crossing-viz .grid line {{ stroke:var(--grid); stroke-opacity:.72; }}
    .spin23-crossing-viz .grid path {{ display:none; }}
    .spin23-crossing-viz .label {{ fill:var(--muted); font-size:12px; }}
    .spin23-crossing-viz .legend {{ fill:var(--ink); font-size:11px; }}
    .spin23-crossing-viz .tip {{ position:absolute; pointer-events:none; opacity:0;
      background:var(--popover); color:var(--popover-foreground); border:1px solid var(--border); border-radius:10px;
      padding:9px 11px; font-size:12px; min-width:190px;
      font-variant-numeric:tabular-nums; }}
    .spin23-crossing-viz .tip b {{ display:inline-block; min-width:72px; }}
    .spin23-crossing-viz .foot {{ color:var(--muted); font-size:12px; margin:4px 0 0; }}
  </style>
  <h2>Spin(23) sphere crossing on {x_min:.2f} ≤ z ≤ {x_max:.2f}</h2>
  <div class="chart"><svg role="img" aria-label="Direct and crossed sphere correlators and crossing residual versus z"></svg><div class="tip tooltip" role="tooltip"></div></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("spin23-z-dependence-viz");
  const data = {encoded};
  const svg = d3.select(root).select("svg");
  const tip = d3.select(root).select(".tip");
  const rootStyle = getComputedStyle(root);
  const orderKeys = Object.keys(data.residuals).sort((a,b)=>+a-+b);
  const colors = Object.fromEntries(orderKeys.map((order,i)=>[order,rootStyle.getPropertyValue(`--viz-series-${{(i%6)+1}}`).trim()]));
  colors.complex=rootStyle.getPropertyValue("--viz-series-6").trim();
  const series = data.x.map((x,i) => ({{x,q:data.q[i],gs:data.gs[i],gt:data.gt[i],i}}));
  const nearest = d3.bisector(d => d.x).center;

  function render() {{
    const width = Math.max(360, root.querySelector(".chart").clientWidth);
    const height = 620;
    const compact = width < 520;
    const margin = {{left:compact?66:72,right:compact?12:24,top:48,bottom:46}};
    const top = {{y0:54,y1:268}}, bottom = {{y0:compact?392:370,y1:height-margin.bottom}};
    svg.attr("viewBox",`0 0 ${{width}} ${{height}}`).selectAll("*").remove();
    const css = getComputedStyle(root);
    const ink=css.getPropertyValue("--ink").trim(), muted=css.getPropertyValue("--muted").trim();
    const x = d3.scaleLinear().domain(d3.extent(data.x)).range([margin.left,width-margin.right]);
    const gExtent=d3.extent(data.gs.concat(data.gt)), useLogG=gExtent[0]>0 && gExtent[1]/gExtent[0]>15, gPad=(gExtent[1]-gExtent[0])*0.05;
    const yG = (useLogG?d3.scaleLog().domain([gExtent[0]*0.9,gExtent[1]*1.1]):d3.scaleLinear().domain([gExtent[0]-gPad,gExtent[1]+gPad]).nice()).range([top.y1,top.y0]);
    const residualValues = Object.values(data.residuals).flat().concat(data.complex);
    const residualDomainValues=residualValues.concat([1e-7]).filter(v=>v>0);
    const rLo=Math.floor(Math.log10(d3.min(residualDomainValues))), rHi=Math.ceil(Math.log10(d3.max(residualDomainValues)));
    const yR = d3.scaleLog().domain([10**rLo,10**rHi]).range([bottom.y1,bottom.y0]);
    const residualTicks=d3.range(rLo,rHi+1,2).map(e=>10**e);
    const xTicks = x.ticks(compact?4:5);
    const qAt = z => data.q[Math.max(0,Math.min(data.q.length-1,nearest(series,z)))];

    svg.append("rect").attr("data-chart-frame","").attr("x",margin.left).attr("y",top.y0).attr("width",width-margin.left-margin.right).attr("height",top.y1-top.y0).attr("fill","none").attr("stroke",css.getPropertyValue("--border").trim());
    svg.append("rect").attr("data-chart-frame","").attr("x",margin.left).attr("y",bottom.y0).attr("width",width-margin.left-margin.right).attr("height",bottom.y1-bottom.y0).attr("fill","none").attr("stroke",css.getPropertyValue("--border").trim());

    svg.append("g").attr("class","grid").attr("transform",`translate(0,${{top.y1}})`)
      .call(d3.axisBottom(x).tickValues(xTicks).tickSize(-(top.y1-top.y0)).tickFormat(""));
    svg.append("g").attr("class","grid").attr("transform",`translate(${{margin.left}},0)`)
      .call(d3.axisLeft(yG).ticks(5).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
    svg.append("g").attr("class","grid").attr("transform",`translate(0,${{bottom.y1}})`)
      .call(d3.axisBottom(x).tickValues(xTicks).tickSize(-(bottom.y1-bottom.y0)).tickFormat(""));
    svg.append("g").attr("class","grid").attr("transform",`translate(${{margin.left}},0)`)
      .call(d3.axisLeft(yR).tickValues(residualTicks).tickSize(-(width-margin.left-margin.right)).tickFormat(""));
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${{top.y0}})`)
      .call(d3.axisTop(x).tickValues(xTicks).tickFormat(z => d3.format(".4f")(qAt(z))));
    svg.append("g").attr("class","axis").attr("transform",`translate(${{margin.left}},0)`).call(d3.axisLeft(yG).ticks(5));
    svg.append("g").attr("class","axis").attr("transform",`translate(0,${{bottom.y1}})`).call(d3.axisBottom(x).tickValues(xTicks));
    svg.append("g").attr("class","axis").attr("transform",`translate(${{margin.left}},0)`)
      .call(d3.axisLeft(yR).tickValues(residualTicks).tickFormat(d3.format(".0e")));

    const lineG = key => d3.line().x(d=>x(d.x)).y(d=>yG(d[key]));
    svg.append("path").datum(series).attr("fill","none").attr("stroke",css.getPropertyValue("--blue").trim()).attr("stroke-width",2.7).attr("d",lineG("gs"));
    svg.append("path").datum(series).attr("fill","none").attr("stroke",css.getPropertyValue("--orange").trim()).attr("stroke-width",2).attr("stroke-dasharray","7 5").attr("d",lineG("gt"));
    const lineR = d3.line().x((d,i)=>x(data.x[i])).y(d=>yR(Math.max(d,1e-13)));
    Object.entries(data.residuals).forEach(([order,values]) => svg.append("path").datum(values).attr("fill","none").attr("stroke",colors[order]).attr("stroke-width",+order===data.highest_order?2.5:1.7).attr("d",lineR));
    svg.append("path").datum(data.complex).attr("fill","none").attr("stroke",colors.complex).attr("stroke-width",2).attr("stroke-dasharray","3 4").attr("d",lineR);
    svg.append("line").attr("x1",margin.left).attr("x2",width-margin.right).attr("y1",yR(1e-7)).attr("y2",yR(1e-7)).attr("stroke",muted).attr("stroke-dasharray","6 5");
    svg.append("text").attr("class","label").attr("x",width-margin.right).attr("y",yR(1e-7)-6).attr("text-anchor","end").text("review target 10⁻⁷");
    svg.append("text").attr("class","label axis-title").attr("data-axis","x").attr("x",width/2).attr("y",20).attr("text-anchor","middle").text("elliptic nome |q(z)|");
    svg.append("text").attr("class","label axis-title").attr("data-axis","y").attr("transform",`translate(17,${{(top.y0+top.y1)/2}}) rotate(-90)`).attr("text-anchor","middle").text(`integrated correlator Re G${{useLogG?" (log scale)":""}}`);
    svg.append("text").attr("class","label axis-title").attr("data-axis","y").attr("transform",`translate(17,${{(bottom.y0+bottom.y1)/2}}) rotate(-90)`).attr("text-anchor","middle").text("relative crossing residual");
    svg.append("text").attr("class","label axis-title").attr("data-axis","x").attr("x",width/2).attr("y",height-5).attr("text-anchor","middle").text("cross-ratio x = Re z");

    const legends=[{{label:"Gs(z)",color:css.getPropertyValue("--blue").trim()}},{{label:"Gt(1−z)",color:css.getPropertyValue("--orange").trim(),dash:true}}];
    legends.forEach((item,i)=>{{ const gx=margin.left+20+i*105; svg.append("line").attr("x1",gx).attr("x2",gx+24).attr("y1",top.y0+16).attr("y2",top.y0+16).attr("stroke",item.color).attr("stroke-width",2.5).attr("stroke-dasharray",item.dash?"7 5":null); svg.append("text").attr("class","legend").attr("x",gx+30).attr("y",top.y0+20).text(item.label); }});
    orderKeys.forEach((order,i)=>{{ const gx=compact?margin.left+i*69:margin.left+i*83; const gy=bottom.y0-(compact?35:17); svg.append("line").attr("x1",gx).attr("x2",gx+18).attr("y1",gy).attr("y2",gy).attr("stroke",colors[order]).attr("stroke-width",2); svg.append("text").attr("class","legend").attr("x",gx+22).attr("y",gy+4).text(compact?order:`order ${{order}}`); }});
    const cx=compact?margin.left:width-margin.right-132, cy=bottom.y0-17; svg.append("line").attr("x1",cx).attr("x2",cx+24).attr("y1",cy).attr("y2",cy).attr("stroke",colors.complex).attr("stroke-width",2).attr("stroke-dasharray","3 4"); svg.append("text").attr("class","legend").attr("x",cx+29).attr("y",cy+4).text("z=x+0.12i");

    const cursor = svg.append("line").attr("data-chart-hover-guide","").attr("y1",top.y0).attr("y2",bottom.y1).attr("stroke",ink).attr("stroke-opacity",.35).style("display","none");
    svg.append("rect").attr("data-chart-hit","").attr("data-chart-hover-overlay","cross-series").attr("x",margin.left).attr("y",top.y0).attr("width",width-margin.left-margin.right).attr("height",bottom.y1-top.y0).attr("fill","transparent")
      .on("pointermove", event => {{ const [mx]=d3.pointer(event); const i=nearest(series,x.invert(mx)); const d=series[i]; cursor.style("display",null).attr("x1",x(d.x)).attr("x2",x(d.x)); tip.style("opacity",1).style("left",`${{Math.min(width-210,x(d.x)+12)}}px`).style("top",`${{Math.max(70,d3.pointer(event,root)[1]-38)}}px`).html(`<b>z</b>${{d.x.toFixed(4)}}<br><b>|q(z)|</b>${{d.q.toFixed(6)}}<br><b>Gs</b>${{d.gs.toPrecision(9)}}<br><b>Gt</b>${{d.gt.toPrecision(9)}}<br><b>order ${{data.highest_order}} ε</b>${{data.residuals[String(data.highest_order)][i].toExponential(3)}}`); }})
      .on("pointerleave",()=>{{cursor.style("display","none");tip.style("opacity",0);}});
  }}
  render();
  new ResizeObserver(render).observe(root.querySelector(".chart"));
}})();
</script>
""".strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--comparison-input", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        fragment(read_data([args.input, *args.comparison_input])), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
