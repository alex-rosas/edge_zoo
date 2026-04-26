"""
quantize/onnx_interrogate.py
----------------------------
ONNX graph interrogation: quantized vs FP32 operator classification.

Loads an exported ONNX graph and classifies every node as quantized or
FP32. The result is one of three tables in the engineering report
(proposal, Section 4.2, 4.4).

Why interrogation and not just verification:
  onnx.checker.check_model() verifies the graph is well-formed.
  Interrogation goes further: it identifies which operators were not
  lowered to INT8 during export. Each unquantized operator is a finding:
    - a PyTorch operator with no ONNX INT8 equivalent at opset 17
    - an activation function the quantizer did not annotate
    - a graph lowering failure specific to the architecture

  These findings motivate the question a hardware compiler engineer
  would ask: "Which subgraphs would your toolchain reject, and what
  would you do about them?"

Quantized operator recognition:
  ONNX opset 17 encodes INT8 quantization through a small set of
  operator types. Any node whose op_type appears in QUANTIZED_OPS
  is counted as quantized; all others are FP32.
"""

import onnx
import pandas as pd
from pathlib import Path
from typing import Dict, List


# ─── Quantized operator names in ONNX opset 17 ───────────────────────────────
# These are the op_types that represent INT8 computation or the
# quantize/dequantize boundary in an ONNX graph. Any node not in this
# set is executing in FP32.

QUANTIZED_OPS = {
    "QLinearConv",          # INT8 convolution
    "QLinearMatMul",        # INT8 matrix multiply
    "QLinearAdd",           # INT8 elementwise add
    "QLinearMul",           # INT8 elementwise multiply
    "QLinearAveragePool",   # INT8 average pooling
    "QLinearGlobalAveragePool",
    "QLinearLeakyRelu",
    "QLinearSigmoid",
    "QuantizeLinear",       # FP32 → INT8 boundary
    "DequantizeLinear",     # INT8 → FP32 boundary
    "ConvInteger",          # alternative INT8 conv (some exporters)
    "MatMulInteger",        # alternative INT8 matmul
}

# Operators that are expected to remain in FP32 — structural graph nodes
# that carry no arithmetic weight. Flagging these as "FP32 operators"
# would be misleading; they are excluded from the FP32 count.
STRUCTURAL_OPS = {
    "Reshape", "Flatten", "Transpose", "Squeeze", "Unsqueeze",
    "Gather", "Slice", "Concat", "Shape", "Cast", "Constant",
    "Identity",
}


# ─── Graph interrogation ──────────────────────────────────────────────────────

def interrogate_onnx_graph(onnx_path: str) -> Dict:
    """
    Load an ONNX graph and classify every node as quantized, FP32, or structural.

    Walks the graph node list and checks each node's op_type against
    QUANTIZED_OPS and STRUCTURAL_OPS. Returns a structured summary
    with counts and the full list of FP32 arithmetic nodes — the ones
    that matter for a hardware compiler.

    Args:
        onnx_path: Path to the exported .onnx file.

    Returns:
        {
            "model_name":      str   — derived from filename
            "total_nodes":     int   — all nodes in the graph
            "quantized_nodes": int   — nodes in QUANTIZED_OPS
            "fp32_nodes":      int   — arithmetic nodes NOT in QUANTIZED_OPS
            "structural_nodes":int   — nodes in STRUCTURAL_OPS
            "quantized_ops":   List  — op_type counts for quantized nodes
            "fp32_ops":        List  — op_type counts for FP32 arithmetic nodes
            "coverage_pct":    float — quantized / (quantized + fp32) * 100
        }
    """
    path = Path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    model      = onnx.load(str(path))
    graph      = model.graph
    model_name = path.stem   # e.g. "resnet18_int8"

    quantized: List[str] = []
    fp32:      List[str] = []
    structural:List[str] = []

    for node in graph.node:
        op = node.op_type
        if op in QUANTIZED_OPS:
            quantized.append(op)
        elif op in STRUCTURAL_OPS:
            structural.append(op)
        else:
            fp32.append(op)

    total      = len(graph.node)
    n_quant    = len(quantized)
    n_fp32     = len(fp32)
    n_struct   = len(structural)

    coverage = (
        n_quant / (n_quant + n_fp32) * 100
        if (n_quant + n_fp32) > 0
        else 0.0
    )

    # Count occurrences of each op type for the report.
    def _count(ops: List[str]) -> List[Dict]:
        from collections import Counter
        return [
            {"op_type": op, "count": cnt}
            for op, cnt in Counter(ops).most_common()
        ]

    return {
        "model_name":       model_name,
        "total_nodes":      total,
        "quantized_nodes":  n_quant,
        "fp32_nodes":       n_fp32,
        "structural_nodes": n_struct,
        "quantized_ops":    _count(quantized),
        "fp32_ops":         _count(fp32),
        "coverage_pct":     coverage,
    }


# ─── Display ──────────────────────────────────────────────────────────────────

def print_interrogation_report(result: Dict) -> None:
    """
    Print a formatted interrogation report for one model.

    The coverage percentage is the headline number: what fraction of
    arithmetic operators were successfully lowered to INT8. The FP32
    operator list is the finding: each entry is a specific operator that
    a hardware compiler would need to handle as a FP32 subgraph.

    Example output:
      ── ONNX interrogation: resnet18_int8 ──
        Total nodes:      142
        Quantized nodes:   98   (QLinearConv, QuantizeLinear, ...)
        FP32 nodes:        12
        Structural nodes:  32
        Coverage:         89.1%

      FP32 arithmetic operators:
        Relu              ×  8   ← activation not in INT8 op set
        BatchNormalization×  4   ← should have been folded; investigate
    """
    print(f"\n── ONNX interrogation: {result['model_name']} ──")
    print(f"  Total nodes:       {result['total_nodes']:>5}")
    print(f"  Quantized nodes:   {result['quantized_nodes']:>5}")
    print(f"  FP32 nodes:        {result['fp32_nodes']:>5}")
    print(f"  Structural nodes:  {result['structural_nodes']:>5}")
    print(f"  Coverage:          {result['coverage_pct']:>5.1f}%")

    if result["fp32_ops"]:
        print("\n  FP32 arithmetic operators (each is a finding):")
        for entry in result["fp32_ops"]:
            op    = entry["op_type"]
            count = entry["count"]
            print(f"    {op:<30} × {count:>3}")
    else:
        print("\n  No FP32 arithmetic operators — full quantization coverage.")

    if result["quantized_ops"]:
        print("\n  Quantized operators:")
        for entry in result["quantized_ops"]:
            op    = entry["op_type"]
            count = entry["count"]
            print(f"    {op:<30} × {count:>3}")


# ─── Multi-model runner ───────────────────────────────────────────────────────

def run_onnx_interrogation(onnx_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Interrogate all three exported ONNX graphs and return a summary DataFrame.

    Entry point called by the benchmark script. Runs interrogate_onnx_graph()
    for each model, prints individual reports, and assembles the result into
    the ONNX interrogation table (results/onnx_graph.csv).

    Args:
        onnx_paths: Dict mapping model name → onnx file path.
                    e.g. {"resnet18": "onnx_models/resnet18_int8.onnx", ...}

    Returns:
        pd.DataFrame with one row per model:
            model_name, total_nodes, quantized_nodes, fp32_nodes,
            structural_nodes, coverage_pct
    """
    records = []

    for name, path in onnx_paths.items():
        result = interrogate_onnx_graph(path)
        print_interrogation_report(result)

        records.append({
            "model_name":       result["model_name"],
            "total_nodes":      result["total_nodes"],
            "quantized_nodes":  result["quantized_nodes"],
            "fp32_nodes":       result["fp32_nodes"],
            "structural_nodes": result["structural_nodes"],
            "coverage_pct":     result["coverage_pct"],
        })

    return pd.DataFrame(records)