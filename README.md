# rsomics-bipartite-clustering

Bipartite clustering coefficients — value-exact Rust reimplementation of
[`networkx.algorithms.bipartite.cluster`](https://networkx.org/documentation/stable/reference/algorithms/bipartite.html)
(networkx 3.6.1).

Three operations selected via `--op`:

| `--op` | Description |
|---|---|
| `latapy` (default) | Per-node Latapy clustering (dot/min/max mode) |
| `average` | Mean Latapy clustering over a node set |
| `robins-alexander` | 4·C₄/L₃ whole-graph coefficient |

## Usage

```
rsomics-bipartite-clustering [OPTIONS] [EDGELIST]

Options:
  --op <OP>          latapy | average | robins-alexander  [default: latapy]
  --mode <MODE>      dot | min | max  [default: dot]
  --nodes <LABELS>   comma-separated node labels (latapy/average only)
  --json             emit rsomics-common JSON envelope
```

Edge list: one `u v` per line, `#`/blank ignored, self-loops dropped, parallel edges
collapsed (`nx.Graph` semantics). The graph is treated as undirected and bipartite
by structure (no explicit partition required — matching nx behaviour).

## Value-exactness

The Latapy modes compute integer set-cardinalities (intersection / union / min / max)
then perform a single float division — result is ≤1 ULP from networkx. The
Robins-Alexander coefficient is 4·(integer)/( integer), also ≤1 ULP.

All tests are verified against hardcoded constants captured from networkx 3.6.1
(Python 3.11). The Davis Southern Women graph reproduces `round(RA, 3) = 0.468`
as documented in the nx docstring.

## Origin

This crate is an independent Rust reimplementation of the bipartite clustering
functions in NetworkX based on:

- Latapy, M., Magnien, C., and Del Vecchio, N. (2008). Basic notions for the
  analysis of large two-mode networks. *Social Networks* 30(1), 31–48.
- Robins, G. and Alexander, M. (2004). Small worlds among interlocking directors:
  Network structure and distance in bipartite graphs. *Computational & Mathematical
  Organization Theory* 10(1), 69–94.
- The networkx 3.6.1 source (MIT licence): `networkx/algorithms/bipartite/cluster.py`
  — the exact set-formula and motif-counting helpers were matched from this source.
- Black-box behaviour testing against networkx 3.6.1.

No GPL source was used. Test fixtures are independently generated (random bipartite
graphs with fixed seeds; Davis Southern Women graph reconstructed from nx built-in).

License: MIT OR Apache-2.0.
Upstream credit: [NetworkX](https://networkx.org/) (BSD-3-Clause).
