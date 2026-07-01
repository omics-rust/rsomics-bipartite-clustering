/// Latapy bipartite clustering mode — selects the pairwise denominator.
///
/// Latapy et al. (2008), Social Networks 30(1):31–48.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// |N(v) ∩ N(u)| / |N(v) ∪ N(u)|  (Jaccard — the paper's primary mode)
    #[default]
    Dot,
    /// |N(v) ∩ N(u)| / min(|N(v)|, |N(u)|)
    Min,
    /// |N(v) ∩ N(u)| / max(|N(v)|, |N(u)|)
    Max,
}

use crate::io::Graph;

/// Per-node Latapy bipartite clustering coefficients.
///
/// Matches `networkx.algorithms.bipartite.clustering(G, nodes, mode)` exactly.
///
/// For each node v:
///   nbrs2 = { u ∈ N(N(v)) } \ {v}          (two-hop neighbours excluding v)
///   c(v)  = Σ_{u∈nbrs2} cc_func(N(v), N(u)) / |nbrs2|   (0 if nbrs2 is empty)
///
/// cc_func chosen by `mode` (see `Mode`). Isolated nodes (deg 0) get 0.0.
///
/// Returns one coefficient per node in the `nodes` slice (order-preserving).
/// If `nodes` is empty the output is also empty (caller interprets as all nodes).
pub fn latapy_clustering(g: &Graph, nodes: &[u32], mode: Mode) -> Vec<(u32, f64)> {
    // Pre-materialise neighbor sets as bitset-style sorted vecs for O(|nbr|) intersection.
    // For the graph sizes in scope (bipartite social/genomic) this is faster than HashSet.
    let n = g.n();
    let adj_sets: Vec<Vec<u32>> = (0..n)
        .map(|v| {
            let mut s = g.adj[v].clone();
            s.sort_unstable();
            s
        })
        .collect();

    let targets: &[u32] = nodes;
    let mut out = Vec::with_capacity(targets.len());

    for &v in targets {
        let v_nbrs = &adj_sets[v as usize];

        // nbrs2 = { u ∈ N(N(v)) } \ {v}
        // Collect, dedup, then compute per the nx formula.
        let mut nbrs2: Vec<u32> = v_nbrs
            .iter()
            .flat_map(|&w| g.adj[w as usize].iter().copied())
            .filter(|&u| u != v)
            .collect();
        nbrs2.sort_unstable();
        nbrs2.dedup();

        let mut cc = 0.0_f64;
        for &u in &nbrs2 {
            let u_nbrs = &adj_sets[u as usize];
            let inter = intersection_count(v_nbrs, u_nbrs);
            let pairwise = match mode {
                Mode::Dot => {
                    let union = v_nbrs.len() + u_nbrs.len() - inter;
                    inter as f64 / union as f64
                }
                Mode::Min => inter as f64 / v_nbrs.len().min(u_nbrs.len()) as f64,
                Mode::Max => inter as f64 / v_nbrs.len().max(u_nbrs.len()) as f64,
            };
            cc += pairwise;
        }

        // Mirror nx: `if cc > 0.0: cc /= len(nbrs2)` — isolated nodes stay 0.
        if cc > 0.0 {
            cc /= nbrs2.len() as f64;
        }

        out.push((v, cc));
    }

    out
}

/// Mean of Latapy clustering over the given nodes.
///
/// Matches `networkx.algorithms.bipartite.average_clustering(G, nodes, mode)`.
/// Returns 0.0 if `nodes` is empty.
pub fn average_clustering(g: &Graph, nodes: &[u32], mode: Mode) -> f64 {
    if nodes.is_empty() {
        return 0.0;
    }
    let ccs = latapy_clustering(g, nodes, mode);
    ccs.iter().map(|(_, c)| c).sum::<f64>() / nodes.len() as f64
}

/// Robins-Alexander bipartite clustering coefficient.
///
/// CC₄ = 4·C₄ / L₃
///
/// C₄ = number of four-cycles (undirected squares), L₃ = number of three-paths.
/// Matches `networkx.algorithms.bipartite.robins_alexander_clustering(G)` exactly,
/// including the early-return conditions (order < 4 or size < 3 or L₃ = 0).
pub fn robins_alexander_clustering(g: &Graph) -> f64 {
    if g.n() < 4 || g.m() < 3 {
        return 0.0;
    }
    let l3 = three_paths(g);
    if l3 == 0 {
        return 0.0;
    }
    let c4 = four_cycles(g);
    (4.0 * c4 as f64) / l3 as f64
}

// ---------------------------------------------------------------------------
// Internals — matching NX _four_cycles and _threepaths exactly
// ---------------------------------------------------------------------------

/// Count four-cycles (C₄) matching `_four_cycles` in networkx.
///
/// Iterates nodes in insertion order (seen set to avoid double-counting):
///   for each v (unseen), for each two-hop neighbour x (not in seen):
///     p2 = |N(v) ∩ N(x)|
///     cycles += p2*(p2-1)
///   return cycles / 4  (as integer via exact integer arithmetic)
///
/// NX returns a float (cycles/4); since cycles is always divisible by 4
/// in an undirected graph, we return an integer for cleanliness.
fn four_cycles(g: &Graph) -> u64 {
    let n = g.n();
    let adj_sets: Vec<Vec<u32>> = (0..n)
        .map(|v| {
            let mut s = g.adj[v].clone();
            s.sort_unstable();
            s
        })
        .collect();

    let mut cycles: u64 = 0;
    let mut seen = vec![false; n];

    for v in 0..n as u32 {
        seen[v as usize] = true;
        let v_nbrs = &adj_sets[v as usize];
        if v_nbrs.len() < 2 {
            continue;
        }

        // two-hop neighbours of v not yet seen
        let mut two_hop: Vec<u32> = v_nbrs
            .iter()
            .flat_map(|&u| g.adj[u as usize].iter().copied())
            .filter(|&x| !seen[x as usize])
            .collect();
        two_hop.sort_unstable();
        two_hop.dedup();

        for x in two_hop {
            let p2 = intersection_count(v_nbrs, &adj_sets[x as usize]) as u64;
            cycles += p2 * (p2 - 1);
        }
    }

    // NX divides by 4 (each four-cycle counted 4 times, once per node pair).
    // cycles is always divisible by 4 in a well-formed undirected graph.
    debug_assert_eq!(cycles % 4, 0, "four-cycle count not divisible by 4");
    cycles / 4
}

/// Count three-paths (L₃) matching `_threepaths` in networkx.
///
/// NX formula:
///   for v: for u in N(v): for w in N(u)\{v}: paths += |N(w) \ {v,u}|
///   return paths / 2
fn three_paths(g: &Graph) -> u64 {
    let n = g.n();
    let mut paths: u64 = 0;

    for v in 0..n as u32 {
        for &u in &g.adj[v as usize] {
            for &w in &g.adj[u as usize] {
                if w == v {
                    continue;
                }
                // |N(w) \ {v, u}|
                let count = g.adj[w as usize]
                    .iter()
                    .filter(|&&x| x != v && x != u)
                    .count() as u64;
                paths += count;
            }
        }
    }

    // Each three-path counted twice (once from each endpoint).
    debug_assert_eq!(paths % 2, 0, "three-path count not divisible by 2");
    paths / 2
}

/// Count elements in the sorted-set intersection of two sorted slices.
fn intersection_count(a: &[u32], b: &[u32]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut count = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_node_is_zero() {
        // Single isolated node: no edges, clustering must be 0.
        use crate::io::Graph;
        use std::collections::HashMap;
        let g = Graph {
            labels: vec!["a".into()],
            index: HashMap::from([("a".into(), 0)]),
            adj: vec![vec![]],
        };
        let r = latapy_clustering(&g, &[0], Mode::Dot);
        assert_eq!(r[0].1, 0.0);
    }
}
