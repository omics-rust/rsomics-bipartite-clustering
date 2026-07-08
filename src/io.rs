use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use rsomics_common::{Result, RsomicsError};

/// Undirected graph with integer-mapped node IDs in first-seen order.
///
/// Neighbor lists are deduped in insertion order, matching `nx.Graph` semantics.
/// Self-loops are kept: a self-loop on `v` places `v` once in `adj[v]`, mirroring
/// `nx.Graph` where `G[v]` lists `v` once while `degree(v)` counts it twice.
pub struct Graph {
    pub labels: Vec<String>,
    pub index: HashMap<String, u32>,
    /// Neighbors in first-seen insertion order per node (deduped).
    pub adj: Vec<Vec<u32>>,
}

impl Graph {
    pub fn n(&self) -> usize {
        self.labels.len()
    }

    /// Edge count matching `nx.Graph.size()`: a self-loop is one edge but two
    /// degree endpoints, so it appears once in `adj` yet must count as two in
    /// the degree sum.
    pub fn m(&self) -> usize {
        let deg_sum: usize = self.adj.iter().map(|nbrs| nbrs.len()).sum();
        let self_loops = (0..self.n())
            .filter(|&i| self.adj[i].contains(&(i as u32)))
            .count();
        (deg_sum + self_loops) / 2
    }
}

/// Parse an undirected edge list, matching `nx.read_edgelist` / `nx.Graph`.
///
/// A `#` anywhere in a line begins a comment: text from the first `#` onward
/// is dropped before tokenising, so `1 2 # note` yields edge (1, 2). Lines
/// that are blank after comment stripping are skipped. Each data line needs at
/// least two whitespace-separated tokens; extras are ignored. Self-loops are
/// kept (matching `nx.Graph`). Duplicate edges collapse to a simple graph in
/// first-seen order.
pub fn read_edgelist(path: Option<&Path>) -> Result<Graph> {
    let reader: Box<dyn BufRead> = match path {
        None => Box::new(BufReader::new(std::io::stdin())),
        Some(p) if p == Path::new("-") => Box::new(BufReader::new(std::io::stdin())),
        Some(p) => Box::new(BufReader::new(File::open(p).map_err(|e| {
            RsomicsError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", p.display()),
            ))
        })?)),
    };
    let mut buf = String::new();
    let mut reader = reader;
    reader.read_to_string(&mut buf).map_err(RsomicsError::Io)?;
    read_edgelist_str(&buf)
}

/// Parse an undirected edge list from an in-memory string. Same semantics as
/// [`read_edgelist`]; used where the input is already buffered.
pub fn read_edgelist_str(input: &str) -> Result<Graph> {
    let mut labels: Vec<String> = Vec::new();
    let mut index: HashMap<String, u32> = HashMap::new();
    let mut raw_edges: Vec<(u32, u32)> = Vec::new();

    for (lineno, line) in input.lines().enumerate() {
        let lineno = lineno + 1;
        // nx.parse_edgelist strips a '#' comment anywhere in the line before tokenising.
        let t = line.split('#').next().unwrap_or("").trim();
        if t.is_empty() {
            continue;
        }
        let mut tokens = t.split_ascii_whitespace();
        let u_str = tokens.next().unwrap();
        let v_str = tokens.next().ok_or_else(|| {
            RsomicsError::InvalidInput(format!("line {lineno}: expected two node labels, got one"))
        })?;
        let u = intern(&mut labels, &mut index, u_str);
        let v = intern(&mut labels, &mut index, v_str);
        raw_edges.push((u, v));
    }

    let n = labels.len();
    // A self-loop (u == v) yields raw edge (u, u); the first insert records it
    // once in adj[u], the mirrored insert is a no-op — one entry, as in nx.Graph.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut seen: Vec<std::collections::HashSet<u32>> = vec![Default::default(); n];
    for (u, v) in raw_edges {
        if seen[u as usize].insert(v) {
            adj[u as usize].push(v);
        }
        if seen[v as usize].insert(u) {
            adj[v as usize].push(u);
        }
    }

    Ok(Graph { labels, index, adj })
}

fn intern(labels: &mut Vec<String>, index: &mut HashMap<String, u32>, s: &str) -> u32 {
    if let Some(&id) = index.get(s) {
        return id;
    }
    let id = labels.len() as u32;
    labels.push(s.to_owned());
    index.insert(s.to_owned(), id);
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized(g: &Graph) -> (Vec<String>, Vec<Vec<u32>>) {
        let mut adj = g.adj.clone();
        for nbrs in adj.iter_mut() {
            nbrs.sort_unstable();
        }
        (g.labels.clone(), adj)
    }

    #[test]
    fn inline_hash_comment_matches_comment_free_graph() {
        // A '#' mid-line begins a comment: "1 2#c" is edge (1,2) not node "2#c",
        // a trailing "2 3 # note" ignores the note, and a whole-line comment is
        // skipped — all matching nx.parse_edgelist.
        let with_comments =
            read_edgelist_str("0 1\n1 2#c\n2 3 # note\n# full-line comment\n").unwrap();
        let clean = read_edgelist_str("0 1\n1 2\n2 3\n").unwrap();
        assert_eq!(normalized(&with_comments), normalized(&clean));
        assert_eq!(with_comments.n(), 4);
        assert_eq!(with_comments.m(), 3);
    }
}
