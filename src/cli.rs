use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use serde::Serialize;

use rsomics_common::{run, CommonFlags, Result, ToolMeta};

use rsomics_bipartite_clustering::{
    average_clustering, io, latapy_clustering, robins_alexander_clustering, Mode,
};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Bipartite clustering coefficients (Latapy 2008, Robins-Alexander 2004),
/// value-exact networkx 3.6.1 equivalent.
///
/// Reads an undirected bipartite edge list (one `u v` per line) from FILE or
/// stdin (`-`). Comment lines (`#`) and blank lines are ignored. Self-loops are
/// dropped; duplicate edges collapse (nx.Graph semantics).
///
/// Select the operation with `--op`:
///
///   latapy   — per-node clustering; prints `label value` lines sorted by label.
///              Use `--mode dot|min|max` (default: dot). Use `--nodes n1,n2,...`
///              to restrict output to a subset of node labels.
///
///   average  — mean Latapy clustering over all nodes (or the `--nodes` subset).
///              Prints a single float.
///
///   robins-alexander — 4·C₄/L₃ for the whole graph. Prints a single float.
#[derive(Parser, Debug)]
#[command(name = "rsomics-bipartite-clustering", version, about, long_about = None)]
pub struct Cli {
    /// Operation to compute.
    #[arg(long = "op", value_enum, default_value_t = Op::Latapy)]
    pub op: Op,

    /// Pairwise bipartite clustering mode (latapy and average only).
    #[arg(long = "mode", value_enum, default_value_t = CliMode::Dot)]
    pub mode: CliMode,

    /// Comma-separated node labels to restrict output (latapy and average only).
    /// Omit to compute over all nodes.
    #[arg(long = "nodes", value_delimiter = ',', value_name = "LABELS")]
    pub nodes: Option<Vec<String>>,

    /// Edge list file (`-` or omitted reads stdin).
    #[arg(value_name = "EDGELIST")]
    pub edgelist: Option<PathBuf>,

    #[command(flatten)]
    pub common: CommonFlags,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Op {
    Latapy,
    Average,
    #[value(name = "robins-alexander")]
    RobinsAlexander,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Default)]
pub enum CliMode {
    #[default]
    Dot,
    Min,
    Max,
}

impl From<CliMode> for Mode {
    fn from(m: CliMode) -> Self {
        match m {
            CliMode::Dot => Mode::Dot,
            CliMode::Min => Mode::Min,
            CliMode::Max => Mode::Max,
        }
    }
}

#[derive(Serialize)]
#[serde(untagged)]
enum Out {
    Latapy {
        op: &'static str,
        mode: &'static str,
        clustering: Vec<NodeCC>,
    },
    Scalar {
        op: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        mode: Option<&'static str>,
        value: f64,
    },
}

#[derive(Serialize)]
struct NodeCC {
    node: String,
    clustering: f64,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || self.execute(&common))
    }

    fn execute(self, common: &CommonFlags) -> Result<Out> {
        let g = io::read_edgelist(self.edgelist.as_deref())?;
        let mode: Mode = self.mode.into();

        match self.op {
            Op::Latapy => {
                let targets = resolve_nodes(&g, self.nodes.as_deref())?;
                let mut ccs = latapy_clustering(&g, &targets, mode)?;
                // Sort by label string, matching nx's dict-iteration-in-sorted-label-order output.
                ccs.sort_by(|(a, _), (b, _)| g.labels[*a as usize].cmp(&g.labels[*b as usize]));

                if !common.json {
                    for (id, cc) in &ccs {
                        println!("{} {}", g.labels[*id as usize], cc);
                    }
                }

                Ok(Out::Latapy {
                    op: "latapy",
                    mode: mode_str(mode),
                    clustering: ccs
                        .into_iter()
                        .map(|(id, cc)| NodeCC {
                            node: g.labels[id as usize].clone(),
                            clustering: cc,
                        })
                        .collect(),
                })
            }

            Op::Average => {
                let targets = resolve_nodes(&g, self.nodes.as_deref())?;
                let value = average_clustering(&g, &targets, mode)?;

                if !common.json {
                    println!("{value}");
                }

                Ok(Out::Scalar {
                    op: "average",
                    mode: Some(mode_str(mode)),
                    value,
                })
            }

            Op::RobinsAlexander => {
                let value = robins_alexander_clustering(&g);

                if !common.json {
                    println!("{value}");
                }

                Ok(Out::Scalar {
                    op: "robins-alexander",
                    mode: None,
                    value,
                })
            }
        }
    }
}

fn mode_str(m: Mode) -> &'static str {
    match m {
        Mode::Dot => "dot",
        Mode::Min => "min",
        Mode::Max => "max",
    }
}

fn resolve_nodes(
    g: &rsomics_bipartite_clustering::io::Graph,
    labels: Option<&[String]>,
) -> rsomics_common::Result<Vec<u32>> {
    match labels {
        None => Ok((0..g.n() as u32).collect()),
        Some(ls) => {
            let mut ids = Vec::with_capacity(ls.len());
            for l in ls {
                let id = *g.index.get(l.as_str()).ok_or_else(|| {
                    rsomics_common::RsomicsError::InvalidInput(format!(
                        "node {l:?} is not present in the graph"
                    ))
                })?;
                ids.push(id);
            }
            Ok(ids)
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_debug_assert() {
        super::Cli::command().debug_assert();
    }
}
