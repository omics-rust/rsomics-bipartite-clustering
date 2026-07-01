use std::io::Write;

use criterion::{criterion_group, criterion_main, Criterion};

use rsomics_bipartite_clustering::{io, latapy_clustering, robins_alexander_clustering, Mode};

fn load() -> rsomics_bipartite_clustering::Graph {
    let mut f = tempfile::Builder::new()
        .tempfile_in("/Volumes/KIOXIA/tmp")
        .unwrap();
    f.write_all(include_bytes!("bench_edges.txt")).unwrap();
    f.flush().unwrap();
    io::read_edgelist(Some(f.path())).unwrap()
}

fn bench(c: &mut Criterion) {
    // Parse once; measure compute only.
    let g = load();
    let nodes: Vec<u32> = (0..g.n() as u32).collect();

    c.bench_function("latapy_dot gnmk(300,300,3000)", |b| {
        b.iter(|| latapy_clustering(&g, &nodes, Mode::Dot))
    });

    c.bench_function("robins_alexander gnmk(300,300,3000)", |b| {
        b.iter(|| robins_alexander_clustering(&g))
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
