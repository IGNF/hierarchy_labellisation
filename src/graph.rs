use ndarray::{Array1, Array2, Array3};
use petgraph::{adj::NodeIndex, prelude::UnGraph};

pub struct SuperpixelNode {
    area: u32, // number of pixels in the superpixel
    perimeter: u32, // perimiter of the superpixel
    values: Array1<u32>, // sum of values inside the superpixel
}

impl SuperpixelNode {
    fn new(area: u32, perimeter: u32, values: Array1<u32>) -> Self {
        Self {
            area,
            perimeter,
            values,
        }
    }
}

impl Default for SuperpixelNode {
    fn default() -> Self {
        Self {
            area: 0,
            perimeter: 0,
            values: Array1::zeros(0),
        }
    }
}

pub fn graph_from_labels(img: Array3<u8>, labels: Array2<usize>) -> UnGraph<SuperpixelNode, f32> {
    let num_vertex = *labels.iter().max().unwrap() + 1;

    let mut graph = UnGraph::<SuperpixelNode, f32>::new_undirected();
    for _ in 0..num_vertex {
        graph.add_node(SuperpixelNode::default());
    }

    for ((y, x), label) in labels.indexed_iter() {
        for (dy, dx) in [(0, 1), (1, 0)].iter() {
            let y2 = y + dy;
            let x2 = x + dx;

            if let Some(n_label) = labels.get((y2, x2)) {
                let i = NodeIndex::from(*label as u32);
                let j = NodeIndex::from(*n_label as u32);
                if n_label != label && !graph.contains_edge(i, j) {
                    graph.add_edge(i, j, 1.0);
                }
            }
        }
    }

    return graph;
}

// TODO:
// Node area: number of pixels in the node
// Edge length: number of pixels between the two nodes
// Node perimeter: number of pixels in the node boundary
