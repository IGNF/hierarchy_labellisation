use ndarray::{s, Array1, Array2, Array3, Zip};
use petgraph::{graph::NodeIndex, prelude::UnGraph};

use crate::plef::{Plef, PlefPiece};

pub type SuperpixelGraph = UnGraph<SuperpixelNode, SuperpixelEdge>;

#[derive(Debug, Clone)]
pub struct SuperpixelNode {
    pub area: u32,      // number of pixels in the superpixel
    pub perimeter: u32, // perimiter of the superpixel
    pub values: Array1<u64>,       // sum of values inside the superpixel
    pub values_sq: Array1<u64>,    // sum of squared values inside the superpixel
    pub optimal_energy: Plef<f64>, // optimal energy of the superpixel
}

impl SuperpixelNode {
    pub fn new(
        area: u32,
        perimeter: u32,
        values: Array1<u64>,
        values_sq: Array1<u64>,
        optimal_energy: Plef<f64>,
    ) -> Self {
        Self {
            area,
            perimeter,
            values,
            values_sq,
            optimal_energy,
        }
    }

    fn init(channels: usize) -> Self {
        Self::new(
            0,
            0,
            Array1::zeros(channels),
            Array1::zeros(channels),
            Plef::init(),
        )
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct SuperpixelEdge {
    pub weight: f64,
    pub length: u32,
    pub active: bool, // Maybe move into an array in the hierarchy algorithm
}

impl SuperpixelEdge {
    pub fn new(weight: f64, length: u32) -> Self {
        Self {
            weight,
            length,
            active: true,
        }
    }

    fn init() -> Self {
        Self::new(0., 0)
    }
}

pub fn data_fidelity(values: &Array1<u64>, values_sq: &Array1<u64>, area: u32) -> f64 {
    Zip::from(values_sq)
        .and(values)
        .fold(0., |acc, &value_sq, &value| {
            acc + value_sq as f64 - (value as f64).powi(2) / area as f64
        })
}

pub fn apparition_scale(source: &SuperpixelNode, target: &SuperpixelNode, edge_length: u32) -> f64 {
    let mut e = source.optimal_energy.sum(&target.optimal_energy, None);

    let values = &source.values + &target.values;
    let values_sq = &source.values_sq + &target.values_sq;
    let a = source.area + target.area;

    let data_fidelity = data_fidelity(&values, &values_sq, a);

    let merge_perimeter = source.perimeter + target.perimeter - 2 * edge_length;

    e.infimum(PlefPiece {
        start_x: 0.0,
        start_y: data_fidelity,
        slope: merge_perimeter as f64,
    })
}

pub fn graph_from_labels(img: &Array3<u8>, labels: &Array2<usize>) -> SuperpixelGraph {
    let (height, width, channels) = img.dim();
    let num_vertex = *labels.iter().max().unwrap() + 1;

    let mut graph = SuperpixelGraph::new_undirected();
    for _ in 0..num_vertex {
        graph.add_node(SuperpixelNode::init(channels));
    }

    for ((y, x), label) in labels.indexed_iter() {
        let i = NodeIndex::from(*label as u32);

        // Update superpixel area and values
        let node_i = &mut graph[i];
        node_i.area += 1;
        let pixel = img.slice(s![y, x, ..]).mapv(u64::from);
        node_i.values += &pixel;
        node_i.values_sq += &pixel.mapv(|x| x * x);

        // Loop over the neighbors (right and bottom)
        for (dy, dx) in [(0, 1), (1, 0)].iter() {
            let y2 = y + dy;
            let x2 = x + dx;

            if let Some(n_label) = labels.get((y2, x2)) {
                let j = NodeIndex::from(*n_label as u32);
                if n_label != label {
                    // We are on the border of the superpixel
                    // Update superpixel perimeters
                    let (node_i, node_j) = graph.index_twice_mut(i, j);
                    node_i.perimeter += 1;
                    node_j.perimeter += 1;

                    if !graph.contains_edge(i, j) {
                        graph.add_edge(i, j, SuperpixelEdge::init());
                    }

                    // Update superpixel edge length
                    let edge = graph.find_edge(i, j).unwrap();
                    let edge = &mut graph[edge];
                    edge.length += 1;
                }
            }
        }
    }

    // Take into account superpixels that are on the edge of the image
    for x in 0..width {
        graph[NodeIndex::from(labels[[0, x]] as u32)].perimeter += 1;
        graph[NodeIndex::from(labels[[height - 1, x]] as u32)].perimeter += 1;
    }
    for y in 0..height {
        graph[NodeIndex::from(labels[[y, 0]] as u32)].perimeter += 1;
        graph[NodeIndex::from(labels[[y, width - 1]] as u32)].perimeter += 1;
    }

    // Initialize optimal energy
    for node in graph.node_weights_mut() {
        let data_fidelity = data_fidelity(&node.values, &node.values_sq, node.area);

        let plef = Plef::from(PlefPiece::new(0., data_fidelity, node.perimeter as f64));
        node.optimal_energy = plef;
    }

    for edge_i in graph.edge_indices() {
        let (s_i, t_i) = graph.edge_endpoints(edge_i).unwrap();

        let s_node = &graph[s_i];
        let t_node = &graph[t_i];
        let edge = &graph[edge_i];

        graph[edge_i].weight = apparition_scale(s_node, t_node, edge.length);
    }

    graph
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_graph_from_labels() {
        // 0 0 1
        // 0 0 1
        // 2 2 2
        let labels = Array2::from_shape_vec((3, 3), vec![0, 0, 1, 0, 0, 1, 2, 2, 2]).unwrap();
        // Pixel values are from 0 to 27 (3 channels)
        let img = Array3::from_shape_vec((3, 3, 3), (0..27).into_iter().collect()).unwrap();

        let graph = graph_from_labels(&img, &labels);

        assert_eq!(graph.node_count(), 3);

        let node_0 = graph.node_weight(NodeIndex::from(0)).unwrap();
        assert_eq!(node_0.area, 4);
        assert_eq!(node_0.perimeter, 8);
        assert_eq!(node_0.values, array![24, 28, 32]);

        let node_1 = graph.node_weight(NodeIndex::from(1)).unwrap();
        assert_eq!(node_1.area, 2);
        assert_eq!(node_1.perimeter, 6);
        assert_eq!(node_1.values, array![21, 23, 25]);

        let node_2 = graph.node_weight(NodeIndex::from(2)).unwrap();
        assert_eq!(node_2.area, 3);
        assert_eq!(node_2.perimeter, 8);
        assert_eq!(node_2.values, array![63, 66, 69]);

        assert_eq!(graph.edge_count(), 3);

        let edge_0_1_weight = &graph[graph
            .find_edge(NodeIndex::from(0), NodeIndex::from(1))
            .unwrap()];
        assert_eq!(edge_0_1_weight.length, 2);

        let edge_0_2_weight = &graph[graph
            .find_edge(NodeIndex::from(0), NodeIndex::from(2))
            .unwrap()];
        assert_eq!(edge_0_2_weight.length, 2);

        let edge_1_2_weight = &graph[graph
            .find_edge(NodeIndex::from(1), NodeIndex::from(2))
            .unwrap()];
        assert_eq!(edge_1_2_weight.length, 1);
    }
}
