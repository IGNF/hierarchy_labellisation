use ndarray::{s, Array1, Array2, Array3};
use petgraph::{adj::NodeIndex, prelude::UnGraph};

pub struct SuperpixelNode {
    area: u32,           // number of pixels in the superpixel
    perimeter: u32,      // perimiter of the superpixel
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

    fn init(channels: usize) -> Self {
        Self::new(0, 0, Array1::zeros(channels))
    }
}

pub struct SuperpixelEdge {
    weight: f32,
    length: u32,
}

impl SuperpixelEdge {
    fn new(weight: f32, length: u32) -> Self {
        Self { weight, length }
    }

    fn init() -> Self {
        Self::new(0., 0)
    }
}

pub fn graph_from_labels(
    img: Array3<u8>,
    labels: Array2<usize>,
) -> UnGraph<SuperpixelNode, SuperpixelEdge> {
    let (_height, _width, channels) = img.dim();
    let num_vertex = *labels.iter().max().unwrap() + 1;

    let mut graph = UnGraph::<SuperpixelNode, SuperpixelEdge>::new_undirected();
    for _ in 0..num_vertex {
        graph.add_node(SuperpixelNode::init(channels));
    }

    for ((y, x), label) in labels.indexed_iter() {
        let i = NodeIndex::from(*label as u32);

        // Update superpixel area and values
        let node_i: &mut SuperpixelNode = &mut graph[i];
        node_i.area += 1;
        let pixel = img.slice(s![y, x, ..]).mapv(u32::from);
        node_i.values = pixel + &node_i.values;

        // Loop over the neighbors (right and bottom)
        for (dy, dx) in [(0, 1), (1, 0)].iter() {
            let y2 = y + dy;
            let x2 = x + dx;

            if let Some(n_label) = labels.get((y2, x2)) {
                let j = NodeIndex::from(*n_label as u32);
                if n_label != label {
                    // We are on the border of the superpixel
                    // Update superpixel perimeters
                    let (node_i, node_j): (&mut SuperpixelNode, &mut SuperpixelNode) =
                        graph.index_twice_mut(i, j);
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

    return graph;
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

        let graph = graph_from_labels(img, labels);

        assert_eq!(graph.node_count(), 3);

        let node_0 = graph.node_weight(NodeIndex::from(0)).unwrap();
        assert_eq!(node_0.area, 4);
        assert_eq!(node_0.perimeter, 4);
        assert_eq!(node_0.values, array![24, 28, 32]);

        let node_1 = graph.node_weight(NodeIndex::from(1)).unwrap();
        assert_eq!(node_1.area, 2);
        assert_eq!(node_1.perimeter, 3);
        assert_eq!(node_1.values, array![21, 23, 25]);

        let node_2 = graph.node_weight(NodeIndex::from(2)).unwrap();
        assert_eq!(node_2.area, 3);
        assert_eq!(node_2.perimeter, 3);
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
