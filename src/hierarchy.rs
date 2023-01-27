use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use petgraph::prelude::{EdgeIndex, NodeIndex};

use crate::{
    console_log,
    graph::{apparition_scale, data_fidelity, SuperpixelEdge, SuperpixelGraph, SuperpixelNode},
    plef::PlefPiece,
};

#[derive(Debug, PartialEq)]
struct EdgeWrapper {
    index: EdgeIndex,
    weight: f64,
}

impl Eq for EdgeWrapper {}

impl PartialOrd for EdgeWrapper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeWrapper {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want the smallest weight to be the first element
        other.weight.partial_cmp(&self.weight).unwrap()
    }
}

pub struct PartitionTree {
    pub parents: Vec<usize>,
    pub levels: Vec<f64>,
}

pub(crate) fn binary_partition_tree(mut graph: SuperpixelGraph) -> PartitionTree {
    let num_points = graph.node_count();
    let mut parents = (0..num_points).collect::<Vec<_>>();
    let mut levels = vec![0.0; num_points];

    let mut heap: BinaryHeap<EdgeWrapper> = BinaryHeap::new();

    // Iterate over all edges
    for edge_id in graph.edge_indices() {
        let edge = graph.edge_weight(edge_id).unwrap();
        let weight = edge.weight;

        let wrapper = EdgeWrapper {
            index: edge_id,
            weight,
        };

        heap.push(wrapper);
    }

    let mut merge_operations = 0;

    // Used to store the neighbors of a fused node and its edges (allows to avoid re-allocating)
    let mut neighors = HashMap::<NodeIndex, Vec<EdgeIndex>>::new();

    while !heap.is_empty() {
        let top = heap.pop().unwrap();

        let fusion_edge_index = top.index;
        let fusion_edge = graph.edge_weight_mut(fusion_edge_index).unwrap();

        if !fusion_edge.active {
            continue;
        }

        assert!(fusion_edge.weight == top.weight, "Heap consistency assert");

        fusion_edge.active = false;

        let (a, b) = graph.edge_endpoints(fusion_edge_index).unwrap();

        neighors.clear();
        // Find all neighbors of a and b
        for (node, other) in [(a, b), (b, a)] {
            for neighbor in graph.neighbors(node) {
                let edge_id = graph.find_edge(node, neighbor).unwrap();
                let edge = graph.edge_weight(edge_id).unwrap();

                if neighbor == other || !edge.active {
                    assert!(!(neighbor == other && edge.active), "Active edge assert");
                    continue;
                }

                if let Some(edges) = neighors.get_mut(&neighbor) {
                    edges.push(edge_id);
                } else {
                    neighors.insert(neighbor, vec![edge_id]);
                }
            }
        }

        // Fuse the two nodes
        let node_a = graph.node_weight(a).unwrap();
        let node_b = graph.node_weight(b).unwrap();

        let fusion_edge = graph.edge_weight(fusion_edge_index).unwrap();

        let new_node = {
            let area = node_a.area + node_b.area;
            let perimeter = node_a.perimeter + node_b.perimeter - 2 * fusion_edge.length;
            let values = &node_a.values + &node_b.values;
            let values_sq = &node_a.values_sq + &node_b.values_sq;

            let data_fidelity = data_fidelity(&values, &values_sq, area);
            let mut plef = node_a.optimal_energy.sum(&node_b.optimal_energy, None);
            plef.infimum(PlefPiece::new(0., data_fidelity, perimeter as f64));

            SuperpixelNode::new(
                node_a.area + node_b.area,
                node_a.perimeter + node_b.perimeter - 2 * fusion_edge.length,
                &node_a.values + &node_b.values,
                &node_a.values_sq + &node_b.values_sq,
                plef,
            )
        };

        let fusion_weight = fusion_edge.weight;
        let new_node_id = graph.add_node(new_node);

        assert!(parents.len() == new_node_id.index());
        assert!(levels.len() == new_node_id.index());

        parents.push(new_node_id.index());
        levels.push(fusion_weight);

        parents[a.index()] = new_node_id.index();
        parents[b.index()] = new_node_id.index();

        for (neighbor_id, old_edges) in &neighors {
            let neighbor_id = *neighbor_id;
            let mut length = 0;

            for edge_id in old_edges {
                let edge = graph.edge_weight_mut(*edge_id).unwrap();
                length += edge.length;
                edge.active = false;
            }

            let neighbor_node = graph.node_weight(neighbor_id).unwrap();

            let weight = apparition_scale(&graph[new_node_id], neighbor_node, length);
            let new_edge = SuperpixelEdge::new(weight, length);
            let new_edge_id = graph.add_edge(new_node_id, neighbor_id, new_edge);
            heap.push(EdgeWrapper {
                index: new_edge_id,
                weight,
            });
        }

        merge_operations += 1;
    }

    console_log!("Merge operations: {:?}", merge_operations);

    PartitionTree { parents, levels }
}
