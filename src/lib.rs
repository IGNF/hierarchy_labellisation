mod graph;
mod hierarchy;
mod logger;
mod plef;
mod seed;
mod slic;
mod slic_helpers;
mod utils;

use graph::graph_from_labels;
use hierarchy::PartitionTree;
use slic::slic;

use hierarchy::binary_partition_tree;
use ndarray::{Array2, Array3};
use std::{collections::HashMap, panic};
use utils::array_to_rgba_bitmap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

pub fn hierarchical_segmentation(
    img: Array3<u8>,
    n_clusters: usize,
) -> (Array2<usize>, PartitionTree) {
    console_log!("Running SLIC...");
    let labels = slic(n_clusters as u32, 1, Some(1), &img).expect_throw("SLIC failed");

    console_log!("Creating graph from segmentation...");

    let graph = graph_from_labels(&img, &labels);

    console_log!(
        "Nodes: {},  Edges: {}",
        graph.node_count(),
        graph.edge_count()
    );

    let partition_tree = binary_partition_tree(graph);

    (labels, partition_tree)
}

#[wasm_bindgen(getter_with_clone)]
#[derive(Clone, Debug)]
pub struct Hierarchy {
    pub labels: Vec<usize>,
    pub parents: Vec<usize>,
    pub levels: Vec<f64>,
    pub max_level: f64,
}

#[wasm_bindgen]
pub fn build_hierarchy_wasm(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    n_clusters: usize,
) -> Hierarchy {
    let mut array = Array3::from_shape_vec((channels, height, width), data.to_vec())
        .expect_throw("Data doesn't have the right shape");

    array.swap_axes(0, 1);
    array.swap_axes(1, 2);

    let (labels, tree) = hierarchical_segmentation(array, n_clusters);

    let labels = labels.as_standard_layout();
    let labels = labels.as_slice().unwrap();
    let labels = labels.to_vec();

    let max_level = tree.levels.iter().fold(0.0f64, |acc, l| acc.max(*l));

    Hierarchy {
        labels,
        parents: tree.parents,
        levels: tree.levels,
        max_level,
    }
}

#[wasm_bindgen]
pub fn cut_hierarchy_wasm(hierarchy: &Hierarchy, level: f64) -> Vec<usize> {
    let levels = hierarchy.levels.iter().cloned().enumerate();

    let mut label_rewrites = HashMap::<usize, Vec<usize>>::new();

    for (i, l) in levels {
        if l >= level {
            break;
        }

        let parent = hierarchy.parents[i];

        let children = label_rewrites.remove(&i);
        let parent_family = label_rewrites.entry(parent).or_insert_with(Vec::new);

        parent_family.push(i);
        if let Some(children) = children {
            parent_family.extend(children);
        }
    }

    let mut label_mappings = (0..hierarchy.parents.len()).collect::<Vec<_>>();
    for (parent, children) in label_rewrites {
        for child in children {
            label_mappings[child] = parent;
        }
    }

    let mut distinct = label_mappings.clone();
    distinct.sort();
    distinct.dedup();

    console_log!("Distinct labels: {}", distinct.len());

    let labels = hierarchy
        .labels
        .iter()
        .cloned()
        .map(|l| label_mappings[l])
        .collect::<Vec<_>>();

    labels
}

#[wasm_bindgen]
pub fn display_labels_wasm(
    mut img: Vec<u8>,
    width: usize,
    height: usize,
    labels: Vec<usize>,
) -> Vec<u8> {
    // Only take first 3 channels
    img.truncate(width * height * 3);

    let mut img = Array3::from_shape_vec((3, height, width), img).expect_throw("Img wrong shape");

    img.swap_axes(0, 1);
    img.swap_axes(1, 2);

    let labels = Array2::from_shape_vec((height, width), labels).expect_throw("Labels wrong shape");

    for (i, row) in labels.outer_iter().enumerate().take(height - 1) {
        for (j, label) in row.iter().enumerate().take(width - 1) {
            if label != &labels[[i + 1, j]] || label != &labels[[i, j + 1]] {
                img[[i, j, 0]] = 0;
                img[[i, j, 1]] = 0;
                img[[i, j, 2]] = 0;
            }
        }
    }

    array_to_rgba_bitmap(img.view())
}
