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

use crate::{hierarchy::binary_partition_tree, slic::slic};
use ndarray::{Array2, Array3};
use std::{collections::HashMap, io::Cursor};
use utils::{array_to_image, array_to_rgba_bitmap};
use wasm_bindgen::prelude::*;

use crate::utils::{array_to_png, draw_segments};

#[wasm_bindgen]
pub fn convert_to_png(data: &[u8], width: usize, height: usize, channels: usize) -> Box<[u8]> {
    let array = Array3::from_shape_vec((channels, height, width), data.to_vec())
        .expect_throw("Data doesn't have the right shape");

    let output = array_to_image(array.view());

    // Convert to png and return
    let mut buffer = Vec::new();
    output
        .write_to(&mut Cursor::new(&mut buffer), image::ImageOutputFormat::Png)
        .unwrap();

    buffer.into_boxed_slice()
}

pub fn hierarchical_segmentation(
    img: Array3<u8>,
    n_clusters: usize,
) -> (Array2<usize>, PartitionTree) {
    console_log!("Starting slic");
    let labels = slic(n_clusters as u32, 1, Some(3), &img).expect_throw("SLIC failed");

    console_log!("Slic done");
    console_log!("Creating graph from segmentation...");

    let graph = graph_from_labels(&img, &labels);

    console_log!(
        "Nodes: {},  Edges: {}",
        graph.node_count(),
        graph.edge_count()
    );

    let partition_tree = binary_partition_tree(graph);

    // let segmented_img = draw_segments(&img, labels);

    (labels, partition_tree)
}

#[wasm_bindgen]
pub fn slic_from_js(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    n_clusters: usize,
    _compactness: f32,
) -> Box<[u8]> {
    let mut array = Array3::from_shape_vec((channels, height, width), data.to_vec())
        .expect_throw("Data doesn't have the right shape");

    array.swap_axes(0, 1);
    array.swap_axes(1, 2);

    console_log!("Dimensions: {:?}", array.dim());

    let labels = slic::slic(n_clusters as u32, 1, Some(10), &array).expect_throw("SLIC failed");

    let segmented_img = draw_segments(&array, labels);

    let buffer = array_to_png(segmented_img.view());
    buffer.into_boxed_slice()
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
    let Hierarchy {
        labels,
        parents,
        levels,
        ..
    } = hierarchy.clone();

    let mut levels = levels.into_iter().enumerate().collect::<Vec<_>>();

    // Sort levels by ascending order
    // levels.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut label_rewrites = HashMap::<usize, Vec<usize>>::new();

    for (i, l) in levels {
        if l >= level {
            break;
        }

        let parent = parents[i];

        let children = label_rewrites.remove(&i);
        let parent_family = label_rewrites.entry(parent).or_insert_with(Vec::new);

        parent_family.push(i);
        if let Some(children) = children {
            parent_family.extend(children);
        }
    }

    let mut label_mappings = (0..parents.len()).collect::<Vec<_>>();
    for (parent, children) in label_rewrites {
        for child in children {
            label_mappings[child] = parent;
        }
    }

    let mut distinct = label_mappings.clone();
    distinct.sort();
    distinct.dedup();

    console_log!("Distinct labels: {}", distinct.len());

    let labels = labels
        .into_iter()
        .map(|l| label_mappings[l])
        .collect::<Vec<_>>();

    labels
}

#[wasm_bindgen]
pub fn display_labels_wasm(
    img: Vec<u8>,
    width: usize,
    height: usize,
    channels: usize,
    labels: Vec<usize>,
) -> Vec<u8> {
    let mut img =
        Array3::from_shape_vec((channels, height, width), img).expect_throw("Img wrong shape");

    img.swap_axes(0, 1);
    img.swap_axes(1, 2);

    let labels = Array2::from_shape_vec((height, width), labels).expect_throw("Labels wrong shape");

    let res_img = draw_segments(&img, labels);

    array_to_rgba_bitmap(res_img.view())
}

#[cfg(test)]
mod tests {
    use std::io::Result;

    use image::{GenericImageView, ImageResult};

    use super::*;

    fn load_image_geotiff(path: &str) -> Result<Array3<u8>> {
        let image = geotiff::TIFF::open(path)?;

        let height = image.image_data.len();
        let width = image.image_data[0].len();
        let channels = image.image_data[0][0].len();

        // Create ndarray
        let mut data = Array3::zeros((height, width, channels));

        for (i, row) in image.image_data.iter().enumerate() {
            for (j, pixel) in row.iter().enumerate() {
                for (k, channel) in pixel.iter().enumerate() {
                    data[[i, j, k]] = *channel as u8;
                }
            }
        }

        Ok(data)
    }

    fn load_image_tiff(path: &str) -> ImageResult<Array3<u8>> {
        let image = image::open(path)?;

        let (width, height) = image.dimensions();

        println!("Image dimensions: {}x{}", width, height);

        // Create ndarray
        let mut data = Array3::zeros((height as usize, width as usize, 3));

        for (x, y, pixel) in image.pixels() {
            let (i, j) = (y as usize, x as usize);

            for k in 0..3 {
                data[[i, j, k]] = pixel[k];
            }
        }

        Ok(data)
    }

    #[test]
    fn load_image() {
        let input_path = "./img/test_img.tif";
        // let input_path = "../transfer01/images_2018/57_2018_u5_974848_6902784_irc.tif";
        let output_path = "./img/test_img.png";

        let data = load_image_geotiff(input_path).unwrap();

        let (height, width, channels) = data.dim();

        // Print image size
        println!(
            "Image size: {} {} with {:?} channels",
            height, width, channels
        );

        let output = array_to_image(data.view());

        output.save(output_path).unwrap();

        println!("Image saved");
    }

    #[test]
    fn slic_test() {
        let input_path = "./img/test_img.tif";
        let output_path = "./img/test_img_slic.png";

        let data = load_image_geotiff(input_path).unwrap();
        // let data = load_image_tiff("./img/test_img_cropped.png").unwrap();

        println!("Image loaded");

        let labels = {
            let img = data;
            let _compactness = 10.0;
            console_log!("Dimensions: {:?}", img.dim());

            let labels = slic::slic(255 as u32, 1, Some(10), &img).expect_throw("SLIC failed");

            let segmented_img = draw_segments(&img, labels);

            segmented_img
        };

        let (height, width, _channels) = labels.dim();

        let mut output = image::ImageBuffer::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel = image::Rgb([
                    labels[[y, x, 0]] as u8,
                    labels[[y, x, 1]] as u8,
                    labels[[y, x, 2]] as u8,
                ]);
                output.put_pixel(x as u32, y as u32, pixel);
            }
        }

        output.save(output_path).unwrap();

        println!("Image saved");
    }
}
