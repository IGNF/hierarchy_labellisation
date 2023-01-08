mod graph;
mod logger;
mod seed;
mod slic;
mod slic_helpers;
mod utils;
mod plef;
mod hierarchy;

use graph::graph_from_labels;

use crate::{slic::slic, hierarchy::binary_partition_tree};
use ndarray::Array3;
use std::io::Cursor;
use utils::array_to_image;
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

pub fn hierarchical_segmentation(img: Array3<u8>, n_clusters: usize) -> Array3<u8> {
    console_log!("Starting slic");
    let labels = slic(n_clusters as u32, 1, Some(3), &img).expect_throw("SLIC failed");

    console_log!("Slic done");
    console_log!("Creating graph from segmentation...");

    let mut graph = graph_from_labels(&img, &labels);

    console_log!(
        "Nodes: {},  Edges: {}",
        graph.node_count(),
        graph.edge_count()
    );

    binary_partition_tree(&mut graph);

    let segmented_img = draw_segments(&img, labels);

    segmented_img
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

#[wasm_bindgen]
pub fn hierarchical_segmentation_from_js(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    n_clusters: usize,
) -> Box<[u8]> {
    let mut array = Array3::from_shape_vec((channels, height, width), data.to_vec())
        .expect_throw("Data doesn't have the right shape");

    array.swap_axes(0, 1);
    array.swap_axes(1, 2);

    let img = hierarchical_segmentation(array, n_clusters);

    let buffer = array_to_png(img.view());

    buffer.into_boxed_slice()
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
