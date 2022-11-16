pub mod seed;
pub mod slic;
pub mod slic_helpers;

use slic::slic_from_bytes;

use simple_clustering::image::segment_contours;

use ndarray::{s, Array3};
use std::io::Cursor;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[wasm_bindgen]
pub fn convert_to_png(data: &[u8], width: usize, height: usize, channels: usize) -> Box<[u8]> {
    let array = Array3::from_shape_vec((channels, height, width), data.to_vec())
        .expect_throw("Data doesn't have the right shape");

    let mut output = image::ImageBuffer::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let pixel = image::Rgb([
                array[[0, y, x]] as u8,
                array[[1, y, x]] as u8,
                array[[2, y, x]] as u8,
            ]);
            output.put_pixel(x as u32, y as u32, pixel);
        }
    }

    // Convert to png and return
    let mut buffer = Vec::new();
    output
        .write_to(&mut Cursor::new(&mut buffer), image::ImageOutputFormat::Png)
        .unwrap();

    buffer.into_boxed_slice()
}

pub fn slic(img: Array3<u8>, n_clusters: usize, compactness: f32) -> Array3<u8> {
    let (height, width, _channels) = img.dim();

    let img_array_sliced = img.slice(s![.., .., ..3usize]).to_owned();
    let mut img_array_std_layout = img_array_sliced.as_standard_layout();
    let img_slice = img_array_std_layout
        .as_slice_mut()
        .expect_throw("Fail to convert to slice");

    let clusters = slic_from_bytes(
        n_clusters as u32,
        1,
        width as u32,
        height as u32,
        Some(10),
        img_slice,
    )
    .expect_throw("SLIC failed");

    segment_contours(img_slice, width as u32, height as u32, &clusters, [0; 3])
        .expect_throw("Failed to compute contours");

    img_array_std_layout.to_owned()
}

#[wasm_bindgen]
pub fn slic_from_js(
    data: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    n_clusters: usize,
    compactness: f32,
) -> Box<[u8]> {
    let mut array = Array3::from_shape_vec((channels, height, width), data.to_vec())
        .expect_throw("Data doesn't have the right shape");

    array.swap_axes(0, 1);
    array.swap_axes(1, 2);

    let labels = slic(array, n_clusters, compactness);

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

    // Convert to png and return
    let mut buffer = Vec::new();
    output
        .write_to(&mut Cursor::new(&mut buffer), image::ImageOutputFormat::Png)
        .expect_throw("Failed to write to png");

    buffer.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use std::io::Result;

    use image::{GenericImageView, ImageResult};

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

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

        let mut output = image::ImageBuffer::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let pixel = image::Rgb([
                    data[[y, x, 0]] as u8,
                    data[[y, x, 1]] as u8,
                    data[[y, x, 2]] as u8,
                ]);
                output.put_pixel(x as u32, y as u32, pixel);
            }
        }

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

        let labels = slic(data, 255, 10.0);

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
