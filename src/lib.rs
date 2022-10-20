use ndarray::Array3;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(left: usize, right: usize) -> usize {
    left + right
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

        // Create ndarray
        let mut data = Array3::zeros((height as usize, width as usize, 3));

        for (i, j, pixel) in image.pixels() {
            let (i, j) = (i as usize, j as usize);

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
}
