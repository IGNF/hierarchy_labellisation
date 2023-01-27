use std::io::Cursor;

use image::{ImageBuffer, ImageOutputFormat, Rgb};
use ndarray::{concatenate, s, Array2, Array3, ArrayView3, Axis};
use simple_clustering::image::segment_contours;
use wasm_bindgen::UnwrapThrowExt;

pub(crate) fn array_to_image(input: ArrayView3<u8>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (height, width, _channels) = input.dim();
    let mut output = ImageBuffer::new(width as u32, height as u32);

    for (y, row) in input.outer_iter().enumerate() {
        for (x, pixel) in row.outer_iter().enumerate() {
            let pixel = Rgb([pixel[0], pixel[1], pixel[2]]);
            output.put_pixel(x as u32, y as u32, pixel);
        }
    }

    output
}

pub(crate) fn image_to_png(img: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<u8> {
    let mut buffer = Vec::new();
    img.write_to(&mut Cursor::new(&mut buffer), ImageOutputFormat::Png)
        .expect_throw("Failed to write to png");

    buffer
}

pub(crate) fn array_to_png(input: ArrayView3<u8>) -> Vec<u8> {
    let img = array_to_image(input.slice(s![.., .., 0..3]));
    image_to_png(img)
}

pub(crate) fn array_to_rgba_bitmap(input: ArrayView3<u8>) -> Vec<u8> {
    let (height, width, _channels) = input.dim();

    let mut output = vec![255; (height * width * 4) as usize];

    for (y, row) in input.outer_iter().enumerate() {
        for (x, pixel) in row.outer_iter().enumerate() {
            let i = (y * width + x) as usize;
            output[i * 4] = pixel[0];
            output[i * 4 + 1] = pixel[1];
            output[i * 4 + 2] = pixel[2];
        }
    }

    output
}

pub(crate) fn draw_segments(img: &Array3<u8>, labels: Array2<usize>) -> Array3<u8> {
    let (height, width, _channels) = img.dim();
    let labels = labels.into_raw_vec();

    // Only take the first 3 channels to visualize the segmentation
    let img_visu = img.slice(s![.., .., ..3usize]);
    let mut img_visu_std = img_visu.as_standard_layout();
    let img_visu_slice = img_visu_std
        .as_slice_mut()
        .expect_throw("Fail to convert to slice");
    segment_contours(img_visu_slice, width as u32, height as u32, &labels, [0; 3])
        .expect_throw("Failed to compute contours");
    img_visu_std.to_owned()
}
