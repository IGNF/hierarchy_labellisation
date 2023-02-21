use std::io::Cursor;

use image::{ImageBuffer, ImageOutputFormat, Rgb};
use ndarray::{s, ArrayView3};
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
