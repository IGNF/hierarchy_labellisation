use crate::slic_helpers::{distance_pixel, div_ceil, get_pixel};

use ndarray::{s, Array1, Array3};
use simple_clustering::error::{ScError, SeedErrorKind};
use simple_clustering::Superpixel;

use num_traits::ToPrimitive;

/// Initialize the superpixel seed centers.
///
/// `width`, `height`, `s`, and `k` must not be `0`.
pub fn init_seeds<T: Copy>(
    s: u32,
    k: u32,
    image: &Array3<T>,
    seeds: &mut Vec<Superpixel<Array1<T>>>,
) -> Result<(), ScError> {
    seeds.clear();
    let width = image.shape()[1] as u32;
    let height = image.shape()[0] as u32;
    let s = s;
    let half_s = div_ceil(s, 2);
    let mut x_seeds = div_ceil(width, s);
    let mut y_seeds = div_ceil(height, s);

    // The seeds per row and column might be too high due to the div_ceil
    if u64::from(s) * u64::from(x_seeds) > u64::from(width) {
        x_seeds -= 1;
    }
    if u64::from(s) * u64::from(y_seeds) > u64::from(height) {
        y_seeds -= 1;
    }

    // If the seed count is larger than k, reduce until we're below (we may add
    // seeds in the "enforce connectivity" step later for SLIC)
    while u64::from(x_seeds) * u64::from(y_seeds) > u64::from(k) {
        x_seeds -= 1;
        y_seeds -= 1;
    }

    // Edge case for very small image sizes where no clusters would be produced
    if x_seeds == 0 {
        x_seeds += 1;
    }
    if y_seeds == 0 {
        y_seeds += 1;
    }

    // Error correction for spreading the seeds out more evenly along rows/cols
    let x_correction = (f64::from(width) - f64::from(x_seeds) * f64::from(s)) / f64::from(x_seeds);
    let y_correction = (f64::from(height) - f64::from(y_seeds) * f64::from(s)) / f64::from(y_seeds);

    let total_seeds = usize::try_from(u64::from(x_seeds) * u64::from(y_seeds))
        .or(Err(ScError::SeedError(SeedErrorKind::InvalidTotalSeeds)))?;

    if total_seeds > seeds.capacity() {
        seeds.try_reserve_exact(total_seeds - seeds.capacity())?;
    }

    for ydx in 0..y_seeds {
        let y_correct = (f64::from(ydx) * y_correction)
            .to_u32()
            .ok_or("Could not convert Y correction")?;
        for xdx in 0..x_seeds {
            let x_correct = (f64::from(xdx) * x_correction)
                .to_u32()
                .ok_or("Could not convert X correction")?;
            let x = xdx
                .saturating_mul(s)
                .saturating_add(half_s)
                .saturating_add(x_correct);
            let y = ydx
                .saturating_mul(s)
                .saturating_add(half_s)
                .saturating_add(y_correct);
            let i = usize::try_from(
                u64::from(y)
                    .saturating_mul(u64::from(width))
                    .saturating_add(u64::from(x)),
            )
            .or(Err(ScError::SeedError(SeedErrorKind::InvalidImageIndex)))?;
            if x < width && y < height && i < image.len() {
                seeds.push(Superpixel {
                    data: image.slice(s![y as i32, x as i32, ..]).to_owned(),
                    x,
                    y,
                });
            }
        }
    }

    Ok(())
}

/// Find the lowest gradient in a 3x3 neighborhood for a seed.
///
/// This step minimizes the chance that a noisy pixel is chosen as a seed.
pub fn perturb(seed: &mut Superpixel<Array1<u8>>, image: &Array3<u8>) -> Result<(), ScError> {
    let mut min = f64::INFINITY;
    let default = Array1::<u8>::zeros(image.shape()[2]);
    let sp_x = seed.x as i32;
    let sp_y = seed.y as i32;

    // Gradient equation is
    // fn gradient() -> f64 {
    //     (I[x + 1, y] - I[x - 1, y]).powi(2) +
    //     (I[x, y + 1] - I[x, y - 1]).powi(2)
    // }
    for ydx in -1..=1 {
        for xdx in -1..=1 {
            let superpixel = if let Some(pixel) = get_pixel(sp_x + xdx, sp_y + ydx, image) {
                (pixel, sp_x + xdx, sp_y + ydx)
            } else {
                continue;
            };
            let a_x = sp_x + xdx + 1;
            let b_x = sp_x + xdx - 1;
            let ab_y = sp_y + ydx;
            let cd_x = sp_x + xdx;
            let c_y = sp_y + ydx + 1;
            let d_y = sp_y + ydx - 1;

            let a = get_pixel(a_x, ab_y, image).unwrap_or_else(|| default.view());
            let b = get_pixel(b_x, ab_y, image).unwrap_or_else(|| default.view());
            let c = get_pixel(cd_x, c_y, image).unwrap_or_else(|| default.view());
            let d = get_pixel(cd_x, d_y, image).unwrap_or_else(|| default.view());

            let gradient = distance_pixel(a, b) + distance_pixel(c, d);

            if gradient < min {
                min = gradient;
                seed.data = superpixel.0.to_owned();
                seed.x = u32::try_from(superpixel.1)
                    .or(Err(ScError::SeedError(SeedErrorKind::PerturbConversion)))?;
                seed.y = u32::try_from(superpixel.2)
                    .or(Err(ScError::SeedError(SeedErrorKind::PerturbConversion)))?;
            }
        }
    }

    Ok(())
}
