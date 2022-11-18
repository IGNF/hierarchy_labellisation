use ndarray::{s, Array3, ArrayView1};
use num_traits::{Float, One, Unsigned, Zero};
use std::ops::{Add, Div, Rem};

/// Calculate the superpixel side length, `S`.
///
/// `S * S` is the approximate size of each superpixel in pixels. The formula is
/// `S = (N / K).sqrt()`, where `N` is the number of pixels and `K` is the
/// number of desired superpixels.
#[inline]
pub fn calculate_grid_interval(width: u32, height: u32, superpixels: u32) -> f64 {
    ((f64::from(width) * f64::from(height)) / f64::from(superpixels)).sqrt()
}

/// Calculate the distance between two pixels
#[inline]
pub fn distance_pixel(lhs: ArrayView1<u8>, rhs: ArrayView1<u8>) -> f64 {
    let mut sum = 0.0;
    for i in 0..lhs.len() {
        let diff = lhs[i] as f32 - rhs[i] as f32;
        sum += diff * diff;
    }

    sum as f64
}

/// Calculate the distance between two two-dimensional points.
#[inline]
pub fn distance_xy<T: Float>(lhs: (T, T), rhs: (T, T)) -> T {
    (rhs.0 - lhs.0).powi(2) + (rhs.1 - lhs.1).powi(2)
}

/// Calculate the `s` distance.
#[inline]
pub fn distance_s<T: Float>(m_div_s: T, d_lab: T, d_xy: T) -> T {
    d_lab + m_div_s * d_xy
}

/// Calculate the superpixel scaling factor.
///
/// `m_div_s` is `(m / s).powi(2)`.
#[inline]
pub fn m_div_s(m: f64, s: f64) -> f64 {
    (m / s).powi(2)
}

/// Calculates the quotient of `lhs` and `rhs`, rounding the result towards
/// positive infinity.
// FIXME: Remove when stable
#[inline]
pub fn div_ceil<T>(lhs: T, rhs: T) -> T
where
    T: PartialOrd + Copy + Div + Rem + Add + Unsigned + Zero + One,
{
    let d = lhs / rhs;
    let r = lhs % rhs;
    if r > T::zero() && rhs > T::zero() {
        d + T::one()
    } else {
        d
    }
}

/// Checks if the index is in bounds and returns a reference to the data at that
/// point if it exists.
#[inline]
pub fn get_in_bounds<T>(width: i64, height: i64, x: i64, y: i64, image: &[T]) -> Option<&T> {
    if (0..width).contains(&x) && (0..height).contains(&y) {
        let i = u64::try_from(y)
            .ok()?
            .checked_mul(u64::try_from(width).ok()?)?
            .checked_add(u64::try_from(x).ok()?)
            .and_then(|i| usize::try_from(i).ok())?;
        image.get(i)
    } else {
        None
    }
}

/// Checks if the index is in bounds and returns a mutable referance to the data
/// at that point if it exists.
#[inline]
pub fn get_mut_in_bounds<T>(
    width: i64,
    height: i64,
    x: i64,
    y: i64,
    image: &mut [T],
) -> Option<&mut T> {
    if (0..width).contains(&x) && (0..height).contains(&y) {
        let i = u64::try_from(y)
            .ok()?
            .checked_mul(u64::try_from(width).ok()?)?
            .checked_add(u64::try_from(x).ok()?)
            .and_then(|i| usize::try_from(i).ok())?;
        image.get_mut(i)
    } else {
        None
    }
}

/// Checks if the index is in bounds and returns the pixel data at that point if it exists.
#[inline]
pub fn get_pixel(x: i32, y: i32, image: &Array3<u8>) -> Option<ArrayView1<u8>> {
    if (0..(image.shape()[0] as i32)).contains(&y) && (0..(image.shape()[1] as i32)).contains(&x) {
        Some(image.slice(s![y, x, ..]))
    } else {
        None
    }
}
