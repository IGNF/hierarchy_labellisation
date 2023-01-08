use std::collections::VecDeque;

use num_traits::Float;

#[derive(Debug, Clone)]
pub struct PlefPiece<T: Float> {
    pub start_x: T,
    pub start_y: T,
    pub slope: T,
}

impl<T: Float> PlefPiece<T> {
    pub fn new(start_x: T, start_y: T, slope: T) -> Self {
        Self {
            start_x,
            start_y,
            slope,
        }
    }

    pub fn eval(&self, x: T) -> T {
        self.start_y + self.slope * (x - self.start_x)
    }
}

/// A Plef is a piecewise linear energy function.
/// It is used by the Mumford-Shah algorithm to compute the optimal segmentation.
/// It contains a set of pieces which are linear functions. It is concave.
#[derive(Debug, Clone)]
pub struct Plef<T: Float> {
    pub pieces: VecDeque<PlefPiece<T>>,
}

impl<T: Float> Plef<T> {
    pub fn init() -> Self {
        Self {
            pieces: VecDeque::new(),
        }
    }

    pub fn sum(&self, other: &Self, max_pieces: Option<u32>) -> Self {
        if other.pieces.is_empty() {
            return self.clone();
        } else if self.pieces.is_empty() {
            return other.clone();
        }

        let max_p = max_pieces.unwrap_or(10);

        let mut res = Self::init();

        let mut i = self.pieces.iter().rev().peekable();
        let mut j = other.pieces.iter().rev().peekable();

        let mut count = 0;

        while i.peek().is_some() && j.peek().is_some() && count < max_p {
            let piece_i = i.peek().unwrap();
            let piece_j = j.peek().unwrap();

            let new_slope = piece_i.slope + piece_j.slope;

            count += 1;

            let new_start_x;
            let new_start_y;

            if piece_i.start_x >= piece_j.start_x {
                new_start_x = piece_i.start_x;
                new_start_y = piece_i.start_y + piece_j.eval(new_start_x);
                if piece_i.start_x == piece_j.start_x {
                    j.next();
                }
                i.next();
            } else {
                new_start_x = piece_j.start_x;
                new_start_y = piece_j.start_y + piece_i.eval(new_start_x);
                j.next();
            }

            res.pieces
                .push_front(PlefPiece::new(new_start_x, new_start_y, new_slope));
        }

        if let Some(first_piece) = res.pieces.front_mut() {
            if first_piece.start_x > T::zero() {
                first_piece.start_y = first_piece.start_y - first_piece.slope * first_piece.start_x;
                first_piece.start_x = T::zero();
            }
        }

        res
    }

    pub fn infimum(&mut self, other: PlefPiece<T>) -> T {
        let mut i = self.pieces.len() as i32 - 1;

        let last_piece = self.pieces.back().unwrap();

        if other.slope == last_piece.slope {
            let y = other.eval(last_piece.start_x);
            if y > last_piece.start_y {
                return T::infinity();
            } else if y == last_piece.start_y {
                return last_piece.start_x;
            } else {
                self.pieces.pop_back();
                i -= 1;
            }
        }

        let mut xi = T::zero();
        for i in (0..(i + 1)).rev() {
            let piece = &self.pieces[i as usize];
            xi = (other.start_x * other.slope
                - piece.start_x * piece.slope
                - (other.start_y - piece.start_y))
                / (other.slope - piece.slope);
            if xi > piece.start_x {
                break;
            } else {
                self.pieces.pop_back();
            }
        }

        self.pieces
            .push_back(PlefPiece::new(xi, other.eval(xi), other.slope));

        xi
    }
}

impl From<PlefPiece<f64>> for Plef<f64> {
    fn from(piece: PlefPiece<f64>) -> Self {
        let mut plef = Self::init();
        plef.pieces.push_back(piece);
        plef
    }
}
