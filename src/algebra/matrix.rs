//! A simplified version of a general-purpose matrix library, containing just
//! the operations and trait implementations we need.

use std::{
    fmt::{self, Debug, Formatter},
    ops::{Add, Index, IndexMut, Mul},
};

/// A general-purpose MxN matrix laid out seqentially in memory.
#[derive(Clone, PartialEq)]
pub(crate) struct Matrix<T> {
    cells: Box<[T]>,
    columns: usize,
    rows: usize,
}

impl<T> Matrix<T> {
    /// Create a new [`Matrix`] by invoking some `fn(column, row) -> T` function
    /// for each cell.
    pub fn init<F>(columns: usize, rows: usize, mut get_cell: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        use std::convert::Infallible;

        Matrix::try_init::<_, Infallible>(columns, rows, |column, row| {
            Ok(get_cell(column, row))
        })
        .expect(
            "This can never fail, all error checking should be optimised away",
        )
    }

    /// A version of [`Matrix::init()`] which lets you initialize a matrix using
    /// a function which may fail.
    pub fn try_init<F, E>(
        columns: usize,
        rows: usize,
        mut get_cell: F,
    ) -> Result<Self, E>
    where
        F: FnMut(usize, usize) -> Result<T, E>,
    {
        let mut cells = Vec::with_capacity(columns * rows);

        for row in 0..rows {
            for column in 0..columns {
                cells.push(get_cell(column, row)?);
            }
        }

        Ok(Matrix {
            cells: cells.into_boxed_slice(),
            columns,
            rows,
        })
    }

    pub fn rows(&self) -> impl Iterator<Item = &[T]> + '_ {
        let rows = self.rows;
        let columns = self.columns;

        (0..rows)
            .map(move |row| row * columns..(row + 1) * columns)
            .map(move |range| &self.cells[range])
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [T]> + '_ {
        self.cells.chunks_exact_mut(self.columns)
    }

    fn index(&self, column: usize, row: usize) -> usize {
        row * self.columns + column
    }

    pub fn get(&self, column: usize, row: usize) -> Option<&T> {
        self.cells.get(self.index(column, row))
    }

    pub fn get_mut(&mut self, column: usize, row: usize) -> Option<&mut T> {
        self.cells.get_mut(self.index(column, row))
    }

    pub fn try_map<F, Q, E>(&self, mut func: F) -> Result<Matrix<Q>, E>
    where
        F: FnMut(usize, usize, &T) -> Result<Q, E>,
    {
        Matrix::try_init(self.columns, self.rows, |column, row| {
            func(column, row, &self[(column, row)])
        })
    }

    pub fn transposed(&self) -> Self
    where
        T: Clone,
    {
        Matrix::init(self.rows, self.columns, |column, row| {
            self[(row, column)].clone()
        })
    }

    /// Evaluate some function, `fn(column, row, &mut item)` for each cell in
    /// the matrix.
    pub fn for_each_mut<F>(&mut self, mut func: F)
    where
        F: FnMut(usize, usize, &mut T),
    {
        self.cells_mut()
            .for_each(|(column, row, value)| func(column, row, value));
    }

    pub fn cells(&self) -> impl Iterator<Item = (usize, usize, &T)> + '_ {
        let columns = self.columns;
        let rows = self.rows;

        (0..rows)
            .flat_map(move |row| (0..columns).map(move |column| (column, row)))
            .map(move |(column, row)| (column, row, &self[(column, row)]))
    }

    pub fn cells_mut(
        &mut self,
    ) -> impl Iterator<Item = (usize, usize, &mut T)> + '_ {
        self.rows_mut().enumerate().flat_map(|(i, row)| {
            row.iter_mut()
                .enumerate()
                .map(move |(j, cell)| (j, i, cell))
        })
    }
}

impl Matrix<f64> {
    pub fn determinant(&self) -> f64 { todo!() }

    pub fn inverted(&self) -> Self {
        let mut transposed = self.transposed();
        let determinant = self.determinant();
        transposed.for_each_mut(|_, _, item| *item /= determinant);

        transposed
    }
}

impl<T: Debug> Debug for Matrix<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.rows()).finish()
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (column, row): (usize, usize)) -> &Self::Output {
        assert!(column < self.columns, "Column index out of bounds");
        assert!(row < self.rows, "Row index out of bounds");

        self.get(column, row)
            .expect("We've already done bounds checks")
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(
        &mut self,
        (column, row): (usize, usize),
    ) -> &mut Self::Output {
        assert!(column < self.columns, "Column index out of bounds");
        assert!(row < self.rows, "Row index out of bounds");

        self.get_mut(column, row)
            .expect("We've already done bounds checks")
    }
}

impl<T: Add> Add for Matrix<T> {
    type Output = Matrix<<T as Add>::Output>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        let Matrix {
            cells,
            columns,
            rows,
        } = self;
        assert_eq!(columns, rhs.columns);
        assert_eq!(rows, rhs.rows);

        let cells = cells
            .into_vec()
            .into_iter()
            .zip(rhs.cells.into_vec().into_iter())
            .map(|(left, right)| left + right)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Matrix {
            cells,
            columns,
            rows,
        }
    }
}

impl<T> Mul for Matrix<T>
where
    T: Add<Output = T> + Mul<Output = T> + Default + Clone,
{
    type Output = Matrix<T>;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.columns, other.rows);

        Matrix::init(other.columns, self.rows, |column, row| {
            let mut sum = T::default();

            for i in 0..other.columns {
                sum = sum + self[(i, row)].clone() * other[(column, i)].clone();
            }

            sum
        })
    }
}

/// A macro to work around the fact that you can't impl `From<[[T; n]; m]>` for
/// any constand `n` and `m` (aka const generics).
macro_rules! matrix_array_impls {
    ($n:expr, $($rest:expr),*) => {
        // the square matrix of this size
        matrix_array_impls!(@from $n, $n);

        $(
            // MxN
            matrix_array_impls!(@from $n, $rest);
            // NxM
            matrix_array_impls!(@from $rest, $n);
        )*

        // recurse
        matrix_array_impls!($($rest),*);
    };
    ($n:expr) => {
        // the zero case
        matrix_array_impls!(@from $n, $n);
    };
    (@from $n:expr, $m:expr) => {
        impl<T> From<[[T; $n]; $m]> for Matrix<T> {
            fn from(other: [[T; $n]; $m]) -> Self {
                let mut cells = Vec::<T>::with_capacity($n * $m);

                // SAFETY: We need to move the contents from the `other` array
                // into a vector, however arrays don't let you move items
                // directly (it'd leave the place logically uninitialized), so
                // we do a `memmove()` using raw pointers.
                unsafe {
                    debug_assert!(
                        cells.is_empty(),
                        "The vector should think it's empty",
                    );
                    debug_assert_eq!(
                        cells.capacity(),
                        $n * $m,
                        "We should have enough space to copy all the items across",
                    );

                    for (i, row) in other.iter().enumerate() {
                        let start_index = i * $n;
                        let ptr = cells.as_mut_ptr().offset(start_index as isize);
                        std::ptr::copy_nonoverlapping(row.as_ptr(), ptr, row.len());
                    }

                    // the items have been moved out of the array. Make sure
                    // `other` is treated as uninitialized memory from here on.
                    std::mem::forget(other);

                    // Tell the vector it has ownership of the items *after*
                    // copying the items across and treating `other` as
                    // uninitialized. That way if anything bad happened during
                    // the copy we'd just leak memory and not hit UB.
                    cells.set_len($n * $m);
                }

                Matrix {
                    cells: cells.into_boxed_slice(),
                    columns: $n,
                    rows: $m,
                }
            }
        }

        impl<T: PartialEq> PartialEq<[[T; $n]; $m]> for Matrix<T> {
            fn eq(&self, other: &[[T; $n]; $m]) -> bool {
                self.columns != $m && self.rows != $n
                    && self.cells()
                        .all(|(column, row, value)| *value == other[row][column])
            }
        }

        impl<T> Mul<[[T; $n]; $m]> for Matrix<T>
            where T: Add<Output = T> + Mul<Output = T> + Default + Clone
        {
            type Output = Matrix<T>;

            fn mul(self, other: [[T; $n]; $m]) -> Self {
                assert_eq!(self.columns, $m);

                Matrix::init($n, self.rows, |column, row| {
                    let mut sum = T::default();

                    for i in 0..$m {
                        sum = sum + self[(i, row)].clone() * other[i][column].clone();
                    }

                    sum
                })
            }
        }
    };
}

matrix_array_impls!(8, 7, 6, 5, 4, 3, 2, 1, 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_representation() {
        let matrix = Matrix::init(3, 2, |column, row| column + row);
        let should_be = "[[0, 1, 2], [1, 2, 3]]";

        let got = format!("{:?}", matrix);

        assert_eq!(got, should_be);
    }

    #[test]
    fn wide_matrix_from_array() {
        let array = [[1, 2, 3, 4], [5, 6, 7, 8]];

        let got = Matrix::from(array);

        assert_eq!(got, array);
    }

    #[test]
    fn tall_matrix_from_array() {
        let array = [[1, 2], [3, 4], [5, 6], [7, 8]];

        let got = Matrix::from(array);

        assert_eq!(got, array);
    }

    #[test]
    fn square_matrix_multiply() {
        let matrix = Matrix::from([[0, 1, 2], [3, 4, 5], [6, 7, 8]]);
        let vector = [[9], [10], [11]];
        let should_be = [[32], [122], [212]];

        let got = matrix * vector;

        assert_eq!(got, should_be);
    }
}
