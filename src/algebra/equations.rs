use crate::algebra::{
    ops::{self, Context, EvaluationError},
    Expression, Parameter, ParseError,
};
use std::{
    collections::HashMap,
    fmt::{self, Debug, Formatter},
    iter::{Extend, FromIterator},
    ops::Index,
    str::FromStr,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    body: Expression,
}

impl Equation {
    pub fn new(left: Expression, right: Expression) -> Self {
        debug_assert_ne!(
            left.params().count() + right.params().count(),
            0,
            "Equations should contain at least one unknown"
        );
        Equation { body: left - right }
    }
}

impl FromStr for Equation {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (left, right) = match s.find("=") {
            Some(index) => {
                let (left, right) = s.split_at(index);
                (left, &right[1..])
            },
            None => todo!(),
        };

        Ok(Equation::new(left.parse()?, right.parse()?))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolveError {
    Eval(EvaluationError),
}

impl From<EvaluationError> for SolveError {
    fn from(e: EvaluationError) -> Self { SolveError::Eval(e) }
}

/// A builder for constructing a system of equations and solving them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SystemOfEquations {
    equations: Vec<Equation>,
}

impl SystemOfEquations {
    pub fn new() -> Self { SystemOfEquations::default() }

    pub fn with(mut self, equation: Equation) -> Self {
        self.push(equation);
        self
    }

    pub fn push(&mut self, equation: Equation) {
        self.equations.push(equation);
    }

    pub fn solve<C>(self, ctx: &C) -> Result<Solution, EvaluationError>
    where
        C: Context,
    {
        let jacobian = Jacobian::create(&self.equations, ctx)?;
        let got = solve_with_newtons_method(&jacobian, ctx);

        todo!()
    }
}

impl Extend<Equation> for SystemOfEquations {
    fn extend<T: IntoIterator<Item = Equation>>(&mut self, iter: T) {
        self.equations.extend(iter);
    }
}

impl FromIterator<Equation> for SystemOfEquations {
    fn from_iter<T: IntoIterator<Item = Equation>>(iter: T) -> Self {
        SystemOfEquations {
            equations: Vec::from_iter(iter),
        }
    }
}

fn solve_with_newtons_method<C>(
    jacobian: &Jacobian,
    ctx: &C,
) -> Result<HashMap<Parameter, f64>, SolveError>
where
    C: Context,
{
    const MAX_ITERATIONS: usize = 50;

    let mut solution = jacobian.initial_values();

    for _ in 0..MAX_ITERATIONS {
        // In each iteration we need to evaluate the symbolic jacobian matrix
        // with our tentative values. This gives us the derivative of each
        // parameter for each equation.
        let coefficients = jacobian.evaluate(&solution, ctx)?;

        // Next, use least squares to refine the solution
        solve_least_squares(
            coefficients,
            &jacobian.unknowns,
            &mut solution,
            ctx,
        )?;

        // Take the Newton step using our refined solution;
        //      J(x_n) (x_{n+1} - x_n) = 0 - F(x_n)
        todo!();

        // we need to re-evaluate the functions because our parameters just
        // changed
        let re_evaluated = jacobian.evaluate(&solution, ctx)?;

        // and check to see if we've converged yet
        todo!()
    }

    todo!()
}

fn solve_least_squares<C>(
    matrix: Matrix<f64>,
    unknowns: &[Parameter],
    unknown_values: &mut HashMap<Parameter, f64>,
    ctx: &C,
) -> Result<(), SolveError>
where
    C: Context,
{
    todo!()
}

/// Solve the matrix equation `Ax = b`
fn solve_linear_system() {}

#[derive(Debug, Clone, PartialEq)]
struct Jacobian {
    matrix: Matrix<Expression>,
    unknowns: Vec<Parameter>,
}

impl Jacobian {
    fn create<C>(
        equations: &[Equation],
        ctx: &C,
    ) -> Result<Self, EvaluationError>
    where
        C: Context,
    {
        let mut all_params: Vec<_> = equations
            .iter()
            .flat_map(|eq| eq.body.params())
            .cloned()
            .collect();
        all_params.sort();
        all_params.dedup();

        let matrix = Matrix::try_init(
            all_params.len(),
            equations.len(),
            |column, row| {
                let equation = &equations[row];
                let param = &all_params[column];

                if equation.body.depends_on(param) {
                    ops::partial_derivative(&equation.body, param, ctx)
                        .map(|derivative| ops::fold_constants(&derivative, ctx))
                } else {
                    Ok(Expression::Constant(0.0))
                }
            },
        )?;

        Ok(Jacobian {
            matrix,
            unknowns: all_params,
        })
    }

    fn initial_values(&self) -> HashMap<Parameter, f64> {
        self.unknowns
            .clone()
            .into_iter()
            .map(|p| (p, 0.0))
            .collect()
    }

    fn evaluate<C>(
        &self,
        params: &HashMap<Parameter, f64>,
        ctx: &C,
    ) -> Result<Matrix<f64>, EvaluationError>
    where
        C: Context,
    {
        self.matrix.try_map(|column, row, expr| {
            // evaluate the expression using the corresponding parameter's
            // value.
            //
            // Note: This relies on the fact that the i'th column corresponds to
            // the i'th parameter in the jacobian
            let param = &self.unknowns[column];
            let value = params[param];

            let expr =
                ops::substitute(expr, param, &Expression::Constant(value));
            ops::evaluate(&expr, ctx)
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    known_values: HashMap<Parameter, f64>,
}

#[derive(Clone, PartialEq)]
struct Matrix<T> {
    cells: Box<[T]>,
    columns: usize,
    rows: usize,
}

impl<T> Matrix<T> {
    fn init<F>(columns: usize, rows: usize, mut get_cell: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut cells = Vec::with_capacity(columns * rows);

        for row in 0..rows {
            for column in 0..columns {
                cells.push(get_cell(column, row));
            }
        }

        Matrix {
            cells: cells.into_boxed_slice(),
            columns,
            rows,
        }
    }

    fn try_init<F, E>(
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

    fn rows(&self) -> impl Iterator<Item = &[T]> + '_ {
        let rows = self.rows;
        let columns = self.columns;

        (0..rows)
            .map(move |row| row * columns..(row + 1) * columns)
            .map(move |range| &self.cells[range])
    }

    fn get(&self, column: usize, row: usize) -> Option<&T> {
        let ix = row * self.rows + column;
        self.cells.get(ix)
    }

    fn try_map<F, Q, E>(&self, mut func: F) -> Result<Matrix<Q>, E>
    where
        F: FnMut(usize, usize, &T) -> Result<Q, E>,
    {
        Matrix::try_init(self.columns, self.rows, |column, row| {
            func(column, row, &self[(column, row)])
        })
    }

    fn transpose(&self) -> Self
    where
        T: Clone,
    {
        Matrix::init(self.rows, self.columns, |column, row| {
            self[(row, column)].clone()
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ops::Builtins;

    #[test]
    #[ignore]
    fn single_equality() {
        let equation: Equation = "x = 5".parse().unwrap();
        let x = Parameter::named("x");
        let builtins = Builtins::default();

        let got = SystemOfEquations::new()
            .with(equation)
            .solve(&builtins)
            .unwrap();

        assert_eq!(got.known_values.len(), 1);
        assert_eq!(got.known_values[&x], 5.0);
    }

    #[test]
    fn matrix_representation() {
        let matrix = Matrix::init(3, 2, |column, row| column + row);
        let should_be = "[[0, 1, 2], [1, 2, 3]]";

        let got = format!("{:?}", matrix);

        assert_eq!(got, should_be);
    }
}
