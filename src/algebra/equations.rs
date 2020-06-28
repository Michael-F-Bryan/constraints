use crate::algebra::{
    matrix::Matrix,
    ops::{self, Context, EvaluationError},
    Expression, Parameter, ParseError,
};
use std::{
    collections::HashMap,
    fmt::Debug,
    iter::{Extend, FromIterator},
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

    pub fn solve<C>(self, ctx: &C) -> Result<Solution, SolveError>
    where
        C: Context,
    {
        let jacobian = Jacobian::create(&self.equations, ctx)?;
        let got = solve_with_newtons_method(&jacobian, ctx)?;

        Ok(Solution { known_values: got })
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
    // To solve the function, F, using Newton's method:
    //   x_next = x_current - jacobian(F).inverse() * F(x_current)
    //
    // See also: https://en.wikipedia.org/wiki/Newton%27s_method#Nonlinear_systems_of_equations

    let mut solution = jacobian.initial_values();

    for _ in 0..MAX_ITERATIONS {
        // In each iteration we need to evaluate the symbolic jacobian matrix
        // with our tentative values. This gives us the derivative of each
        // parameter for each equation.
        let inverted_jacobian = jacobian.evaluate(&solution, ctx)?.inverted();

        // Take the Newton step using our refined solution
        //
        //   x_next = x_current - jacobian(F).inverse() * F(x_current)
        //      J(x_n) (x_{n+1} - x_n) = 0 - F(x_n)

        // we need to re-evaluate the functions because our parameters just
        // changed
        let re_evaluated = jacobian.evaluate(&solution, ctx)?;

        // and check to see if we've converged yet
        todo!()
    }

    todo!()
}

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

    fn initial_values(&self) -> Vec<f64> { vec![0.0; self.unknowns.len()] }

    fn evaluate<C>(
        &self,
        current_values: &[f64],
        ctx: &C,
    ) -> Result<Matrix<f64>, EvaluationError>
    where
        C: Context,
    {
        self.matrix.try_map(|column, _row, expr| {
            let value = current_values[column];
            let param = &self.unknowns[column];

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
}
