use crate::algebra::{
    matrix::Matrix,
    ops::{self, Context, EvaluationError},
    Expression, Parameter, ParseError,
};
use std::{
    collections::{HashMap, HashSet},
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
        match s.find("=") {
            Some(index) => {
                let (left, right) = s.split_at(index);
                let right = &right[1..];
                Ok(Equation::new(left.parse()?, right.parse()?))
            },
            None => Ok(Equation { body: s.parse()? }),
        }
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
        let jacobian = self.jacobian(ctx)?;
        let got = solve_with_newtons_method(&jacobian, &self.equations, ctx)?;

        Ok(Solution {
            known_values: jacobian.unknowns.into_iter().zip(got).collect(),
        })
    }

    pub fn unknowns(&self) -> usize {
        let params: HashSet<_> = self
            .equations
            .iter()
            .flat_map(|eq| eq.body.params())
            .collect();

        params.len()
    }

    pub fn from_equations<E, S>(equations: E) -> Result<Self, ParseError>
    where
        E: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut system = SystemOfEquations::new();

        for equation in equations {
            system.push(equation.as_ref().parse()?);
        }

        Ok(system)
    }

    fn jacobian<C>(&self, ctx: &C) -> Result<Jacobian, EvaluationError>
    where
        C: Context,
    {
        Jacobian::create(&self.equations, ctx)
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
    equations: &[Equation],
    ctx: &C,
) -> Result<Vec<f64>, SolveError>
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
        let parameter_value = |p: &Parameter| {
            jacobian
                .unknowns
                .iter()
                .position(|other_p| p == other_p)
                .map(|ix| solution[ix])
        };
        let current_value = equations
            .iter()
            .map(|eq| ops::evaluate(&eq.body, &parameter_value, ctx))
            .collect::<Result<Vec<_>, _>>()?;
        let b = Matrix::column_vector(solution.clone())
            - dbg!(inverted_jacobian) * Matrix::column_vector(current_value);
        let new_solution = b.as_column_vector().unwrap().to_vec();

        // Note: If it's converged, `jacobian(F) * x` should be the zero vector
        if solution
            .iter()
            .zip(new_solution.iter())
            .all(|(left, right)| (left - right).abs() <= 1e-10)
        {
            break;
        }

        solution = new_solution;
    }

    Ok(solution)
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

    fn initial_values(&self) -> Vec<f64> {
        // TODO: Figure out a smarter way to get the initial guess. How fast
        // newton's method converges changes a lot depending on the quality of
        // your starting point.
        vec![0.0; self.unknowns.len()]
    }

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

            let getter = |param: &Parameter| {
                self.unknowns
                    .iter()
                    .position(|p| p == param)
                    .map(|parameter_index| current_values[parameter_index])
            };
            ops::evaluate(&expr, &getter, ctx)
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

    #[test]
    fn calculate_jacobian_of_known_system_of_equations() {
        // See https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant#Example_5
        let system = SystemOfEquations::from_equations(&[
            "5 * b",
            "4*a*a - 2*sin(b*c)",
            "b*c",
        ])
        .unwrap();
        let ctx = Builtins::default();

        let got = system.jacobian(&ctx).unwrap();

        assert_eq!(
            got.matrix.num_columns(),
            system.unknowns(),
            "There are 3 unknowns"
        );
        assert_eq!(
            got.matrix.num_rows(),
            system.equations.len(),
            "There are 3 equations"
        );
        // Note: I needed to rearrange the Wikipedia equations a bit because
        // our solver multiplied things differently (i.e. "c*2" instead of
        // "2*c")
        let should_be = [
            ["0", "5", "0"],
            ["8*a", "-cos(b*c)*c*2", "-cos(b*c)*b*2"],
            ["0", "c", "b"],
        ];
        let should_be: Matrix<Expression> = Matrix::from(should_be)
            .try_map(|_, _, expr| expr.parse())
            .unwrap();
        // Usually I wouldn't compare strings, but it's possible to get
        // different (but equivalent!) trees when calculating the jacobian vs
        // parsing from a string
        assert_eq!(got.matrix.to_string(), should_be.to_string());
    }

    #[test]
    fn solve_simple_equations() {
        let system =
            SystemOfEquations::from_equations(&["x-1", "y-2", "z-3"]).unwrap();
        let ctx = Builtins::default();
        let jacobian = system.jacobian(&ctx).unwrap();

        let got = solve_with_newtons_method(&jacobian, &system.equations, &ctx)
            .unwrap();

        let named_parameters: HashMap<Parameter, f64> =
            jacobian.unknowns.into_iter().zip(got).collect();
        let x = Parameter::named("x");
        let y = Parameter::named("y");
        let z = Parameter::named("z");
        assert_eq!(named_parameters[&x], 1.0);
        assert_eq!(named_parameters[&y], 2.0);
        assert_eq!(named_parameters[&z], 3.0);
    }
}
