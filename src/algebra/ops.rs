//! [`Expression`] operations.

use crate::algebra::{BinaryOperation, Expression, Parameter};
use euclid::approxeq::ApproxEq;
use smol_str::SmolStr;

/// Contextual information used when evaluating an [`Expression`].
pub trait Context {
    fn evaluate_function(
        &self,
        name: &str,
        argument: f64,
    ) -> Result<f64, EvaluationError>;

    /// For some [`Parameter`], `x`, and function, `f`, get `f'(x)`.
    fn differentiate_function(
        &self,
        name: &str,
        param: &Parameter,
    ) -> Result<Expression, EvaluationError>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    UnknownFunction { name: SmolStr },
    UnableToDifferentiate { name: SmolStr },
}

/// The set of builtin functions.
#[derive(Debug, Default)]
struct Builtins;

impl Context for Builtins {
    fn evaluate_function(
        &self,
        name: &str,
        argument: f64,
    ) -> Result<f64, EvaluationError> {
        match name {
            "sin" => Ok(argument.to_radians().sin()),
            "cos" => Ok(argument.to_radians().cos()),
            "tan" => Ok(argument.to_radians().tan()),
            "asin" => Ok(argument.asin().to_degrees()),
            "acos" => Ok(argument.acos().to_degrees()),
            "atan" => Ok(argument.atan().to_degrees()),
            "sqrt" => Ok(argument.sqrt()),
            _ => Err(EvaluationError::UnknownFunction { name: name.into() }),
        }
    }

    fn differentiate_function(
        &self,
        name: &str,
        param: &Parameter,
    ) -> Result<Expression, EvaluationError> {
        match name {
            "sin" => Ok(Expression::FunctionCall {
                function: "cos".into(),
                argument: Box::new(Expression::Parameter(param.clone())),
            }),
            "cos" => Ok(-Expression::FunctionCall {
                function: "sin".into(),
                argument: Box::new(Expression::Parameter(param.clone())),
            }),
            "sqrt" => {
                let sqrt_x = Expression::FunctionCall {
                    function: "sqrt".into(),
                    argument: Box::new(Expression::Parameter(param.clone())),
                };
                Ok(Expression::Constant(0.5) / sqrt_x)
            },
            _ => Err(EvaluationError::UnableToDifferentiate {
                name: name.into(),
            }),
        }
    }
}

/// Simplify an expression by evaluating all constant operations.
pub fn fold_constants<C>(expr: &Expression, ctx: &C) -> Expression
where
    C: Context,
{
    match expr {
        Expression::Binary { left, right, op } => {
            fold_binary_op(left, right, *op, ctx)
        },
        Expression::Negate(expr) => match fold_constants(expr, ctx) {
            Expression::Constant(value) => Expression::Constant(-value),
            // double negative
            Expression::Negate(inner) => *inner,
            other => Expression::Negate(Box::new(other)),
        },
        Expression::FunctionCall { function, argument } => {
            let argument = fold_constants(argument, ctx);

            if let Expression::Constant(argument) = argument {
                if let Ok(result) = ctx.evaluate_function(function, argument) {
                    return Expression::Constant(result);
                }
            }

            Expression::FunctionCall {
                function: function.clone(),
                argument: Box::new(argument),
            }
        },
        _ => expr.clone(),
    }
}

fn fold_binary_op<C>(
    left: &Expression,
    right: &Expression,
    op: BinaryOperation,
    ctx: &C,
) -> Expression
where
    C: Context,
{
    let left = fold_constants(left, ctx);
    let right = fold_constants(right, ctx);

    // If our operands contain constants, we can use arithmetic's identity laws
    // to simplify things
    match (left, right, op) {
        (
            Expression::Parameter(p_left),
            Expression::Parameter(p_right),
            BinaryOperation::Plus,
        ) if p_left == p_right => {
            Expression::Constant(2.0) * Expression::Parameter(p_right.clone())
        },
        (
            Expression::Parameter(p_left),
            Expression::Parameter(p_right),
            BinaryOperation::Minus,
        ) if p_left == p_right => Expression::Constant(0.0),
        (
            Expression::Parameter(p_left),
            Expression::Parameter(p_right),
            BinaryOperation::Divide,
        ) if p_left == p_right => Expression::Constant(1.0),

        // x + 0 = x
        (Expression::Constant(l), right, BinaryOperation::Plus)
            if l.approx_eq(&0.0) =>
        {
            right
        },
        (left, Expression::Constant(r), BinaryOperation::Plus)
            if r.approx_eq(&0.0) =>
        {
            left
        },

        // 0 * x = 0
        (Expression::Constant(l), _, BinaryOperation::Times)
            if l.approx_eq(&0.0) =>
        {
            Expression::Constant(0.0)
        },
        (_, Expression::Constant(r), BinaryOperation::Times)
            if r.approx_eq(&0.0) =>
        {
            Expression::Constant(0.0)
        },

        // 1 * x = x
        (Expression::Constant(l), right, BinaryOperation::Times)
            if l.approx_eq(&1.0) =>
        {
            right
        },
        (left, Expression::Constant(r), BinaryOperation::Times)
            if r.approx_eq(&1.0) =>
        {
            left
        },

        // 0 / x = 0
        (Expression::Constant(l), _, BinaryOperation::Divide)
            if l.approx_eq(&0.0) =>
        {
            Expression::Constant(0.0)
        },

        // x / 1 = x
        (left, Expression::Constant(r), BinaryOperation::Divide)
            if r.approx_eq(&1.0) =>
        {
            left
        },

        // 0 - x = -x
        (Expression::Constant(l), right, BinaryOperation::Minus)
            if l.approx_eq(&0.0) =>
        {
            -right
        },

        // x - 0 = x
        (left, Expression::Constant(r), BinaryOperation::Minus)
            if r.approx_eq(&0.0) =>
        {
            left
        },

        // (x * y) * z
        (
            Expression::Constant(constant_a),
            Expression::Binary {
                left,
                right,
                op: BinaryOperation::Times,
            },
            BinaryOperation::Times,
        ) if left.is_constant() || right.is_constant() => {
            let (constant_b, expr) = match (&*left, &*right) {
                (Expression::Constant(left), right) => (left, right),
                (left, Expression::Constant(right)) => (right, left),
                _ => unreachable!(),
            };
            Expression::Constant(constant_a * constant_b)
                * Expression::clone(expr)
        },
        (
            Expression::Binary {
                left,
                right,
                op: BinaryOperation::Times,
            },
            Expression::Constant(constant_a),
            BinaryOperation::Times,
        ) if left.is_constant() || right.is_constant() => {
            let (constant_b, expr) = match (&*left, &*right) {
                (Expression::Constant(left), right) => (left, right),
                (left, Expression::Constant(right)) => (right, left),
                _ => unreachable!(),
            };
            Expression::Constant(constant_a * constant_b)
                * Expression::clone(expr)
        },

        // Evaluate in-place
        (Expression::Constant(l), Expression::Constant(r), op) => {
            let value = match op {
                BinaryOperation::Plus => l + r,
                BinaryOperation::Minus => l - r,
                BinaryOperation::Times => l * r,
                BinaryOperation::Divide => l / r,
            };

            Expression::Constant(value)
        },

        // Oh well, we tried
        (left, right, op) => Expression::Binary {
            left: Box::new(left),
            right: Box::new(right),
            op,
        },
    }
}

/// Replace all references to a [`Parameter`] with an [`Expression`].
pub fn substitute(
    expression: &Expression,
    param: &Parameter,
    value: &Expression,
) -> Expression {
    match expression {
        Expression::Parameter(p) => {
            if p == param {
                value.clone()
            } else {
                Expression::Parameter(p.clone())
            }
        },
        Expression::Constant(value) => Expression::Constant(*value),
        Expression::Binary { left, right, op } => {
            let left = substitute(left, param, value);
            let right = substitute(right, param, value);
            Expression::Binary {
                left: Box::new(left),
                right: Box::new(right),
                op: *op,
            }
        },
        Expression::Negate(inner) => -substitute(inner, param, value),
        Expression::FunctionCall { function, argument } => {
            Expression::FunctionCall {
                function: function.clone(),
                argument: Box::new(substitute(argument, param, value)),
            }
        },
    }
}

/// Calculate an [`Expression`]'s partial derivative with respect to a
/// particular [`Parameter`].
pub fn partial_derivative<C>(
    expr: &Expression,
    param: &Parameter,
    ctx: &C,
) -> Result<Expression, EvaluationError>
where
    C: Context,
{
    let got = match expr {
        Expression::Parameter(p) => {
            if p == param {
                Expression::Constant(1.0)
            } else {
                Expression::Constant(0.0)
            }
        },
        Expression::Constant(_) => Expression::Constant(0.0),
        Expression::Binary {
            left,
            right,
            op: BinaryOperation::Plus,
        } => {
            partial_derivative(left, param, ctx)?
                + partial_derivative(right, param, ctx)?
        },
        Expression::Binary {
            left,
            right,
            op: BinaryOperation::Minus,
        } => {
            partial_derivative(left, param, ctx)?
                - partial_derivative(right, param, ctx)?
        },
        Expression::Binary {
            left,
            right,
            op: BinaryOperation::Times,
        } => {
            // The product rule
            let d_left = partial_derivative(left, param, ctx)?;
            let d_right = partial_derivative(right, param, ctx)?;
            let left = Expression::clone(left);
            let right = Expression::clone(right);

            d_left * right + d_right * left
        },
        Expression::Binary {
            left,
            right,
            op: BinaryOperation::Divide,
        } => {
            // The quotient rule
            let d_left = partial_derivative(left, param, ctx)?;
            let d_right = partial_derivative(right, param, ctx)?;
            let right = Expression::clone(right);
            let left = Expression::clone(left);

            (d_left * right.clone() + left * d_right) / (right.clone() * right)
        },

        Expression::Negate(inner) => -partial_derivative(inner, param, ctx)?,
        Expression::FunctionCall { function, argument } => {
            // implement the chain rule: (f o g)' = (f' o g) * g'
            let g = Parameter::named("__temp__");
            let f_dash_of_g = ctx.differentiate_function(function, &g)?;
            let g_dash = partial_derivative(argument, param, ctx)?;

            substitute(&f_dash_of_g, &g, argument) * g_dash
        },
    };

    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_fold_simple_arithmetic() {
        let inputs = vec![
            ("1", 1.0),
            ("1 + 1.5", 1.0 + 1.5),
            ("1 - 1.5", 1.0 - 1.5),
            ("2 * 3", 2.0 * 3.0),
            ("4 / 2", 4.0 / 2.0),
            ("sqrt(4)", 4_f64.sqrt()),
            ("sqrt(2 + 2)", (2_f64 + 2.0).sqrt()),
            ("sin(90)", 90_f64.to_radians().sin()),
            ("atan(1)", 45.0),
            ("sqrt(2 + sqrt(4))", (2.0 + 4_f64.sqrt()).sqrt()),
            ("-(1 + 2)", -(1.0 + 2.0)),
            ("0 * x", 0.0),
            ("x - x", 0.0),
            ("x/x", 1.0),
        ];
        let ctx = Builtins::default();

        for (src, should_be) in inputs {
            let expr: Expression = src.parse().unwrap();
            let got = fold_constants(&expr, &ctx);

            match got {
                Expression::Constant(value) => assert_eq!(
                    value, should_be,
                    "{} -> {} != {}",
                    expr, value, should_be
                ),
                other => panic!(
                    "Expected a constant expression, but got \"{}\"",
                    other
                ),
            }
        }
    }

    #[test]
    fn constant_folding_leaves_unknowns_unevaluated() {
        let inputs = vec![
            ("x", "x"),
            ("-(2 * 3 + x)", "-(6 + x)"),
            ("unknown_function(3)", "unknown_function(3)"),
            ("2 * x * 3", "6 * x"),
            ("x + 5", "x + 5"),
            ("x + 5*2", "x + 10"),
            ("x + x", "2*x"),
            ("0 + x", "x"),
            ("x + 0", "x"),
            ("1 * x", "x"),
            ("x * 1", "x"),
            ("x - 0", "x"),
            ("0 - x", "-x"),
            ("x / 1", "x"),
            ("--x", "x"),
            ("(x + x)*3 + 5", "6*x + 5"),
        ];
        let ctx = Builtins::default();

        for (src, should_be) in inputs {
            let expr: Expression = src.parse().unwrap();

            let got = fold_constants(&expr, &ctx);

            let should_be: Expression = should_be.parse().unwrap();

            assert_eq!(got, should_be, "{} != {}", got, should_be);
        }
    }

    #[test]
    fn basic_substitutions() {
        let parameter = Parameter::named("x");
        let inputs = vec![
            ("1 + 2", "3", "1 + 2"),
            ("x", "5", "5"),
            ("y", "5", "y"),
            ("x + 5", "5", " 5 + 5"),
            ("-x", "5", "-5"),
            ("sin(x)", "y + y", "sin(y + y)"),
        ];

        for (src, new_value, should_be) in inputs {
            let original: Expression = src.parse().unwrap();
            let new_value: Expression = new_value.parse().unwrap();
            let should_be: Expression = should_be.parse().unwrap();

            let got = substitute(&original, &parameter, &new_value);

            assert_eq!(got, should_be, "{} != {}", got, should_be);
        }
    }

    #[test]
    fn differentiate_wrt_x() {
        let x = Parameter::named("x");
        let inputs = vec![
            ("x", "1"),
            ("1", "0"),
            ("x*x", "2 * x"),
            ("3*x*x + 5*x + 2", "6*x + 5"),
            ("x - y", "1"),
            ("sin(x)", "cos(x)"),
            ("cos(x)", "-sin(x)"),
            ("sqrt(x)", "0.5 / sqrt(x)"),
            ("x/y", "y/y*y"), // = 1/y, simplification just isn't smart enough
        ];
        let ctx = Builtins::default();

        for (src, should_be) in inputs {
            let original: Expression = src.parse().unwrap();
            let should_be: Expression = should_be.parse().unwrap();

            let got = partial_derivative(&original, &x, &ctx).unwrap();
            let got = fold_constants(&got, &ctx);

            assert_eq!(got, should_be, "{} != {}", got, should_be);
        }
    }
}
