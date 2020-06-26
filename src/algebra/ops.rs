//! [`Expression`] operations.

use crate::algebra::{BinaryOperation, Expression, Parameter};
use euclid::approxeq::ApproxEq;

/// Contextual information used when evaluating an [`Expression`].
pub trait Context {
    fn evaluate_function(
        &self,
        name: &str,
        argument: f64,
    ) -> Result<f64, EvaluationError>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    UnknownFunction,
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
            _ => Err(EvaluationError::UnknownFunction),
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
            Expression::Negate(Box::new(right))
        },

        // x - 0 = x
        (left, Expression::Constant(r), BinaryOperation::Minus)
            if r.approx_eq(&0.0) =>
        {
            left
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
        Expression::Negate(inner) => {
            Expression::Negate(Box::new(substitute(inner, param, value)))
        },
        Expression::FunctionCall { function, argument } => {
            Expression::FunctionCall {
                function: function.clone(),
                argument: Box::new(substitute(argument, param, value)),
            }
        },
    }
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
            ("x + 5", "x + 5"),
            ("x + 5*2", "x + 10"),
            ("0 + x", "x"),
            ("x + 0", "x"),
            ("1 * x", "x"),
            ("x * 1", "x"),
            ("x - 0", "x"),
            ("0 - x", "-x"),
            ("x / 1", "x"),
            ("--x", "x"),
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
}
