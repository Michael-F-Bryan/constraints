//! [`Expression`] operations.

use crate::algebra::{BinaryOperation, Expression};
use euclid::approxeq::ApproxEq;

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
            "sin" => Ok(argument.sin()),
            "cos" => Ok(argument.cos()),
            "tan" => Ok(argument.tan()),
            "asin" => Ok(argument.asin()),
            "acos" => Ok(argument.acos()),
            "atan" => Ok(argument.atan()),
            "sqrt" => Ok(argument.sqrt()),
            _ => Err(EvaluationError::UnknownFunction),
        }
    }
}

/// Contextual information used when evaluating an [`Expression`].
pub trait Context {
    fn evaluate_function(
        &self,
        name: &str,
        argument: f64,
    ) -> Result<f64, EvaluationError>;
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

#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    UnknownFunction,
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
}
