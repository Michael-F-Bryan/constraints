//! [`Expression`] operations.

use crate::algebra::{BinaryOperation, Expression};

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
            let left = fold_constants(left, ctx);
            let right = fold_constants(right, ctx);

            match (left, right) {
                (Expression::Constant(l), Expression::Constant(r)) => {
                    let value = match op {
                        BinaryOperation::Plus => l + r,
                        BinaryOperation::Minus => l - r,
                        BinaryOperation::Times => l * r,
                        BinaryOperation::Divide => l / r,
                    };

                    Expression::Constant(value)
                },
                (left, right) => Expression::Binary {
                    left: Box::new(left),
                    right: Box::new(right),
                    op: *op,
                },
            }
        },
        Expression::Negate(_) => unimplemented!(),
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
            ("2 * 3", 2.0 * 3.0),
            ("sqrt(4)", 2.0),
            ("sqrt(2 + 2)", 2.0),
            ("sqrt(2 + sqrt(4))", 2.0),
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
            ("unknown_function(3)", "unknown_function(3)"),
            ("x + 5", "x + 5"),
            ("x + 5*2", "x + 10"),
        ];
        let ctx = Builtins::default();

        for (src, should_be) in inputs {
            let expr: Expression = src.parse().unwrap();

            let got = fold_constants(&expr, &ctx);

            assert_eq!(got.to_string(), should_be);
        }
    }
}
