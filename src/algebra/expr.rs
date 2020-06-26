#![allow(dead_code)]
use std::{
    fmt::{self, Display, Formatter},
    ops::{Add, Div, Mul, Neg, Sub},
};

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Parameter(Parameter),
    Constant(f64),
    /// An expression involving two operands.
    Binary {
        left: Box<Expression>,
        right: Box<Expression>,
        op: BinaryOperation,
    },
    /// Negate the expression.
    Negate(Box<Expression>),
    /// Invoke a builtin function.
    FunctionCall {
        function: Builtin,
        operand: Box<Expression>,
    },
}

impl Expression {
    fn is_compound(&self) -> bool {
        match self {
            Expression::Parameter(_)
            | Expression::Constant(_)
            | Expression::FunctionCall { .. } => false,
            Expression::Negate(inner) => inner.is_compound(),
            Expression::Binary { .. } => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {}

/// An operation that can be applied to two arguments.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BinaryOperation {
    Plus,
    Minus,
    Times,
    Divide,
}

/// Various builtin functions.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Builtin {
    Sqrt,
    Square,
    Sine,
    Cosine,
    ArcSine,
    ArcCosine,
}

impl Display for Builtin {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Builtin::Sqrt => write!(f, "sqrt"),
            Builtin::Square => write!(f, "square"),
            Builtin::Sine => write!(f, "sin"),
            Builtin::Cosine => write!(f, "cos"),
            Builtin::ArcSine => write!(f, "asin"),
            Builtin::ArcCosine => write!(f, "acos"),
        }
    }
}

// define some operator overloads to make constructing an expression easier.

impl Add for Expression {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Expression {
        Expression::Binary {
            left: Box::new(self),
            right: Box::new(rhs),
            op: BinaryOperation::Plus,
        }
    }
}

impl Sub for Expression {
    type Output = Expression;

    fn sub(self, rhs: Expression) -> Expression {
        Expression::Binary {
            left: Box::new(self),
            right: Box::new(rhs),
            op: BinaryOperation::Minus,
        }
    }
}

impl Mul for Expression {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Expression {
        Expression::Binary {
            left: Box::new(self),
            right: Box::new(rhs),
            op: BinaryOperation::Times,
        }
    }
}

impl Div for Expression {
    type Output = Expression;

    fn div(self, rhs: Expression) -> Expression {
        Expression::Binary {
            left: Box::new(self),
            right: Box::new(rhs),
            op: BinaryOperation::Divide,
        }
    }
}

impl Neg for Expression {
    type Output = Expression;

    fn neg(self) -> Self::Output { Expression::Negate(Box::new(self)) }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Parameter(_) => unimplemented!(),
            Expression::Constant(value) => write!(f, "{}", value),
            Expression::Binary { left, right, op } => {
                write_compound(left, f)?;

                let op = match op {
                    BinaryOperation::Plus => " + ",
                    BinaryOperation::Minus => " - ",
                    BinaryOperation::Times => "*",
                    BinaryOperation::Divide => "/",
                };
                write!(f, "{}", op)?;

                write_compound(right, f)?;

                Ok(())
            },
            Expression::Negate(inner) => {
                write!(f, "-")?;
                write_compound(inner, f)?;
                Ok(())
            },
            Expression::FunctionCall { function, operand } => {
                write!(f, "{}({})", function, operand)
            },
        }
    }
}

fn write_compound(expr: &Expression, f: &mut Formatter<'_>) -> fmt::Result {
    if expr.is_compound() {
        write!(f, "({})", expr)
    } else {
        write!(f, "{}", expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let inputs = vec![
            (Expression::Constant(3.0), "3"),
            (
                Expression::FunctionCall {
                    function: Builtin::Sine,
                    operand: Box::new(Expression::Constant(5.0)),
                },
                "sin(5)",
            ),
            (
                Expression::Negate(Box::new(Expression::Constant(5.0))),
                "-5",
            ),
            (
                Expression::Negate(Box::new(Expression::FunctionCall {
                    function: Builtin::Sine,
                    operand: Box::new(Expression::Constant(5.0)),
                })),
                "-sin(5)",
            ),
            (
                Expression::Binary {
                    left: Box::new(Expression::Constant(1.0)),
                    right: Box::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Plus,
                },
                "1 + 1",
            ),
            (
                Expression::Binary {
                    left: Box::new(Expression::Constant(1.0)),
                    right: Box::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Minus,
                },
                "1 - 1",
            ),
            (
                Expression::Binary {
                    left: Box::new(Expression::Constant(1.0)),
                    right: Box::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Times,
                },
                "1*1",
            ),
            (
                Expression::Binary {
                    left: Box::new(Expression::Constant(1.0)),
                    right: Box::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Divide,
                },
                "1/1",
            ),
            (
                Expression::Binary {
                    left: Box::new(Expression::Binary {
                        left: Box::new(Expression::Constant(1.0)),
                        right: Box::new(Expression::Constant(2.0)),
                        op: BinaryOperation::Plus,
                    }),
                    right: Box::new(Expression::Constant(3.0)),
                    op: BinaryOperation::Divide,
                },
                "(1 + 2)/3",
            ),
        ];

        for (expr, should_be) in inputs {
            let got = expr.to_string();
            assert_eq!(got, should_be);
        }
    }
}
