use crate::algebra::parse::ParseError;
use smol_str::SmolStr;
use std::{
    fmt::{self, Display, Formatter},
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
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
        function: SmolStr,
        argument: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Parameter {
    Named(SmolStr),
    Anonymous { number: usize },
}

/// An operation that can be applied to two arguments.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BinaryOperation {
    Plus,
    Minus,
    Times,
    Divide,
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

impl FromStr for Expression {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> { crate::algebra::parse(s) }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Parameter(Parameter::Named(name)) => {
                write!(f, "{}", name)
            },
            Expression::Parameter(Parameter::Anonymous { number }) => {
                write!(f, "${}", number)
            },
            Expression::Constant(value) => write!(f, "{}", value),
            Expression::Binary { left, right, op } => {
                write_with_precedence(op.precedence(), left, f)?;

                let middle = match op {
                    BinaryOperation::Plus => " + ",
                    BinaryOperation::Minus => " - ",
                    BinaryOperation::Times => "*",
                    BinaryOperation::Divide => "/",
                };
                write!(f, "{}", middle)?;

                write_with_precedence(op.precedence(), right, f)?;

                Ok(())
            },
            Expression::Negate(inner) => {
                write!(f, "-")?;

                write_with_precedence(
                    BinaryOperation::Times.precedence(),
                    inner,
                    f,
                )?;
                Ok(())
            },
            Expression::FunctionCall { function, argument } => {
                write!(f, "{}({})", function, argument)
            },
        }
    }
}

impl Expression {
    fn precedence(&self) -> Precedence {
        match self {
            Expression::Parameter(_)
            | Expression::Constant(_)
            | Expression::FunctionCall { .. } => Precedence::Bi,
            Expression::Negate(_) => Precedence::Md,
            Expression::Binary { op, .. } => op.precedence(),
        }
    }
}

impl BinaryOperation {
    fn precedence(self) -> Precedence {
        match self {
            BinaryOperation::Plus | BinaryOperation::Minus => Precedence::As,
            BinaryOperation::Times | BinaryOperation::Divide => Precedence::Md,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
enum Precedence {
    Bi,
    Md,
    As,
}

fn write_with_precedence(
    current_precedence: Precedence,
    expr: &Expression,
    f: &mut Formatter<'_>,
) -> fmt::Result {
    if expr.precedence() > current_precedence {
        // we need parentheses to maintain ordering
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
                    function: "sin".into(),
                    argument: Box::new(Expression::Constant(5.0)),
                },
                "sin(5)",
            ),
            (
                Expression::Negate(Box::new(Expression::Constant(5.0))),
                "-5",
            ),
            (
                Expression::Negate(Box::new(Expression::FunctionCall {
                    function: "sin".into(),
                    argument: Box::new(Expression::Constant(5.0)),
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
                Expression::Negate(Box::new(Expression::Binary {
                    left: Box::new(Expression::Constant(1.0)),
                    right: Box::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Plus,
                })),
                "-(1 + 1)",
            ),
            (
                Expression::Negate(Box::new(Expression::Binary {
                    left: Box::new(Expression::Constant(1.0)),
                    right: Box::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Times,
                })),
                "-1*1",
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
            (
                Expression::Binary {
                    left: Box::new(Expression::Constant(3.0)),
                    right: Box::new(Expression::Binary {
                        left: Box::new(Expression::Constant(1.0)),
                        right: Box::new(Expression::Constant(2.0)),
                        op: BinaryOperation::Times,
                    }),
                    op: BinaryOperation::Minus,
                },
                "3 - 1*2",
            ),
        ];

        for (expr, should_be) in inputs {
            let got = expr.to_string();
            assert_eq!(got, should_be);
        }
    }
}
