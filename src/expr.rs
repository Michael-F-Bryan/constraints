use crate::parse::ParseError;
use smol_str::SmolStr;
use std::{
    fmt::{self, Display, Formatter},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
    str::FromStr,
};

// PERF: Switch from Rc<Expression> to Arc<Expression> and use Arc::make_mut()
// to get efficient copy-on-write semantics

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// A free variable (e.g. `x`).
    Parameter(Parameter),
    /// A fixed constant (e.g. `3.14`).
    Constant(f64),
    /// An expression involving two operands.
    Binary {
        /// The left operand.
        left: Rc<Expression>,
        /// The right operand.
        right: Rc<Expression>,
        /// The binary operation being executed.
        op: BinaryOperation,
    },
    /// Negate the expression.
    Negate(Rc<Expression>),
    /// Invoke a builtin function.
    FunctionCall {
        /// The name of the function being called.
        name: SmolStr,
        /// The argument passed to this function call.
        argument: Rc<Expression>,
    },
}

impl Expression {
    /// Iterate over all [`Expression`]s in this [`Expression`] tree.
    pub fn iter(&self) -> impl Iterator<Item = &Expression> + '_ {
        Iter {
            to_visit: vec![self],
        }
    }

    /// Iterate over all [`Parameter`]s mentioned in this [`Expression`].
    pub fn params(&self) -> impl Iterator<Item = &Parameter> + '_ {
        self.iter().filter_map(|expr| match expr {
            Expression::Parameter(p) => Some(p),
            _ => None,
        })
    }

    /// Does this [`Expression`] involve a particular [`Parameter`]?
    pub fn depends_on(&self, param: &Parameter) -> bool {
        self.params().any(|p| p == param)
    }

    /// Is this a [`Expression::Constant`] expression?
    pub fn is_constant(&self) -> bool {
        match self {
            Expression::Constant(_) => true,
            _ => false,
        }
    }

    pub fn functions(&self) -> impl Iterator<Item = &str> + '_ {
        self.iter().filter_map(|expr| match expr {
            Expression::FunctionCall { name, .. } => Some(name.as_ref()),
            _ => None,
        })
    }
}

/// A depth-first iterator over the sub-[`Expression`]s in an [`Expression`].
#[derive(Debug)]
struct Iter<'expr> {
    to_visit: Vec<&'expr Expression>,
}

impl<'expr> Iterator for Iter<'expr> {
    type Item = &'expr Expression;

    fn next(&mut self) -> Option<Self::Item> {
        let next_item = self.to_visit.pop()?;

        match next_item {
            Expression::Binary { left, right, .. } => {
                self.to_visit.push(right);
                self.to_visit.push(left);
            },
            Expression::Negate(inner) => self.to_visit.push(inner),
            Expression::FunctionCall { argument, .. } => {
                self.to_visit.push(argument)
            },
            _ => {},
        }

        Some(next_item)
    }
}

/// A free variable.
#[derive(Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub enum Parameter {
    /// A variable with associated name.
    Named(SmolStr),
    /// An anonymous variable generated by the system.
    Anonymous { number: usize },
}

impl Parameter {
    /// Create a new [`Parameter::Named`] parameter, automatically converting
    /// the `name` to a `SmolStr`.
    pub fn named<S>(name: S) -> Self
    where
        S: Into<SmolStr>,
    {
        Parameter::Named(name.into())
    }
}

impl Display for Parameter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Parameter::Named(name) => write!(f, "{}", name),
            Parameter::Anonymous { number } => write!(f, "${}", number),
        }
    }
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
            left: Rc::new(self),
            right: Rc::new(rhs),
            op: BinaryOperation::Plus,
        }
    }
}

impl Sub for Expression {
    type Output = Expression;

    fn sub(self, rhs: Expression) -> Expression {
        Expression::Binary {
            left: Rc::new(self),
            right: Rc::new(rhs),
            op: BinaryOperation::Minus,
        }
    }
}

impl Mul for Expression {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Expression {
        Expression::Binary {
            left: Rc::new(self),
            right: Rc::new(rhs),
            op: BinaryOperation::Times,
        }
    }
}

impl Div for Expression {
    type Output = Expression;

    fn div(self, rhs: Expression) -> Expression {
        Expression::Binary {
            left: Rc::new(self),
            right: Rc::new(rhs),
            op: BinaryOperation::Divide,
        }
    }
}

impl Neg for Expression {
    type Output = Expression;

    fn neg(self) -> Self::Output { Expression::Negate(Rc::new(self)) }
}

impl FromStr for Expression {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> { crate::parse(s) }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Parameter(p) => write!(f, "{}", p),
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
            Expression::FunctionCall { name, argument } => {
                write!(f, "{}({})", name, argument)
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
    fn pretty_printing_works_similarly_to_a_human() {
        let inputs = vec![
            (Expression::Constant(3.0), "3"),
            (
                Expression::FunctionCall {
                    name: "sin".into(),
                    argument: Rc::new(Expression::Constant(5.0)),
                },
                "sin(5)",
            ),
            (Expression::Negate(Rc::new(Expression::Constant(5.0))), "-5"),
            (
                Expression::Negate(Rc::new(Expression::FunctionCall {
                    name: "sin".into(),
                    argument: Rc::new(Expression::Constant(5.0)),
                })),
                "-sin(5)",
            ),
            (
                Expression::Binary {
                    left: Rc::new(Expression::Constant(1.0)),
                    right: Rc::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Plus,
                },
                "1 + 1",
            ),
            (
                Expression::Binary {
                    left: Rc::new(Expression::Constant(1.0)),
                    right: Rc::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Minus,
                },
                "1 - 1",
            ),
            (
                Expression::Binary {
                    left: Rc::new(Expression::Constant(1.0)),
                    right: Rc::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Times,
                },
                "1*1",
            ),
            (
                Expression::Binary {
                    left: Rc::new(Expression::Constant(1.0)),
                    right: Rc::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Divide,
                },
                "1/1",
            ),
            (
                Expression::Negate(Rc::new(Expression::Binary {
                    left: Rc::new(Expression::Constant(1.0)),
                    right: Rc::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Plus,
                })),
                "-(1 + 1)",
            ),
            (
                Expression::Negate(Rc::new(Expression::Binary {
                    left: Rc::new(Expression::Constant(1.0)),
                    right: Rc::new(Expression::Constant(1.0)),
                    op: BinaryOperation::Times,
                })),
                "-1*1",
            ),
            (
                Expression::Binary {
                    left: Rc::new(Expression::Binary {
                        left: Rc::new(Expression::Constant(1.0)),
                        right: Rc::new(Expression::Constant(2.0)),
                        op: BinaryOperation::Plus,
                    }),
                    right: Rc::new(Expression::Constant(3.0)),
                    op: BinaryOperation::Divide,
                },
                "(1 + 2)/3",
            ),
            (
                Expression::Binary {
                    left: Rc::new(Expression::Constant(3.0)),
                    right: Rc::new(Expression::Binary {
                        left: Rc::new(Expression::Constant(1.0)),
                        right: Rc::new(Expression::Constant(2.0)),
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

    #[test]
    fn iterate_over_parameters_in_an_expression() {
        let expr: Expression =
            "a + sin(5*(b + (c - d)) / -e) - a * f".parse().unwrap();
        let a = Parameter::named("a");
        let b = Parameter::named("b");
        let c = Parameter::named("c");
        let d = Parameter::named("d");
        let e = Parameter::named("e");
        let f = Parameter::named("f");

        let got: Vec<_> = expr.params().collect();

        assert_eq!(got, &[&a, &b, &c, &d, &e, &a, &f]);
    }
}
