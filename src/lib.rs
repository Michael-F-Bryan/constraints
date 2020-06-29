//! The symbolic algebra system.

#[cfg(test)]
#[macro_use]
extern crate pretty_assertions;

mod equations;
mod expr;
pub mod ops;
mod parse;

pub use equations::{Equation, SystemOfEquations};
pub use expr::{BinaryOperation, Expression, Parameter};
pub use parse::{parse, ParseError, TokenKind};
