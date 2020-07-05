use crate::{
    ops::{self, Context, EvaluationError},
    solve::{Solution, SolveError},
    Expression, Parameter, ParseError,
};
use nalgebra::DVector as Vector;
use std::{
    fmt::Debug,
    iter::{Extend, FromIterator},
    str::FromStr,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Equation {
    pub(crate) body: Expression,
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

/// A builder for constructing a system of equations and solving them.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SystemOfEquations {
    pub(crate) equations: Vec<Equation>,
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
        let unknowns = self.unknowns();
        crate::solve::solve(&self.equations, &unknowns, &self, ctx)
    }

    pub fn unknowns(&self) -> Vec<Parameter> {
        let mut unknowns: Vec<_> = self
            .equations
            .iter()
            .flat_map(|eq| eq.body.params())
            .cloned()
            .collect();
        unknowns.sort();
        unknowns.dedup();

        unknowns
    }

    pub fn num_unknowns(&self) -> usize { self.unknowns().len() }

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

    pub(crate) fn evaluate<F, C>(
        &self,
        lookup_parameter_value: F,
        ctx: &C,
    ) -> Result<Vector<f64>, EvaluationError>
    where
        F: Fn(&Parameter) -> Option<f64>,
        C: Context,
    {
        let mut values = Vec::new();

        for equation in &self.equations {
            values.push(ops::evaluate(
                &equation.body,
                &lookup_parameter_value,
                ctx,
            )?);
        }

        Ok(Vector::from_vec(values))
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

impl<'a> IntoIterator for &'a SystemOfEquations {
    type IntoIter = <&'a [Equation] as IntoIterator>::IntoIter;
    type Item = &'a Equation;

    fn into_iter(self) -> Self::IntoIter { self.equations.iter() }
}

impl IntoIterator for SystemOfEquations {
    type IntoIter = <Vec<Equation> as IntoIterator>::IntoIter;
    type Item = Equation;

    fn into_iter(self) -> Self::IntoIter { self.equations.into_iter() }
}
