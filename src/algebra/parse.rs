use crate::algebra::{BinaryOperation, Expression, Parameter};
use std::{iter::Peekable, ops::Range};

/// Parse an [`Expression`] tree from some text.
pub fn parse(s: &str) -> Result<Expression, ParseError> {
    Parser::new(s).parse()
}

/// A simple recursive descent parser (`LL(1)`) for converting a string into an
/// expression tree.
///
/// The grammar:
///
/// ```text
/// expression     := term "+" expression
///                 | term "-" expression
///                 | term
///
/// term           := "-" term
///                 | factor "*" term
///                 | factor "/" term
///                 | factor
///
/// factor         := variable_or_function_call
///                 | "(" expression ")"
///                 | NUMBER
///
/// variable_or_function_call = IDENTIFIER "(" expression ")"
///                           | IDENTIFIER
/// ```
#[derive(Debug, Clone)]
pub(crate) struct Parser<'a> {
    tokens: Peekable<Tokens<'a>>,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(src: &'a str) -> Self {
        Parser {
            tokens: Tokens::new(src).peekable(),
        }
    }

    pub(crate) fn parse(mut self) -> Result<Expression, ParseError> {
        let expr = self.expression()?;

        match self.tokens.next() {
            None => Ok(expr),
            Some(Ok(token)) => {
                panic!("Not all tokens consumed! Found {:?}", token)
            },
            Some(Err(e)) => Err(e),
        }
    }

    fn peek(&mut self) -> Option<TokenKind> {
        self.tokens
            .peek()
            .and_then(|result| result.as_ref().ok())
            .map(|tok| tok.kind)
    }

    fn advance(&mut self) -> Result<Token<'a>, ParseError> {
        match self.tokens.next() {
            Some(result) => result,
            None => Err(ParseError::UnexpectedEndOfInput),
        }
    }

    fn expression(&mut self) -> Result<Expression, ParseError> {
        let left = self.term()?;

        self.then_right_part_of_binary_op(
            left,
            &[TokenKind::Plus, TokenKind::Minus],
            |p| p.expression(),
        )
    }

    fn term(&mut self) -> Result<Expression, ParseError> {
        let left = self.factor()?;

        self.then_right_part_of_binary_op(
            left,
            &[TokenKind::Times, TokenKind::Divide],
            |p| p.term(),
        )
    }

    fn then_right_part_of_binary_op<F>(
        &mut self,
        left: Expression,
        expected: &[TokenKind],
        then: F,
    ) -> Result<Expression, ParseError>
    where
        F: FnOnce(&mut Parser<'_>) -> Result<Expression, ParseError>,
    {
        if let Some(kind) = self.peek() {
            for candidate in expected {
                if *candidate == kind {
                    // skip past the operator
                    let _ = self.advance()?;
                    // and parse the second bit
                    let right = then(self)?;

                    return Ok(Expression::Binary {
                        left: Box::new(left),
                        right: Box::new(right),
                        op: candidate.as_binary_op(),
                    });
                }
            }
        }

        Ok(left)
    }

    fn factor(&mut self) -> Result<Expression, ParseError> {
        let expected =
            &[TokenKind::Number, TokenKind::Identifier, TokenKind::Minus];

        match self.peek() {
            Some(TokenKind::Number) => {
                return self.number();
            },
            Some(TokenKind::Minus) => {
                let _ = self.advance()?;
                let operand = self.factor()?;
                return Ok(Expression::Negate(Box::new(operand)));
            },
            Some(TokenKind::Identifier) => {
                return self.variable_or_function_call()
            },
            Some(TokenKind::OpenParen) => {
                let _ = self.advance()?;
                let expr = self.expression()?;
                let close_paren = self.advance()?;

                if close_paren.kind == TokenKind::CloseParen {
                    return Ok(expr);
                } else {
                    return Err(ParseError::UnexpectedToken {
                        found: close_paren.kind,
                        span: close_paren.span,
                        expected: &[TokenKind::CloseParen],
                    });
                }
            },
            _ => {},
        }

        // we couldn't parse the factor, return a nice error
        match self.tokens.next() {
            Some(Ok(Token { span, kind, .. })) => {
                Err(ParseError::UnexpectedToken {
                    found: kind,
                    expected,
                    span,
                })
            },
            Some(Err(e)) => Err(e),
            None => Err(ParseError::UnexpectedEndOfInput),
        }
    }

    fn variable_or_function_call(&mut self) -> Result<Expression, ParseError> {
        let ident = self.advance()?;
        debug_assert_eq!(ident.kind, TokenKind::Identifier);

        if self.peek() == Some(TokenKind::OpenParen) {
            self.function_call(ident)
        } else {
            Ok(Expression::Parameter(Parameter::named(ident.text)))
        }
    }

    fn function_call(
        &mut self,
        identifier: Token<'a>,
    ) -> Result<Expression, ParseError> {
        let open_paren = self.advance()?;
        debug_assert_eq!(open_paren.kind, TokenKind::OpenParen);

        let argument = self.expression()?;

        let Token { kind, span, .. } = self.advance()?;

        if kind == TokenKind::CloseParen {
            Ok(Expression::FunctionCall {
                function: identifier.text.into(),
                argument: Box::new(argument),
            })
        } else {
            Err(ParseError::UnexpectedToken {
                found: kind,
                span,
                expected: &[TokenKind::CloseParen],
            })
        }
    }

    fn number(&mut self) -> Result<Expression, ParseError> {
        let token = self
            .tokens
            .next()
            .ok_or(ParseError::UnexpectedEndOfInput)??;

        debug_assert_eq!(token.kind, TokenKind::Number);
        let number =
            token.text.parse().expect("Guaranteed correct by the lexer");

        Ok(Expression::Constant(number))
    }
}

/// Possible errors that may occur while parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    InvalidCharacter {
        character: char,
        index: usize,
    },
    UnexpectedEndOfInput,
    UnexpectedToken {
        found: TokenKind,
        span: Range<usize>,
        expected: &'static [TokenKind],
    },
}

#[derive(Debug, Clone, PartialEq)]
struct Tokens<'a> {
    src: &'a str,
    cursor: usize,
}

impl<'a> Tokens<'a> {
    fn new(src: &'a str) -> Self { Tokens { src, cursor: 0 } }

    fn rest(&self) -> &'a str { &self.src[self.cursor..] }

    fn peek(&self) -> Option<char> { self.rest().chars().next() }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.cursor += c.len_utf8();
        Some(c)
    }

    fn chomp(
        &mut self,
        kind: TokenKind,
    ) -> Option<Result<Token<'a>, ParseError>> {
        let start = self.cursor;
        self.advance()?;
        let end = self.cursor;

        let tok = Token {
            text: &self.src[start..end],
            span: start..end,
            kind,
        };

        Some(Ok(tok))
    }

    fn take_while<P>(
        &mut self,
        mut predicate: P,
    ) -> Option<(&'a str, Range<usize>)>
    where
        P: FnMut(char) -> bool,
    {
        let start = self.cursor;

        while let Some(c) = self.peek() {
            if !predicate(c) {
                break;
            }

            self.advance();
        }

        let end = self.cursor;

        if start != end {
            let text = &self.src[start..end];
            Some((text, start..end))
        } else {
            None
        }
    }

    fn chomp_integer(&mut self) -> &'a str {
        let (text, _) = self.take_while(|c| c.is_ascii_digit()).unwrap();
        text
    }

    fn chomp_number(&mut self) -> Token<'a> {
        let start = self.cursor;
        self.chomp_integer();

        if self.peek() == Some('.') {
            // skip past the decimal
            self.advance();

            let digits_to_go =
                self.peek().map(|c| c.is_ascii_digit()).unwrap_or(false);
            if digits_to_go {
                self.chomp_integer();
            }
        }

        let end = self.cursor;

        Token::from_text(self.src, start..end, TokenKind::Number)
    }

    fn chomp_identifier(&mut self) -> Token<'a> {
        let mut seen_first_character = false;

        let (_, span) = self
            .take_while(|c| {
                if seen_first_character {
                    c.is_alphanumeric() || c == '_'
                } else {
                    seen_first_character = true;
                    c.is_alphabetic() || c == '_'
                }
            })
            .expect("We know there should be at least 1 character");

        Token::from_text(self.src, span, TokenKind::Identifier)
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Result<Token<'a>, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            return match self.peek()? {
                space if space.is_whitespace() => {
                    self.advance();
                    continue;
                },
                '(' => self.chomp(TokenKind::OpenParen),
                ')' => self.chomp(TokenKind::CloseParen),
                '+' => self.chomp(TokenKind::Plus),
                '-' => self.chomp(TokenKind::Minus),
                '*' => self.chomp(TokenKind::Times),
                '/' => self.chomp(TokenKind::Divide),
                '_' | 'a'..='z' | 'A'..='Z' => {
                    Some(Ok(self.chomp_identifier()))
                },
                '0'..='9' => Some(Ok(self.chomp_number())),
                other => Some(Err(ParseError::InvalidCharacter {
                    character: other,
                    index: self.cursor,
                })),
            };
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Token<'a> {
    text: &'a str,
    span: Range<usize>,
    kind: TokenKind,
}

impl<'a> Token<'a> {
    fn from_text(
        original_source: &'a str,
        span: Range<usize>,
        kind: TokenKind,
    ) -> Self {
        Token {
            text: &original_source[span.clone()],
            span,
            kind,
        }
    }
}

/// The kinds of token that can appear in an [`Expression`]'s text form.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum TokenKind {
    Identifier,
    Number,
    OpenParen,
    CloseParen,
    Plus,
    Minus,
    Times,
    Divide,
}

impl TokenKind {
    fn as_binary_op(self) -> BinaryOperation {
        match self {
            TokenKind::Plus => BinaryOperation::Plus,
            TokenKind::Minus => BinaryOperation::Minus,
            TokenKind::Times => BinaryOperation::Times,
            TokenKind::Divide => BinaryOperation::Divide,
            other => unreachable!("{:?} is not a binary op", other),
        }
    }
}

#[cfg(test)]
mod tokenizer_tests {
    use super::*;

    macro_rules! tokenize_test {
        ($name:ident, $src:expr, $should_be:expr) => {
            #[test]
            fn $name() {
                let mut tokens = Tokens::new($src);

                let got = tokens.next().unwrap().unwrap();

                let Range { start, end } = got.span;
                assert_eq!(start, 0);
                assert_eq!(end, $src.len());
                assert_eq!(got.kind, $should_be);

                assert!(
                    tokens.next().is_none(),
                    "{:?} should be empty",
                    tokens
                );
            }
        };
    }

    tokenize_test!(open_paren, "(", TokenKind::OpenParen);
    tokenize_test!(close_paren, ")", TokenKind::CloseParen);
    tokenize_test!(plus, "+", TokenKind::Plus);
    tokenize_test!(minus, "-", TokenKind::Minus);
    tokenize_test!(times, "*", TokenKind::Times);
    tokenize_test!(divide, "/", TokenKind::Divide);
    tokenize_test!(single_digit_integer, "3", TokenKind::Number);
    tokenize_test!(multi_digit_integer, "31", TokenKind::Number);
    tokenize_test!(number_with_trailing_dot, "31.", TokenKind::Number);
    tokenize_test!(simple_decimal, "3.14", TokenKind::Number);
    tokenize_test!(simple_identifier, "x", TokenKind::Identifier);
    tokenize_test!(longer_identifier, "hello", TokenKind::Identifier);
    tokenize_test!(
        identifiers_can_have_underscores,
        "hello_world",
        TokenKind::Identifier
    );
    tokenize_test!(
        identifiers_can_start_with_underscores,
        "_hello_world",
        TokenKind::Identifier
    );
    tokenize_test!(
        identifiers_can_contain_numbers,
        "var5",
        TokenKind::Identifier
    );
}

#[cfg(test)]
mod parser_tests {
    use super::*;

    macro_rules! parser_test {
        ($name:ident, $src:expr) => {
            parser_test!($name, $src, $src);
        };
        ($name:ident, $src:expr, $should_be:expr) => {
            #[test]
            fn $name() {
                let got = Parser::new($src).parse().unwrap();

                let round_tripped = got.to_string();
                assert_eq!(round_tripped, $should_be);
            }
        };
    }

    parser_test!(simple_integer, "1");
    parser_test!(one_plus_one, "1 + 1");
    parser_test!(one_plus_one_plus_negative_one, "1 + -1");
    parser_test!(one_plus_one_times_three, "1 + 1*3");
    parser_test!(one_plus_one_all_times_three, "(1 + 1)*3");
    parser_test!(negative_one, "-1");
    parser_test!(negative_one_plus_one, "-1 + 1");
    parser_test!(negative_one_plus_x, "-1 + x");
    parser_test!(number_in_parens, "(1)", "1");
    parser_test!(bimdas, "1*2 + 3*4/(5 - 2)*1 - 3");
    parser_test!(function_call, "sin(1)", "sin(1)");
    parser_test!(function_call_with_expression, "sin(1/0)");
    parser_test!(
        function_calls_function_calls_function_with_variable,
        "foo(bar(baz(pi)))"
    );
}
