use constraints::{ops::Builtins, SystemOfEquations};
use std::io::{BufRead, BufReader, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = SystemOfEquations::new();
    let stdin = std::io::stdin();
    let mut lines = BufReader::new(stdin.lock()).lines();

    loop {
        let mut stdout = std::io::stdout();
        write!(stdout, "> ")?;
        stdout.flush()?;

        match lines.next() {
            Some(Ok(line)) => match line.parse() {
                Ok(equation) => system.push(equation),
                Err(e) => eprintln!("Unable to parse \"{}\": {}", line, e),
            },
            Some(Err(e)) => return Err(Box::new(e)),
            None => break,
        }
    }

    println!();

    let unknowns: Vec<_> =
        system.unknowns().iter().map(ToString::to_string).collect();
    println!("Solving for {}", unknowns.join(", "));

    let ctx = Builtins::default();
    let solution = system.solve(&ctx)?;

    println!("Found:");

    for (name, value) in &solution.known_values {
        println!("  {} = {}", name, value);
    }

    Ok(())
}
