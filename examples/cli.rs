use constraints::{ops::Builtins, SystemOfEquations};
use std::io::{BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut system = SystemOfEquations::new();
    let stdin = std::io::stdin();

    for line in BufReader::new(stdin.lock()).lines() {
        let line = line?;
        match line.parse() {
            Ok(equation) => system.push(equation),
            Err(e) => eprintln!("Unable to parse \"{}\": {}", line, e),
        }
    }

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
