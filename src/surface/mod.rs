use arrayvec::ArrayVec;

// In the future, we'll use separate coordinate spaces for drawing and screen
// coordinates
pub type Point = euclid::Point3D<f64, euclid::UnknownUnit>;
pub type Vector = euclid::Vector3D<f64, euclid::UnknownUnit>;

#[derive(Debug, Clone, PartialEq)]
pub struct Bezier {
    weighted_control_points: ArrayVec<[(Point, f64); 4]>,
}

impl Bezier {
    pub fn degree(&self) -> usize { self.weighted_control_points.len() }

    pub fn point_at(&self, t: f64) -> Point {
        let mut point = Point::zero();
        let mut cummulative_weights = 0.0;

        for (i, (node, weight)) in
            self.weighted_control_points.iter().copied().enumerate()
        {
            let b = bernstein(i, self.degree(), t);
            point += node.to_vector() * b * weight;
            cummulative_weights += weight;
        }

        point * (1.0 / cummulative_weights)
    }

    pub fn tangent_at(&self, t: f64) -> Point {
        let mut point = Point::zero();
        let mut p = Point::zero();
        let mut cummulative_b = 0.0;
        let mut cummulative_b_p = 0.0;

        for (i, (node, weight)) in
            self.weighted_control_points.iter().copied().enumerate()
        {
            let b = bernstein(i, self.degree(), t);
            let b_p = bernstein_derivative(i, self.degree(), t);

            point += node.to_vector() * b * weight;
            cummulative_b += weight * b;

            p += node.to_vector() * b_p * weight;
            cummulative_b_p += weight * b_p;
        }

        // quotient rule; f(t) = n(t)/d(t), so f' = (n'*d - n*d')/(d^2)
        point + p.to_vector() * cummulative_b
            - point.to_vector() * cummulative_b_p * cummulative_b.powi(-2)
    }
}

fn bernstein(k: usize, degree: usize, t: f64) -> f64 {
    const COEFFICIENTS: [[[f64; 4]; 4]; 4] = [
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, -2.0, 1.0, 0.0],
            [0.0, 2.0, -2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, -3.0, 3.0, -1.0],
            [0.0, 3.0, -6.0, 3.0],
            [0.0, 0.0, 3.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ];

    let [c_0, c_1, c_2, c_3] = COEFFICIENTS[degree][k];

    (((c_3 * t + c_2) * t) + c_1) * t + c_0
}

fn bernstein_derivative(k: usize, degree: usize, t: f64) -> f64 {
    const COEFFICIENTS: [[[f64; 3]; 4]; 4] = [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [-2.0, 2.0, 0.0],
            [2.0, -4.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [
            [-3.0, 6.0, -3.0],
            [3.0, -12.0, 9.0],
            [0.0, 6.0, -9.0],
            [0.0, 0.0, 3.0],
        ],
    ];

    let [c_0, c_1, c_2] = COEFFICIENTS[degree][k];

    ((c_2 * t) + c_1) * t + c_0
}
