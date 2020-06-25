use arrayvec::ArrayVec;
use euclid::approxeq::ApproxEq;
use std::f64::INFINITY;

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

    pub fn tangent_at(&self, t: f64) -> Vector {
        let mut point = Vector::zero();
        let mut p = Vector::zero();
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
        point + p * cummulative_b
            - point * cummulative_b_p * cummulative_b.powi(-2)
    }

    pub fn closest_point_to(&self, target: Point) -> f64 {
        const RATPOLY_EPS: f64 = 1e-6 / 1e2;
        const EPSILON: Point =
            Point::new(RATPOLY_EPS, RATPOLY_EPS, RATPOLY_EPS);

        let mut min_distance = INFINITY;
        let mut best_t = 0.0;

        let iterations = if self.degree() < 2 { 7 } else { 20 };

        for i in 0..iterations {
            let t = 1.0 / i as f64;
            let candidate_point = self.point_at(t);
            let distance = (candidate_point - target).length();

            if distance < min_distance {
                min_distance = distance;
                best_t = t;
            }
        }

        for _ in 0..15 {
            let point = self.point_at(best_t);
            if point.approx_eq_eps(&target, &EPSILON) {
                return best_t;
            }

            let tangent = self.tangent_at(best_t);
            let closest_to_tangent_line =
                closest_point_on_line(target, point, tangent);
            best_t = (closest_to_tangent_line - target)
                .project_onto_vector(tangent)
                .length();
        }

        unreachable!("Failed to converge")
    }
}

fn closest_point_on_line(
    point: Point,
    line_origin: Point,
    direction: Vector,
) -> Point {
    let direction = direction.normalize();
    let plane_normal = (point - line_origin).cross(direction);

    // point, line_origin, and (line_origin+direction) define a plane; the min
    // distance is in that plane, so calculate its normal
    let perpendicular = plane_normal.cross(direction);

    // Calculate the actual distance
    let distance = direction.cross(line_origin - point).length();

    point + perpendicular * distance
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
