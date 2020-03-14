pub fn rk4<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    let k1 = f(x);
    let k2 = f(x + h * k1 / 2.0);
    let k3 = f(x + h * k2 / 2.0);
    let k4 = f(x + h * k3);

    h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
}