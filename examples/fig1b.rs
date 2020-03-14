use stwm::rate::*;

fn main() {
    let mut model = RateModel::new(0.013, -2.3, 0.3, 4.0, 1.5, 0.2, 1.5, 0.001);
    model.run(2.0);
    model.e0 = -1.0;
    model.run(0.3);
    model.e0 = -2.3;
    model.run(2.7);
    model.view("fig1b.plt", "Fig 1B", "Time [ms]", "Network activity [Hz]", 0.0, 5000.0, 0.0, 80.0);
}