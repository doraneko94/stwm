use stwm::rate::*;

fn main() {
    let mut model = RateModel::new(0.013, -2.3, 0.8, 4.0, 0.0, 0.2, 1.5, 0.001);
    model.set_save(5.0);
    model.run(5.0);
    model.view("fig1a.plt", "Fig 1A", "Time [ms]", "Network activity [Hz]", 0.0, 5000.0, 0.0, 60.0);
}