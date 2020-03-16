use stwm::network::{SpikeNetwork};
use gnuplot::AxesCommon;
use gnuplot::*;

fn main() {
    let mut network = SpikeNetwork::new(13.0);
    //network.set_stim(23.80, (0, 8000));
    network.run_sec(10.0);
    network.set_stim(23.10 * 1.15, network.engram(0));
    network.run_sec(0.35);
    network.set_stim(23.10, network.engram(0));
    network.run_sec(0.85);
    network.set_stim(23.10 * 1.05, network.engram(0));
    network.run_sec(0.25);
    network.set_stim(23.10, network.engram(0));
    network.run_sec(1.55);

    let mut fg = gnuplot::Figure::new();
    let l = network.v_s[0].len();
    let x = (0..l).collect::<Vec<usize>>();
    let (_, end) = network.engram(0);
    
    let mut y = vec![0.0; l];
    for i in 0..l {
        for j in 0..end {
            y[i] += network.v_u[j][i];
        }
        y[i] /= end as f64;
    }
    
    fg.axes2d()
        .lines(x.iter(), y.iter(), &[gnuplot::Color("blue")])
        .set_x_label("Time [ms]", &[])
        .set_y_label("x, u", &[])
        .set_title("test", &[])
        .set_x_range(Fix(9000.0), Fix(13000.0))
        .set_y_range(Fix(0.0), Fix(1.0));
    fg.echo_to_file("test.plt");
    
    println!("{:?}", network.v_s[0].iter().map(|&i| i as usize).sum::<usize>());
}