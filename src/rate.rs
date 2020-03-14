use gnuplot::AxesCommon;
use gnuplot::*;
use crate::solve::*;

pub struct RateModel {
    tau: f64,
    e: f64,
    pub e0: f64,
    u: f64,
    u0: f64,
    x: f64,
    j: f64,
    tau_f: f64,
    tau_d: f64,
    alpha: f64,
    dt: f64,
    save: Vec<f64>,
}

impl RateModel {
    pub fn new(tau: f64, e: f64, u: f64, j: f64,
           tau_f: f64, tau_d: f64, alpha: f64, dt: f64) 
    -> Self 
    {
        let e0 = e;
        let u0 = u;
        let x = 1.0;
        let save = Vec::new();
        RateModel { tau, e, e0, u, u0, x, j, tau_f, tau_d, alpha, dt, save }
    }

    fn g(&self, z: f64) -> f64 {
        self.alpha * ( 1.0 + ( z / self.alpha ).exp() ).log(std::f64::consts::E)
    }

    fn dedt(&self, e: f64) -> f64 {
        ( - e + self.g( self.j * self.u * self.x * e + self.e0 ) ) / self.tau
    }

    fn dudt(&self, u: f64) -> f64 {
        if self.tau_f == 0.0 {
            0.0
        } else {
            ( self.u0 - u ) / self.tau_f + self.u0 * ( 1.0 - u ) * self.e
        }
    }

    fn dxdt(&self, x: f64) -> f64 {
        ( 1.0 - x ) / self.tau_d - self.u * x * self.e
    }

    pub fn euler(&mut self) {
        self.e += self.dt * self.dedt(self.e);
        self.u += self.dt * self.dudt(self.u);
        self.x += self.dt * self.dxdt(self.x);
    }

    pub fn rk4(&mut self) {
        self.e += rk4(|e: f64| self.dedt(e), self.e, self.dt);
        self.u += rk4(|u: f64| self.dudt(u), self.u, self.dt);
        self.x += rk4(|x: f64| self.dxdt(x), self.x, self.dt);
    }

    pub fn run(&mut self, sec: f64) {
        let step = (sec / self.dt) as usize;
        for _ in 0..step {
            self.rk4();
            self.save.push(self.e);
        }
    }

    pub fn set_save(&mut self, sec: f64) {
        let step = (sec / self.dt) as usize;
        self.save = Vec::with_capacity(step);
    }

    pub fn view(&self, filename: &str, title: &str, xlabel: &str, ylabel: &str,
                xmin: f64, xmax: f64, ymin: f64, ymax: f64)
    {
        let mut fg = gnuplot::Figure::new();
        let l = self.save.len();
        let x = (0..l).collect::<Vec<usize>>();
        fg.axes2d()
            .lines(x.iter(), self.save.iter(), &[gnuplot::Color("blue")])
            .set_x_label(xlabel, &[])
            .set_y_label(ylabel, &[])
            .set_title(title, &[])
            .set_x_range(Fix(xmin), Fix(xmax))
            .set_y_range(Fix(ymin), Fix(ymax));
        fg.echo_to_file(filename);
    }
}