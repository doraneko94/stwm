use crate::neuron::*;
use rayon::prelude::*;
use rand::distributions::{Normal, Distribution};

pub const F: f64 = 0.10;
pub const P: usize = 5;
pub const C: f64 = 0.2;
pub const N_E: usize = 8000;
pub const N_I: usize = 2000;
pub const D_L: usize = 1;
pub const D_H: usize = 6;
pub const J_P: f64 = 0.45;
pub const DT: f64 = 0.001;

pub const N: usize = N_E + N_I;
pub const MEM_SIZE: usize = ((N_E as f64) * F) as usize;

const MU_EXT_E: f64 = 23.10;
const SIGMA_EXT_E: f64 = 1.0;
const MU_EXT_I: f64 = 21.0;
const SIGMA_EXT_I: f64 = 1.0;

pub struct SpikeNetwork {
    pub step: usize,
    pub t: usize,
    pub v_s: Vec<Vec<u8>>,
    pub v_x: Vec<Vec<f64>>,
    pub v_u: Vec<Vec<f64>>,
    pub v_e: Vec<ExtNeuron>,
    pub v_i: Vec<InhNeuron>,
    pub stim_e: Vec<f64>,
}

impl SpikeNetwork {
    pub fn new(sec: f64) -> Self {
        let step = (sec / DT) as usize;
        let t = 0;
        let v_s = vec![Vec::with_capacity(step); N];
        let v_x = vec![Vec::with_capacity(step); N];
        let v_u = vec![Vec::with_capacity(step); N];
        println!("constructing exitatory neurons.");
        let v_e = (0..N_E).map(|i| ExtNeuron::new(i)).collect::<Vec<ExtNeuron>>();
        println!("constructing inhibitory neurons.");
        let v_i = (0..N_I).map(|i| InhNeuron::new(i)).collect::<Vec<InhNeuron>>();
        let stim_e = vec![MU_EXT_E; N_E];

        SpikeNetwork { step, t, v_s, v_x, v_u, v_e, v_i, stim_e }
    }

    pub fn run(&mut self) {
        let t = self.t;
        let v_s = &self.v_s;
        let v_x = &self.v_x;
        let v_u = &self.v_u;
        let stim_e = &self.stim_e;
        
        let norm = Normal::new(0.0, SIGMA_EXT_E);
        let out_e = self.v_e.par_iter_mut().enumerate().map(|(i, neuron)| {
            let i_ext = norm.sample(&mut rand::thread_rng()) + stim_e[i];
            neuron.run(t, i_ext, v_s, v_x, v_u)
        }).collect::<Vec<(u8, f64, f64)>>();
        let norm = Normal::new(MU_EXT_I, SIGMA_EXT_I);
        let out_i = self.v_i.par_iter_mut().map(|neuron| {
            let i_ext = norm.sample(&mut rand::thread_rng());
            neuron.run(t, i_ext, v_s, v_x, v_u)
        }).collect::<Vec<(u8, f64, f64)>>();

        for i in 0..N_E {
            self.v_s[i].push(out_e[i].0);
            self.v_x[i].push(out_e[i].1);
            self.v_u[i].push(out_e[i].2);
        }

        for i in 0..N_I {
            self.v_s[N_E+i].push(out_i[i].0);
            self.v_x[N_E+i].push(out_i[i].1);
            self.v_u[N_E+i].push(out_i[i].2);
        }

        self.t += 1;
    }

    pub fn run_sec(&mut self, sec: f64) {
        let time = ( sec / DT ) as usize;
        for _ in 0..time {
            self.run();
        }
    }

    pub fn set_stim(&mut self, stim: f64, range: (usize, usize)) {
        let (start, end) = range;
        for i in start..end {
            self.stim_e[i] = stim;
        }
    }

    pub fn engram(&self, id: usize) -> (usize, usize) {
        let start = MEM_SIZE * id;
        (start, start + MEM_SIZE)
    }
}