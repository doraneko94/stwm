use rand::distributions::{Distribution, Uniform};
use crate::network::{P, C, N_E, N_I, D_L, D_H, J_P, DT, N, MEM_SIZE};
use crate::solve::*;

const J_IE: f64 = 0.135;// * 10.0;
const J_EI: f64 = 0.25;// * 10.0;
const J_II: f64 = 0.20;// * 10.0;
const J_B:  f64 = 0.10;// * 10.0;
// const J_P:  f64 = 0.45;
const GAMMA_0: f64 = 0.10;
// const DELTA: f64 = 0.001;

const U: f64 = 0.2;
const TAU_F: f64 = 1.5;
const TAU_D: f64 = 0.2;

const N_NSP: usize = N_E - ((N_E * P) as f64 * C) as usize;
const N_NSP_P: usize = ((N_NSP as f64) * GAMMA_0) as usize;
const N_CH: usize = ((MEM_SIZE as f64) * C) as usize;
const N_DP: usize = N_CH + N_NSP_P;
const N_DB: usize = N_CH * ( P - 1 ) + N_NSP - N_NSP_P;
const N_DI: usize = ((N_I as f64) * C) as usize;

const MEM_P: usize = ((MEM_SIZE as f64) * GAMMA_0) as usize;

pub trait Neuron {
    const THETA: f64;
    const V_R: f64;
    const TAU: f64;
    const TAU_ARP: usize;

    fn new(id: usize) -> Self;
    fn run(&mut self, t: usize, i_ext: f64, 
           v_s: &Vec<Vec<u8>>, v_x: &Vec<Vec<f64>>, v_u: &Vec<Vec<f64>>) -> (u8, f64, f64);
    fn dvdt(&self, v: f64, i_rec: f64, i_ext: f64) -> f64;
}

pub struct ExtNeuron {
    v: f64,
    x: f64,
    u: f64,
    arp: usize,
    d_p: Vec<(usize, usize)>,
    d_b: Vec<(usize, usize)>,
    d_i: Vec<(usize, usize)>,
}

impl Neuron for ExtNeuron {
    const THETA: f64 = 20.0;
    const V_R: f64 = 16.0;
    const TAU: f64 = 0.015;
    const TAU_ARP: usize = 2;

    fn new(id : usize) -> Self {
        let memory = id / MEM_SIZE;
        let v = ExtNeuron::V_R;
        let x = 1.0;
        let u = U;
        let arp = 0;
        let mut d_p = Vec::new();
        let mut d_b;
        if memory < P {
            d_p = Vec::with_capacity(N_DP);
            d_b = Vec::with_capacity(N_DB);
        } else {
            d_b = Vec::with_capacity(N_DP + N_DB);
        }
        let mut d_i = Vec::with_capacity(N_DI);

        let uni = Uniform::new(D_L, D_H);
        let mut start = 0;
        while start <= N_E {
            let tmp = choice(N_CH, start, start+MEM_SIZE);
            if memory < P {
                if start / MEM_SIZE == memory {
                    for &i in tmp.iter() {
                        d_p.push((i, uni.sample(&mut rand::thread_rng())));
                    }
                } else {
                    for &i in tmp.iter() {
                        d_b.push((i, uni.sample(&mut rand::thread_rng())));
                    }
                }
            } else {
                for (j, &i) in tmp.iter().enumerate() {
                    if j < MEM_P {
                        d_p.push((i, uni.sample(&mut rand::thread_rng())));
                    } else {
                        d_b.push((i, uni.sample(&mut rand::thread_rng())));
                    }
                }
            }
            start += MEM_SIZE;
        }

        let tmp = choice(N_DI, N_E, N);
        for &i in tmp.iter() {
            d_i.push((i, uni.sample(&mut rand::thread_rng())));
        }

        ExtNeuron { v, x, u, arp, d_p, d_b, d_i }
    }

    fn run(&mut self, t: usize, i_ext: f64,
        v_s: &Vec<Vec<u8>>, v_x: &Vec<Vec<f64>>, v_u: &Vec<Vec<f64>>) -> (u8, f64, f64)
    {
        let mut s = 0;
        if self.arp > 0 {
            self.arp -= 1;
            if self.arp == 0 {
                self.v = ExtNeuron::V_R;
            }
            s = 1;
            self.u += rk4(|u: f64| self.dudt(u, 0), self.u, DT);
            self.x += rk4(|x: f64| self.dxdt(x, 0), self.x, DT);
        } else {
            let mut i_rec = 0.0;
            for &(i, d) in self.d_p.iter() {
                if d <= t && v_s[i][t-d] == 1 {
                    i_rec += J_P * v_u[i][t-d] * v_x[i][t-d];
                }
            }

            for &(i, d) in self.d_b.iter() {
                if d <= t && v_s[i][t-d] == 1 {
                    i_rec += J_B * v_u[i][t-d] * v_x[i][t-d];
                }
            }

            for &(i, d) in self.d_i.iter() {
                if d <= t && v_s[i][t-d] == 1 {
                    i_rec -= J_EI;
                }
            }

            self.v += rk4(|v: f64| self.dvdt(v, i_rec, i_ext), self.v, DT);
            if self.v >= ExtNeuron::THETA {
                self.arp = ExtNeuron::TAU_ARP;
                s = 1;
                //self.u += U * (1.0 - self.u) * 0.1;
                //self.x -= self.u * self.x * 0.1;
            }
            self.u += rk4(|u: f64| self.dudt(u, s), self.u, DT);
            self.x += rk4(|x: f64| self.dxdt(x, s), self.x, DT);
            // self.x = 1.0;
            // self.u = 0.25;
        }
        

        (s, self.x, self.u)
    }

    fn dvdt(&self, v: f64, i_rec: f64, i_ext: f64) -> f64 {
        ( - v + i_rec + i_ext ) / ExtNeuron::TAU
    }
}

impl ExtNeuron {
    fn dudt(&self, u: f64, s: u8) -> f64 {
        ( U - u ) / TAU_F + U * ( 1.0 - u ) * (s as f64) / DT
    }

    fn dxdt(&self, x: f64, s: u8) -> f64 {
        ( 1.0 - x ) / TAU_D - self.u * x * (s as f64) / DT
    }
}

pub struct InhNeuron {
    v: f64,
    arp: usize,
    d_e: Vec<(usize, usize)>,
    d_i: Vec<(usize, usize)>,
}

impl Neuron for InhNeuron {
    const THETA: f64 = 20.0;
    const V_R: f64 = 13.0;
    const TAU: f64 = 0.010;
    const TAU_ARP: usize = 2;

    fn new(_id: usize) -> Self {
        let v = InhNeuron::V_R;
        let arp = 0;
        let mut d_e = Vec::with_capacity(N_DP+N_DB);
        let mut d_i = Vec::with_capacity(N_DI);

        let uni = Uniform::new(D_L, D_H);
        let mut start = 0;
        while start <= N_E {
            let tmp = choice(N_CH, start, start+MEM_SIZE);
            for &i in tmp.iter() {
                d_e.push((i, uni.sample(&mut rand::thread_rng())));
            }
            start += MEM_SIZE;
        }

        let tmp = choice(N_DI, N_E, N);
        for &i in tmp.iter() {
            d_i.push((i, uni.sample(&mut rand::thread_rng())));
        }

        InhNeuron { v, arp, d_e, d_i }
    }

    fn run(&mut self, t: usize, i_ext: f64, 
           v_s: &Vec<Vec<u8>>, _v_x: &Vec<Vec<f64>>, _v_u: &Vec<Vec<f64>>) -> (u8, f64, f64)
    {
        let mut s = 0;
        if self.arp > 0 {
            self.arp -= 1;
            if self.arp == 0 {
                self.v = ExtNeuron::V_R;
            }
            //s = 1;
        } else {
            let mut i_rec = 0.0;
            for &(i, d) in self.d_e.iter() {
                if d <= t && v_s[i][t-d] == 1 {
                    i_rec += J_IE;
                }
            }

            for &(i, d) in self.d_i.iter() {
                if d <= t && v_s[i][t-d] == 1 {
                    i_rec -= J_II;
                }
            }
            
            self.v += rk4(|v: f64| self.dvdt(v, i_rec, i_ext), self.v, DT);
            if self.v >= ExtNeuron::THETA {
                self.arp = ExtNeuron::TAU_ARP;
                s = 1;
            }
        }
        (s, 0.0, 0.0)
    }

    fn dvdt(&self, v: f64, i_rec: f64, i_ext: f64) -> f64 {
        ( - v + i_rec + i_ext ) / InhNeuron::TAU
    }
}

pub fn choice(n: usize, start: usize, end: usize) -> Vec<usize> {
    let mut v = (start..end).collect::<Vec<usize>>();
    let mut ret = Vec::with_capacity(n);
    let num = v.len();
    if n > num {
        panic!("n is larger than end - start + 1");
    }
    let uni = Uniform::new(0.0f64, 1.0f64);
    for i in 0..n {
        let pos = (uni.sample(&mut rand::thread_rng()) * (( n - i ) as f64)) as usize;
        ret.push(v[pos]);
        v.remove(pos);
    }
    ret
}