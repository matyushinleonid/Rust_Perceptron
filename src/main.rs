extern crate nalgebra as na;

use na::{DMatrix};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::Write;

type Matrix = na::Matrix<f64, na::Dynamic, na::Dynamic, na::VecStorage<f64, na::Dynamic, na::Dynamic>>;
const E: f64 = std::f64::consts::E;

fn parse(name: &'static str) -> Matrix {
    let file = BufReader::new(File::open(name).unwrap());

    let mat: Vec<Vec<f64>> = file.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
            .map(|number| number.parse().unwrap())
            .collect())
        .collect();

    let n = mat[0].len();
    let m = mat.len();

    let mat: Vec<f64> = mat
        .iter()
        .flat_map(|array| array.iter())
        .cloned()
        .collect();

    let mut a = DMatrix::from_row_slice(m, n, &mat[..]);
    a
}

fn save_to_file(a: Matrix) -> () {
    let strings: Vec<String> = a
        .iter()
        .map(|n| n.to_string())
        .collect();

    let mut file = File::create("./y_test.txt").unwrap();
    writeln!(file, "{}", strings.join(" "));
}

fn parse_params() -> (f64, f64) {
    let file = BufReader::new(File::open("params.txt").unwrap());

    let mat: Vec<Vec<f64>> = file.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
            .map(|number| number.parse().unwrap())
            .collect())
        .collect();

    let mat: Vec<f64> = mat
        .iter()
        .flat_map(|array| array.iter())
        .cloned()
        .collect();

    (mat[0], mat[1])
}

struct Linear {
    s0: usize,
    s1: usize,
    W: Matrix,
    b: Matrix,
    batch_size: usize,
    lr: f64,
    buffer_Y:Matrix,
    buffer_X:Matrix
}
struct MSE {
    s0: usize,
    true_Y: Matrix,
    buffer_X: Matrix,
    buffer_Y: Matrix
}
struct ReLU{
    s0: usize,
    buffer_X: Matrix,
    buffer_Y: Matrix
}
struct Tanh{
    s0: usize,
    buffer_X: Matrix,
    buffer_Y: Matrix
}

trait Layer {
    fn forward(&mut self, X: Matrix) -> Matrix;
    fn init(&mut self) -> ();
    fn backward(&mut self, dLdY: Matrix) -> Matrix;
}

impl Layer for Linear {
    fn forward(&mut self, X: Matrix) -> Matrix {
        self.buffer_X = X.clone();
        self.buffer_Y = &X * &self.W;
        for i in 0 .. self.buffer_Y.len() {
            let n = i / self.batch_size;
            self.buffer_Y[i] += self.b[n];
        }
        self.buffer_Y.clone()
    }
    fn init(&mut self) {
        self.W = Matrix::new_random(self.s0, self.s1);
        //self.b = Matrix::new_random(1, self.s1);
    }
    fn backward(&mut self, dLdY: Matrix) -> Matrix {
        let mut gradW = self.buffer_X.transpose() * &dLdY;
        self.W = &self.W - self.lr * gradW;
        let mut gradb = self.b.clone();
        for i in 0 .. dLdY.len() {
            let n = i / self.batch_size;
            gradb[n] += dLdY[i];
        }
        self.b = &self.b - self.lr * gradb;
        dLdY * self.W.transpose()
    }
}
impl Layer for MSE {
    fn forward(&mut self, X: Matrix) -> Matrix {
        self.buffer_X = X.clone();
        let ones = Matrix::from_element(self.s0, 1, 1.);
        let delta = &X - &self.true_Y;
        self.buffer_Y = Matrix::from_element(1, 1, delta.dot(&delta));
        self.buffer_Y.clone()
    }
    fn init(&mut self) -> () {;}
    fn backward(&mut self, dLdY: Matrix) -> Matrix {
        let grad = &self.buffer_X - &self.true_Y;
        grad
    }
}
impl Layer for ReLU {
    fn forward(&mut self, X: Matrix) -> Matrix {
        self.buffer_X = X.clone();
        self.buffer_Y = X.map(|v|
            if v > 0. { v }
            else { 0.1 * v });
        self.buffer_Y.clone()
    }
    fn init(&mut self) -> () {;}
    fn backward(&mut self, dLdY: Matrix) -> Matrix {
        let grad = self.buffer_X.map(|v|
            if v > 0. { 1. }
            else { 0.1 });
        grad
    }
}
impl Layer for Tanh {
    fn forward(&mut self, X: Matrix) -> Matrix {
        self.buffer_X = X.clone();
        self.buffer_Y = X.map(|v|
            (E.powf(2. * v) - 1.) / (E.powf(2. * v) + 1.));
        self.buffer_Y.clone()
    }
    fn init(&mut self) -> () {;}
    fn backward(&mut self, dLdY: Matrix) -> Matrix {
        let grad = self.buffer_X.map(|v|
            ((E.powf(v) + E.powf(-v)) / 2.).powf(-2.));
        grad
    }
}

struct Perceptrone {
    l1: Linear,
    l2: ReLU,
    l3: Linear,
    l4: MSE
}

fn forward(model: &mut Perceptrone, X: Matrix) -> Matrix{
    let mut temp = model.l1.forward(X);
    temp = model.l2.forward(temp);
    temp = model.l3.forward(temp);
    temp
}
fn forward_backward(model: &mut Perceptrone, X: Matrix, y:Matrix) -> f64 {
    let mut temp = model.l1.forward(X);
    temp = model.l2.forward(temp);
    temp = model.l3.forward(temp);
    temp = model.l4.forward(temp);
    let loss = temp[0].clone();
    let mut gr = model.l4.backward(temp);
    gr = model.l3.backward(gr);
    gr = model.l2.backward(gr);
    gr = model.l1.backward(gr);
    loss
}

fn main() {
    let (lr, batch_size) = parse_params();
    let batch_size = batch_size as usize;
    let mut X_train = parse("X_train.txt");
    let mut y_train = parse("y_train.txt");

    let mut X_test = parse("X_test.txt");

    let W1:Matrix = DMatrix::from_element(4, 32, 0.01);
    let W2:Matrix = DMatrix::from_element(32, 1, 0.01);
    let b1:Matrix = DMatrix::from_element(1, 32, 0.);
    let b2:Matrix = DMatrix::from_element(1, 1, 0.);

    let mut l1 = Linear{
        W: W1.clone(),
        b: b1.clone(),
        s0: 4,
        s1: 32,
        batch_size: 5,
        lr: lr,
        buffer_X: X_train.clone(),
        buffer_Y: X_train.clone(),
    };

    let mut l2 = ReLU{
        s0: 32,
        buffer_X: X_train.clone(),
        buffer_Y: X_train.clone()
    };

    let mut l3 = Linear{
        W: W2.clone(),
        b: b2.clone(),
        s0: 32,
        s1: 1,
        batch_size: 5,
        lr: lr,
        buffer_X: X_train.clone(),
        buffer_Y: X_train.clone(),
    };

    let mut l4 = MSE{
        true_Y: y_train.clone(),
        s0: 1,
        buffer_X: X_train.clone(),
        buffer_Y: y_train.clone()
    };

    l1.init();
    l2.init();

    let mut model = Perceptrone{
        l1: l1,
        l2: l2,
        l3: l3,
        l4: l4
    };


    let mut loss = 0.;
    for i in 1..100 {
        loss = forward_backward(&mut model, X_train.clone(), y_train.clone());
    }
    println!("total loss = {}", loss);

    let y_test = forward(&mut model, X_test);
    save_to_file(y_test);
}
