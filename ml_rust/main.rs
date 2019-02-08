#[derive(Clone)]
struct Matriz {
	vector: Vec<f32>,
	rows: usize,
	columns: usize,
}

impl Matriz {
	fn create_matriz_random(rows: usize, columns: usize) -> Matriz {
		let mut vector = Vec::with_capacity(rows*columns);

		for _i in 0..(rows*columns){
			vector.push(random_number(-1,1));
		}

		Matriz {vector, rows, columns}
	}

	fn create_matriz_null() -> Matriz {
		let vector = Vec::with_capacity(0);
		let rows = 0;
		let columns = 0;

		Matriz {vector, rows, columns}
	}

	fn create_matriz(rows: usize, columns: usize, vector: Vec<f32>) -> Matriz {
		Matriz {vector, rows, columns}
	}

	fn show(&self) {
		print!("{}", "[");
		for i in 0..self.rows {
			print!("{}", "[");
			for j in 0..self.columns {
				print!("{}", self.vector[(i*self.columns)+j]);
				if j!=self.columns-1 {
					print!("{}", ", ");
				}
			}
			print!("{}", "]");
			if i!=self.rows-1 {
				println!("{}", ", ");
			}
		}
		println!("{}", "]");
	}
}

// Incluye la capa de entrada, la cual solo tiene 'output', es decir, es la matriz del set de entrenamiento o de prueba
struct NeuralLayer {
	weights: Matriz,
	bias: Matriz,
	deltas: Matriz,
	output: Matriz,
}

impl NeuralLayer {
	fn create_neural_layer(weights: Matriz, bias: Matriz, deltas: Matriz, output: Matriz) -> NeuralLayer {
		NeuralLayer{weights, bias, deltas, output}
	}
}

struct NeuralNet {
	layers: usize,
	neural_net: Vec<NeuralLayer>,
}

impl NeuralNet {
	fn create_neural_net(layers: usize, input: Matriz, topo_nn: Vec<usize>) -> NeuralNet {
		let mut neural_net: Vec<NeuralLayer> = Vec::with_capacity(layers);

		for (index, (i, j)) in topo_nn.iter().zip(topo_nn.iter().skip(1)).enumerate() {
			if index==0 {
				let m1 = Matriz::create_matriz_null();
				let m2 = Matriz::create_matriz_null();
				let m3 = Matriz::create_matriz_null();

				let nl1 = NeuralLayer::create_neural_layer(m1, m2, m3, input);
				neural_net.push(nl1);
			}else{
				let w = Matriz::create_matriz_random(*i,*j);
				let b = Matriz::create_matriz_random(1,*j);
				let m1 = Matriz::create_matriz_null();
				let m2 = Matriz::create_matriz_null();
				let nlm = NeuralLayer::create_neural_layer(w, b, m1, m2);
				neural_net.push(nlm);
			}
		}

		NeuralNet {layers, neural_net}
	}

	fn show(&self){
		for (i, layer) in self.neural_net.iter().enumerate() {
			if i==0 {
				continue;
			}else{
				print!("{}", "Weights\n");
				layer.weights.show();
				print!("{}", "Bias\n");
				layer.bias.show();
			}
		}
	}

	fn train(&self, exp_input: Matriz){
		unimplemented!();
	}
}

fn random_number(min: i32, max: i32) -> f32 {
	assert!(min < max);

	let dif = max - min;
	let x: f32 = rand::random();

	dif as f32 * x + min as f32
}

fn main() {
	// Sobre la red
	let layers = 3;
	let mut topo_nn = Vec::with_capacity(layers);
	topo_nn = [2, 3, 1].to_vec();

	// Sobre los datos de entrada de entrenamiento
	let rows_x = 3;
	let columns_x = 2;
	let mut vec_x = Vec::with_capacity(rows_x*columns_x);
	vec_x = [0.0,0.0, 1.0,0.0, 0.0,1.0, 1.0,1.0].to_vec();
	let X = Matriz::create_matriz(rows_x, columns_x, vec_x);

	// Sobre los datos de salida esperada
	let rows_y = 3;
	let columns_y = 1;
	let mut vec_y = Vec::with_capacity(rows_y*columns_y);
	vec_y = [0.0, 1.0, 1.0, 0.0].to_vec();
	let Y = Matriz::create_matriz(rows_y, columns_y, vec_y);

	// Creación de la ANN
	let ann = NeuralNet::create_neural_net(layers, X, topo_nn);

	// mostrar ann
	ann.show();

	// Training
	ann.train(Y);
}