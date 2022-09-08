import numpy as np
from ActivationFuncs import tanh, d_tanh, inv_tanh
batch_ind = 0


def ann__init__(ann, num_layers, first_len, hidden_len, last_len):  # set up an ANN list
	ann.append(Layer(ann, first_len))
	ann.append(Layer(ann, hidden_len, first_len, ind=1))
	for i in range(num_layers - 3):
		ann.append(Layer(ann, hidden_len, hidden_len, ind=(i+2)))
	ann.append(Layer(ann, last_len, hidden_len, ind=(num_layers - 1), is_last=True))


def ann_monitor(ann, disp_shape=False):  # print out each layer in an ANN
	if disp_shape:
		print(f'{ann} Layer composition = [{len(ann[0].activ)}, {len(ann[1].activ)}x{len(ann) - 2},',
								f'{len(ann[-1].activ)}]\n\n')
	for layer in ann:
		layer.monitor(disp_shape)

	print("\n")


def ann_activate(ann):  # activate each hidden and output layer in an ANN
	for layer in ann:
		layer.activate()


def ann_calc_grads(ann):
	for layer in range(len(ann) - 1, 0, -1):
		ann[layer].calc_gradients()


def ann_add_grads(ann, batch_size):
	for layer in range(len(ann) - 1, 0, -1):
		ann[layer].add_gradients(batch_size)


class Layer:
	def __init__(self, ann, length, len_prev=0, ind=0, is_last=False):
		self.activ = np.ones(length)
		self.network = ann
		self.ind = ind
		if is_last:
			self.desired = np.ones(length)
		if ind != 0:  # every other layer has weights and biases
			self.weights = np.random.rand(len_prev, length)
			self.weight_grads = np.zeros(shape=(len_prev, length))
			self.bias = np.random.rand(length)
			self.bias_grads = np.zeros(length)

	def activate(self):  # activate this layer's neurons
		if self.network.index(self) != 0:  # Input layer activates differently
			self.activ = tanh(
				self.bias + np.matmul(self.network[self.ind - 1].activ, self.network[self.ind].weights))

	def calc_gradients(self):
		if self.ind == len(self.network) - 1:
			self.bias_grads += 2 * (self.activ - self.desired) * d_tanh(inv_tanh(self.activ))

			for k in range(len(self.network[-2].activ)):
				self.weight_grads[k] += self.network[-2].activ[k] * \
										2 * (self.activ - self.desired) * d_tanh(inv_tanh(self.activ))
		else:
			dc_dal = 0
			for k in range(len(self.network[self.ind - 1].activ) - 1):
				for j in range(len(self.network[self.ind].activ) - 1):
					dc_dal += self.network[self.ind].bias_grads[j] * \
											self.network[self.ind].weights[k, j]

				self.weight_grads[k] += self.network[self.ind - 1].activ[k] * d_tanh(inv_tanh(self.activ))

			self.bias_grads += dc_dal * d_tanh(inv_tanh(self.activ))

	def add_gradients(self, batch_size):  # change weights and biases oppositely their respective gradients
		global batch_ind
		batch_ind += 1
		if self.ind != 0 and batch_ind % batch_size == 0:
			self.weights -= self.weight_grads / batch_size
			self.bias -= self.bias_grads / batch_size
			self.weight_grads = np.zeros_like(self.weight_grads)
			self.bias_grads = np.zeros_like(self.bias_grads)
	
	def monitor(self, disp_shape=True):  # print out this layer with the option to display each array's shape
		if self.ind != 0:
			if disp_shape:
				print(f"Layer {self.ind} weights: (Shape = {self.weights.shape})\n{self.weights}\n")
				# print(f"Layer {self.ind} weight gradients:",
				# 						f"(Shape = {self.weight_grads.shape})\n{self.weight_grads}\n")
				print(f"Layer {self.ind} biases: (Shape = {self.bias.shape})\n{self.bias}\n")
				# print(f"Layer {self.ind} bias gradients: (Shape = {self.bias.shape})\n{self.bias_grads}\n")
				print(f"Layer {self.ind} activations: (Shape = {self.activ.shape})\n{self.activ}\n\n")
			else:
				print(f"Layer {self.ind} weights:\n{self.weights}\n")
				# print(f"Layer {self.ind} weight gradients:\n{self.weight_grads}\n")
				print(f"Layer {self.ind} biases:\n{self.bias}\n")
				# print(f"Layer {self.ind} bias gradients:\n{self.bias_grads}\n")
				print(f"Layer {self.ind} activations:\n{self.activ}\n\n")
		else:
			if disp_shape:
				print(f"Input layer activations: (Shape = {self.activ.shape})\n{self.activ}\n")
			else:
				print(f"Input Layer activations:\n{self.activ}\n")
		
		if 'desired' in vars(self):
			print(f"Network desired results:\n{self.desired}")


def inrange(i, range, target, outside=False):
	if not outside and (target - range <= i <= target + range):
		return True
	elif outside and not (target - range <= i <= target + range):
		return True
	else:
		return False


# i = 1
# ann__init__(ANN1, first_len=4, hidden_len=5, last_len=3, )
# ann_activate()
# ann_calc_grads()
# ann_add_grads()
# while inrange(ANN1[-1].activ[0], 0.00001, ANN1[-1].desired[0], outside=True):
# 	ann_activate()
# 	ann_calc_grads()
# 	ann_add_grads()
# 	i += 1
# ann_monitor()
# print(f'{i} repetitions')
