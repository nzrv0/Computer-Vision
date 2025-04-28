import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(
        self,
        inputs,
        outputs,
        epochs=10000,
        seed=0,
        activation=None,
        learning_rate=0.001,
    ):
        self.inputs = inputs
        self.outputs = outputs

        self.hidden_w = None
        self.output_w = None
        self.seed = seed
        self.activation = activation
        self.classifaciton = False
        self.epochs = epochs
        self.learning_rate = learning_rate

    def load_weights(self):
        np.random.seed(self.seed)
        self.hidden_w = np.random.normal(0, 1, 4 * 10000).reshape(4, -1)
        self.output_w = np.random.normal(0, 1, 4)

    def hidden_layer(self):
        compute = np.dot(self.hidden_w, self.inputs)
        output = np.array([])
        for i in compute.flatten():
            output = np.append(output, self.activation_func(i))
        return output

    def activation_func(self, x1):
        resoult = None
        if self.activation == "sigmoid":
            resoult = 1 / (1 + np.exp(-x1))
        elif self.activation == "tanh":
            resoult = (np.exp(x1) - 1) / (np.exp(x1) + 1)
        else:
            resoult = x1 if x1 > 0 else 0

        return resoult

    def normalization(self, y1):
        norms = np.array([])
        # sum_of_exp = np.sum(np.exp(y1))
        sum_of_exp = sum([np.exp(y) for y in y1])
        for i in y1:
            if i != 0.0:
                softmax = np.exp(i) / sum_of_exp
            norms = np.append(norms, softmax)

        return norms

    def forward(self):
        hidden_output = self.hidden_layer()
        y1 = self.output_w @ hidden_output

        if self.classifaciton:
            return self.normalization(y1)
        else:
            return y1

    def backward(self):
        pass

    def train(self):
        losses = []
        for epoch in range(self.epochs):
            y = self.forward()
            loss = self.gradient(y)
            losses.append(loss)
        return losses

    def gradient(self, y_pred):
        loss = -2 / self.inputs.shape[0] * np.dot(self.inputs, (self.outputs - y_pred))
        for i in range(self.hidden_w.shape[0]):
            self.hidden_w[i, :] = (
                (self.hidden_w[i, :]).ravel() - (loss * self.learning_rate).ravel()
            )
        return loss[0]


inputs = np.random.rand(10_000, 1)
outputs = np.array([[0]])

neural_net = NeuralNetwork(inputs, outputs)
neural_net.load_weights()
res = neural_net.train()
plt.plot(res)
plt.show()
# 4x4 X 4x1
