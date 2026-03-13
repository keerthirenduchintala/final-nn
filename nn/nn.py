# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union(int, str)]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Linear transform: Z = (weight matrix * previous layer activation matrix + bias)
        Z_curr = W_curr @ A_prev + b_curr

        if activation == 'relu':
            A_curr = self._relu(Z_curr)
        elif activation == "sigmoid":
            A_curr = self._sigmoid(Z_curr)
        else:
            raise TypeError("not a valid activation function!")
        
        return (A_curr, Z_curr)

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.
        runs pass on entire network by calling _single_forward on each layer

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features(input neurons)].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        # transpose X to get correct dimensions for A0
        A_curr = X.T
        cache['A0'] = A_curr

        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = layer['activation']
            
            (A_curr, Z_curr) = self._single_forward(W_curr,b_curr,A_curr,activation)

            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr

        output = A_curr.T
        return (output, cache)

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # need to compute dZ
        if activation_curr == 'relu':
            dZ = self._relu_backprop(dA_curr, Z_curr)
        elif activation_curr == "sigmoid":
            dZ = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            raise TypeError("not a valid activation function!")
        # gradients
        batch_size = A_prev.shape[1]
        # gradient of weight
        dW_curr = dZ @ A_prev.T / batch_size
        # gradient of bias
        db_curr = np.sum(dZ, axis=1, keepdims=True) / batch_size
        # gradient of previous activation
        dA_prev = W_curr.T @ dZ

        return (dA_prev, dW_curr, db_curr)

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # initialize grad_dict
        grad_dict = {}

        # initialize dA_curr from the loss function
        if self._loss_func == 'binary_cross_entropy':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat).T
        elif self._loss_func == 'mean_squared_error':
            dA_curr = self._mean_squared_error_backprop(y,y_hat).T
        else:
            raise TypeError('not a valid loss function')
        
        for idx in reversed(range(len(self.arch))):
            layer_idx = idx + 1
            # retrieve inputs for _single_backprop
            activation_curr = self.arch[idx]['activation']
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z'+ str(layer_idx)]
            A_prev = cache['A' + str(idx)]
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr
            dA_curr = dA_prev

        
        return (grad_dict)

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for idx in range(len(self.arch)):
            layer_idx = idx + 1
            # gradient descent
            # W = W - learningrate * dW
            # b = b - learningrate * db
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        #initialize return lists
        per_epoch_loss_train = []
        per_epoch_loss_val = []
        
        # For every epoch, shuffle order data and generate mini-batches - so that mini-batches are different and shuffled each time within epoch
        for epoch in range(self._epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
 
            batch_losses = []

            for start in range(0, X_train.shape[0], self._batch_size):
                end = start + self._batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
 
                # first Forward pass
                y_hat, cache = self.forward(X_batch)
 
                # batch loss
                if self._loss_func == 'binary_cross_entropy':
                    loss = self._binary_cross_entropy(y_batch, y_hat)
                elif self._loss_func == 'mean_squared_error':
                    loss = self._mean_squared_error(y_batch, y_hat)
                batch_losses.append(loss)
 
                # then Backward pass + parameter update
                grad_dict = self.backprop(y_batch, y_hat, cache)
                self._update_params(grad_dict)
 
            # Record average training loss for this epoch
            per_epoch_loss_train.append(np.mean(batch_losses))
 
            # validation loss - no gradient descent update on validation set
            y_val_hat, _ = self.forward(X_val)
            if self._loss_func == 'binary_cross_entropy':
                val_loss = self._binary_cross_entropy(y_val, y_val_hat)
            elif self._loss_func == 'mean_squared_error':
                val_loss = self._mean_squared_error(y_val, y_val_hat)
            per_epoch_loss_val.append(val_loss)
 
        return (per_epoch_loss_train, per_epoch_loss_val)
 
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # forward for prediction

        y_hat, cache = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1+ np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # dZ = dA * sigmoid(Z) * (1-sigmoid(Z))

        sigmoid = self._sigmoid(Z)
        dZ = dA * sigmoid * (1-sigmoid)
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # reLU = max(0,Z)
        nl_transform = np.maximum(0,Z)
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # dZ = dA * Z
        dZ = dA * (Z > 0)
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        #  similar to HW 7 

        # have to prevent log(0)
        y_hat = np.clip(y_hat, 1e-9, 1-(1e-9))

        # binary cross entropy
        sample_loss = (y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        loss = -np.mean(sample_loss)
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # dA = -(y / y_hat) + (1 - y) / (1 - y_hat)

        # avoid division by 0 
        y_hat = np.clip(y_hat, 1e-9, 1-(1e-9))
        dA = -(y / y_hat) + (1 - y) / (1 - y_hat)
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        # MSE = 1/n * sum((y-y_hat)**2)
        n = y.shape[0]
        loss = 1/n * np.sum((y-y_hat)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # dA = 2/n * (y_hat - y)
        n = y.shape[0]
        dA = 2/n * (y_hat - y)
        return dA