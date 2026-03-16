import numpy as np
import pytest

from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

# Create dummy data
nn_arch = [{"input_dim": 3, "output_dim": 2, "activation": "relu"},
    {"input_dim": 2, "output_dim": 1, "activation": "sigmoid"}
]
def neural_net():
    return NeuralNetwork(nn_arch, lr = 0.01, seed=42, batch_size = 2, epochs = 10, loss_function= "mean_squared_error")

def test_single_forward():
    # assert shape and calculation
    W = np.array([[0.5, 0.2, 0.1], [0.8, 0.3, 0.6]])
    b = np.array([[0], [0]])
    a_prev = np.array([[0.8], [0.3], [0.2]])

    nn = neural_net()
    a_curr, z_curr = nn._single_forward(W, b, a_prev, activation = 'relu')

    assert a_curr.shape == (2,1)
    assert np.allclose(a_curr, [[0.48], [0.85]])

def test_forward():
    nn = neural_net()
    X = np.random.randn(2, 3)  
    output, cache = nn.forward(X)
    
    # assert output shape
    assert output.shape == (2,1)
    
    # assert cache has right keys
    assert 'A0' in cache
    assert 'Z2' in cache  

def test_single_backprop():
    nn = neural_net()
    W = np.random.randn(2, 3)
    b = np.random.randn(2, 1)
    Z = np.random.randn(2, 2)   
    A_prev = np.random.randn(3, 2)  
    dA = np.random.randn(2, 2)  
    
    dA_prev, dW, db = nn._single_backprop(W, b, Z, A_prev, dA, 'relu')
    
    # assert shapes
    assert dW.shape == (2,3)
    assert db.shape == (2,1)
    assert dA_prev.shape == (3,2)

def test_predict():
    nn = neural_net()
    X = np.random.randn(2, 3)
    y_hat = nn.predict(X)
    
    # assert output shape
    assert y_hat.shape == (2,1)
    
    # assert output values are between 0 and 1
    assert np.all(y_hat <= 1) and np.all(y_hat >= 0)

def test_binary_cross_entropy():
    nn = neural_net()
    
    # perfect prediction
    y = np.array([[1.0]])
    y_hat = np.array([[1.0]])
    assert np.isclose(nn._binary_cross_entropy(y, y_hat), 0)
    
    # known values
    y = np.array([[1.0]])
    y_hat = np.array([[0.5]])
    assert np.isclose(nn._binary_cross_entropy(y, y_hat), -np.log(0.5))

def test_binary_cross_entropy_backprop():
    nn = neural_net()
    
    y = np.array([[1.0]])
    y_hat = np.array([[0.5]])
    dA = nn._binary_cross_entropy_backprop(y, y_hat)
    
    # assert shape
    assert dA.shape == (1,1)
    
    # assert known value
    assert np.isclose(dA, -2)

def test_mean_squared_error():
    nn = neural_net()
    
    y = np.array([[1.0]])
    y_hat = np.array([[0.0]])
    
    # identical = 0
    assert np.isclose(nn._mean_squared_error(y, y), 0)
    
    # known value
    assert np.isclose(nn._mean_squared_error(y, y_hat), 1)

def test_mean_squared_error_backprop():
    nn = neural_net()
    
    y = np.array([[1.0]])
    y_hat = np.array([[0.0]])
    dA = nn._mean_squared_error_backprop(y, y_hat)
    
    # assert shape
    assert dA.shape == (1,1)
    
    # assert known value
    assert np.isclose(dA, -2)

def test_sample_seqs():
    seqs = ['ACGT', 'TGCA', 'AAAA', 'TTTT', 'CCCC']
    labels = [True, True, False, False, False]
    
    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)
    
    # assert balanced classes
    assert sum(sampled_labels) == 3
    assert len(sampled_labels) - sum(sampled_labels) == 3
    
    # assert total length
    assert len(sampled_seqs) == 6

def test_one_hot_encode_seqs():
    # assert shape
    seqs = ['ACGT']
    result = one_hot_encode_seqs(seqs)
    assert result.shape == (1,16)
    
    # assert known values
    seqs = ['AT']
    result = one_hot_encode_seqs(seqs)
    assert np.allclose(result, [[1, 0, 0, 0, 0, 1, 0, 0]])