import sys
import os
import pytest

# Agregar la carpeta 'src' al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import load_and_preprocess_data

def test_preprocessing_shapes():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]