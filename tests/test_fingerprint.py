"""Test SOAP fingerprinting."""

import numpy as np
from galoop.fingerprint import tanimoto_similarity

def test_tanimoto_identical():
    """Test Tanimoto similarity with identical vectors."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([1.0, 2.0, 3.0])
    
    sim = tanimoto_similarity(vec1, vec2)
    assert sim == 1.0

def test_tanimoto_orthogonal():
    """Test Tanimoto similarity with orthogonal vectors."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    
    sim = tanimoto_similarity(vec1, vec2)
    assert sim == 0.0

def test_tanimoto_similar():
    """Test Tanimoto similarity with similar vectors."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([1.1, 2.1, 3.1])
    
    sim = tanimoto_similarity(vec1, vec2)
    assert 0.9 < sim < 1.0

def test_tanimoto_shape_mismatch():
    """Test Tanimoto with mismatched shapes."""
    vec1 = np.array([1.0, 2.0, 3.0])
    vec2 = np.array([1.0, 2.0])
    
    sim = tanimoto_similarity(vec1, vec2)
    assert sim == 0.0
