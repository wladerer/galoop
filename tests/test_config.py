"""Test configuration loading and validation."""

import tempfile
from pathlib import Path
from galoop.config import load_config, GaloopConfig

def test_config_validation():
    """Test that config validates correctly."""
    cfg = GaloopConfig(
        slab={
            "geometry": "test.vasp",
            "energy": -100.0,
            "sampling_zmin": 10.0,
            "sampling_zmax": 15.0,
        },
        adsorbates=[
            {
                "symbol": "O",
                "chemical_potential": -4.92,
            }
        ],
        calculator_stages=[
            {
                "name": "mace",
                "type": "mace",
                "fmax": 0.10,
            }
        ],
    )
    
    assert cfg.slab.geometry == "test.vasp"
    assert len(cfg.adsorbates) == 1
    assert len(cfg.calculator_stages) == 1

def test_config_validation_fails_on_bad_data():
    """Test that validation catches bad config."""
    import pytest
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError):
        GaloopConfig(
            slab={
                "geometry": "test.vasp",
                "energy": -100.0,
                "sampling_zmin": 15.0,
                "sampling_zmax": 10.0,  # zmin > zmax, should fail
            },
            adsorbates=[
                {
                    "symbol": "O",
                    "chemical_potential": -4.92,
                }
            ],
            calculator_stages=[
                {
                    "name": "mace",
                    "type": "mace",
                }
            ],
        )

def test_load_config_from_yaml():
    """Test loading config from YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
slab:
  geometry: test.vasp
  energy: -100.0
  sampling_zmin: 10.0
  sampling_zmax: 15.0

adsorbates:
  - symbol: O
    chemical_potential: -4.92

calculator_stages:
  - name: mace
    type: mace
    fmax: 0.10
""")
        f.flush()
        
        cfg = load_config(f.name)
        
        assert cfg.slab.geometry == "test.vasp"
        assert len(cfg.adsorbates) == 1
        
        Path(f.name).unlink()
