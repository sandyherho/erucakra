# `erucakra`: A dynamical system toy model for analyzing climate tipping points under various SSP scenarios

[![PyPI version](https://badge.fury.io/py/erucakra.svg)](https://badge.fury.io/py/erucakra)
[![Python](https://img.shields.io/pypi/pyversions/erucakra.svg)](https://pypi.org/project/erucakra/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?logo=matplotlib&logoColor=black)](https://matplotlib.org/)
[![NetCDF](https://img.shields.io/badge/NetCDF-CF--compliant-blue)](https://www.unidata.ucar.edu/software/netcdf/)


## Model Description

The model describes climate system dynamics through a three-variable ordinary differential equation system exhibiting bifurcation behavior:

$$\frac{dx}{dt} = y$$

$$\frac{dy}{dt} = x(z - z_{crit} - x^2) - cy$$

$$\frac{dz}{dt} = \varepsilon \left( \frac{A(t)}{A_{scale}} - z - \beta x^2 \right)$$

### Variables

| Symbol | Description |
|--------|-------------|
| $x$ | Fast climate variability (interannual oscillations, e.g., ENSO-like modes) |
| $y$ | Rate of change of $x$ (momentum/velocity in phase space) |
| $z$ | Slow accumulated forcing state (ocean heat content, ice sheet mass proxy) |
| $A(t)$ | Time-dependent radiative forcing (W/m²) from SSP scenarios |

### Parameters

| Symbol | Description | Default |
|--------|-------------|---------|
| $c$ | Damping coefficient controlling oscillation decay rate | 0.2 |
| $\varepsilon$ | Timescale separation ratio (slow/fast dynamics) | 0.02 |
| $\beta$ | Feedback strength from variability to accumulation | 0.8 |
| $z_{crit}$ | Critical threshold for tipping (computed from forcing) | auto |
| $A_{scale}$ | Forcing normalization scale (W/m²) | scenario-dependent |

### Threshold Computation

The critical threshold $z_{crit}$ is computed from forcing data:

$$z_{crit} = \text{threshold\_fraction} \times \max\left(\frac{A(t)}{A_{scale}}\right)$$

where threshold fraction (default 0.7) controls tipping sensitivity—lower values trigger earlier tipping.

### Tipping Dynamics

The system undergoes a pitchfork bifurcation when $z$ crosses $z_{crit}$:
- $z < z_{crit}$: Single stable equilibrium at $x = 0$ (stable climate)
- $z > z_{crit}$: Bistable regime with equilibria at $x = \pm\sqrt{z - z_{crit}}$ (tipped state)

## Installation

### From PyPI

```bash
pip install erucakra
```

### From Source

```bash
git clone https://github.com/sandyherho/erucakra.git
cd erucakra
pip install poetry
poetry install
```

## Usage

### Command Line

Run with a built-in SSP scenario:
```bash
erucakra run --scenario ssp245
```

Run all scenarios:
```bash
erucakra run --all-scenarios
```

Adjust tipping sensitivity:
```bash
erucakra run --scenario ssp370 --threshold-fraction 0.6
```

Use custom forcing data:
```bash
erucakra run --forcing ./my_forcing.csv
```

Use a configuration file:
```bash
erucakra run --config ./configs/custom.yaml
```

List available scenarios:
```bash
erucakra list
```

Run sensitivity analysis:
```bash
erucakra sensitivity --scenario ssp245 --tf-min 0.5 --tf-max 0.9 --n-samples 10
```

### Python API

```python
from erucakra import ClimateModel

# Initialize model
model = ClimateModel(
    c=0.2,              # damping
    epsilon=0.02,       # timescale separation
    beta=0.8,           # feedback strength
    threshold_fraction=0.7
)

# Run simulation
results = model.run(
    scenario="ssp245",
    t_end=600.0,
    add_noise=True
)

# Check if system tipped
print(f"Tipped: {results.tipped}")
print(f"Max z: {results.max_z:.3f}")
print(f"z_crit: {results.z_crit:.3f}")

# Export outputs
results.to_csv("output.csv")
results.to_netcdf("output.nc")
results.to_png("timeseries.png")
results.to_gif("phase_space.gif")
```

### Custom Forcing Data

Forcing CSV format:
```csv
time,forcing
1750,0.3
1850,0.5
2000,2.5
2100,5.5
```

```python
from erucakra.io import load_forcing_csv

times, values = load_forcing_csv("my_forcing.csv")
results = model.run(forcing=values, forcing_times=times)
```

## Available Scenarios

| Scenario | Description | Forcing Peak |
|----------|-------------|--------------|
| `ssp126` | SSP1-2.6 Sustainability | ~3.6 W/m² |
| `ssp245` | SSP2-4.5 Middle Road | ~5.4 W/m² |
| `ssp370` | SSP3-7.0 Regional Rivalry | ~8.2 W/m² |
| `ssp585` | SSP5-8.5 Fossil Development | ~10+ W/m² |

## Output Formats

- **CSV**: Full data table with all variables
- **NetCDF**: CF-compliant NetCDF4 with compression and metadata
- **PNG**: Time series diagnostic plot
- **GIF**: Animated 3D phase space visualization

## License

MIT License © 2025 Sandy H. S. Herho
