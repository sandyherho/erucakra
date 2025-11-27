# `erucakra`: A physically-motivated dynamical system toy model for analyzing climate tipping points under various SSP scenarios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

**erucakra** is a Python library for simulating and analyzing climate tipping point dynamics using a physically-motivated three-variable dynamical system. The model demonstrates:

- Tipping point behavior via pitchfork bifurcation
- Hysteresis and path-dependence
- Response to various emission scenarios (SSP pathways)

**Key design principle**: The critical threshold `z_crit` is computed from forcing data characteristics, not prescribed per scenario. This ensures the model discovers tipping behavior from physics rather than imposing expected outcomes.

## Physical Model

The core dynamical system:
```
dx/dt = y
dy/dt = x(z - z_crit - x²) - cy
dz/dt = ε(A(t)/A_scale - z - βx²)
```

Where:
- **x**: Fast climate variability (interannual-decadal oscillations)
- **y**: Rate of change / momentum in climate system
- **z**: Slow accumulated forcing (ocean heat content, ice sheet state)
- **A(t)**: Time-dependent effective radiative forcing (W/m²)
- **z_crit**: Critical threshold (computed from forcing data)

### Bifurcation Behavior

- **z < z_crit**: Single stable fixed point at x=0 (not tipped)
- **z > z_crit**: Pitchfork bifurcation → two stable fixed points at x = ±√(z - z_crit) (tipped)

### Threshold Computation

The critical threshold is computed as:
```
z_crit = threshold_fraction × max(A_normalized)
```

Where `A_normalized = A(t) / A_scale` and `A_scale` is the 95th percentile of forcing values. This ensures consistent behavior across scenarios while allowing user control via `threshold_fraction` (default: 0.7).

## Installation
```bash
pip install erucakra
```

Or with Poetry:
```bash
poetry add erucakra
```

## Quick Start

### Command Line Interface

Run a simulation with default settings:
```bash
erucakra run --scenario ssp245
```

Run all SSP scenarios:
```bash
erucakra run --all-scenarios
```

Control tipping sensitivity:
```bash
# Lower threshold_fraction = tips earlier
erucakra run --scenario ssp245 --threshold-fraction 0.6

# Higher threshold_fraction = tips later
erucakra run --scenario ssp245 --threshold-fraction 0.85
```

Use custom forcing file:
```bash
erucakra run --forcing ./forcings/my_forcing.csv --output-dir ./results
```

Specify output formats:
```bash
erucakra run --scenario ssp585 --outputs csv netcdf gif png
```

### Python API
```python
from erucakra import ClimateModel

# Run with built-in SSP scenario
model = ClimateModel(threshold_fraction=0.7)
results = model.run(scenario="ssp245")

# z_crit is computed automatically from forcing data
print(f"Computed z_crit: {results.z_crit:.3f}")
print(f"Computed A_scale: {results.A_scale:.2f} W/m²")
print(f"Tipped: {results.tipped}")

# Export results
results.to_csv("outputs/csv/")
results.to_netcdf("outputs/netcdf/")
results.to_gif("outputs/gif/")
results.to_png("outputs/png/")

# Sensitivity analysis over threshold_fraction
results_list = model.sensitivity_analysis(
    scenario="ssp245",
    threshold_fraction_range=(0.5, 0.9),
    n_samples=10,
)
```

## Supported Scenarios

| Scenario | Name | Description |
|----------|------|-------------|
| `ssp126` | SSP1-2.6: Sustainability | Low emissions, aggressive early mitigation |
| `ssp245` | SSP2-4.5: Middle Road | Intermediate emissions, gradual mitigation |
| `ssp370` | SSP3-7.0: Regional Rivalry | High emissions, fragmented mitigation |
| `ssp585` | SSP5-8.5: Fossil Development | Very high emissions, no significant mitigation |

## Parameters

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `c` | 0.2 | Damping coefficient (energy dissipation) |
| `epsilon` | 0.02 | Timescale separation (slow/fast ratio) |
| `beta` | 0.8 | Feedback strength (variability → accumulation) |
| `threshold_fraction` | 0.7 | Fraction of max forcing for z_crit computation |

### Understanding `threshold_fraction`

The `threshold_fraction` parameter controls when tipping occurs relative to peak forcing:

- **Lower values (0.5-0.6)**: System tips earlier, more sensitive
- **Default (0.7)**: Moderate sensitivity
- **Higher values (0.8-0.9)**: System tips later, less sensitive

This is the primary user-controllable parameter for exploring tipping sensitivity.

## Binary Regime Classification

The model uses a simple binary classification:
- **not_tipped**: z < z_crit (single stable state)
- **tipped**: z > z_crit (bistable, system in altered state)

This emerges from the physics of the pitchfork bifurcation rather than being prescribed.

## Forcing Data

The model uses IPCC AR6-based radiative forcing data from 1750 to 2500. Forcing files are provided in CSV format in the `forcings/` directory.

### Custom Forcing Files

Create a CSV file with columns `time` (years from 1750) and `forcing` (W/m²):
```csv
time,forcing
0,0.0
100,0.5
150,1.0
250,2.0
350,3.5
```

## Output Formats

- **CSV**: Full simulation data with derived quantities
- **NetCDF**: CF-compliant netCDF4 files with metadata
- **GIF**: Animated 3D phase space evolution
- **PNG**: Publication-quality time series plots

## Configuration

Default configuration in `~/.erucakra/config.yaml`:
```yaml
scenarios:
  default: ssp245
  
model:
  damping: 0.2
  epsilon: 0.02
  beta: 0.8
  threshold_fraction: 0.7
  
simulation:
  t_start: 0
  t_end: 600
  n_points: 48000
  add_noise: true
  noise_level: 0.03
  
outputs:
  formats: [csv, netcdf, gif, png]
  base_dir: ./outputs
```

## References

- IPCC AR6 WG1 (2021) - Physical Science Basis
- Lenton et al. (2019) - Climate tipping points
- Ritchie et al. (2020) - SSP scenario analysis

## Author

**Sandy H. S. Herho**

## License

MIT License - see [LICENSE](LICENSE) for details.
