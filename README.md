# erucakra

**Climate Tipping Point Dynamics** - A physically-motivated dynamical systems model for analyzing climate tipping points under various SSP scenarios.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

**erucakra** (from Sanskrit एरुचक्र "wind wheel") is a Python library for simulating and analyzing climate tipping point dynamics using a physically-motivated three-variable dynamical system. The model demonstrates:

- Tipping point behavior in climate subsystems
- Hysteresis and irreversibility  
- Path-dependence of climate outcomes
- Response to various emission scenarios (SSP pathways)

## Physical Model

The core dynamical system:

```
dx/dt = y
dy/dt = x(z - z_crit - x²) - cy
dz/dt = ε(A(t) - z - βx²)
```

Where:
- **x**: Fast climate variability (interannual-decadal oscillations)
- **y**: Rate of change / momentum in climate system
- **z**: Slow accumulated forcing (ocean heat content, ice sheet state)
- **A(t)**: Time-dependent effective radiative forcing (W/m²)

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

Run a simulation with default SSP2-4.5 scenario:

```bash
erucakra run --scenario ssp245
```

Run all SSP scenarios:

```bash
erucakra run --all-scenarios
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
from erucakra import ClimateModel, scenarios
from erucakra.forcing import load_forcing_csv

# Run with built-in SSP scenario
model = ClimateModel()
results = model.run(scenario="ssp245")

# Export results
results.to_csv("outputs/csv/")
results.to_netcdf("outputs/netcdf/")
results.to_gif("outputs/gif/")
results.to_png("outputs/png/")

# Use custom forcing
forcing = load_forcing_csv("my_forcing.csv")
results = model.run(forcing=forcing)
```

## Supported Scenarios

| Scenario | Name | Description |
|----------|------|-------------|
| `ssp126` | SSP1-2.6: Sustainability | Aggressive early mitigation, net-zero by 2075 |
| `ssp245` | SSP2-4.5: Middle Road | Moderate mitigation, emissions peak ~2040 |
| `ssp370` | SSP3-7.0: Regional Rivalry | Delayed fragmented action |
| `ssp585` | SSP5-8.5: Fossil Development | No significant mitigation |
| `overshoot` | Overshoot & Return | Net-zero with temporary threshold exceedance |

## Custom Forcing Files

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
  z_critical: 1.0
  
simulation:
  t_start: 0
  t_end: 150
  n_points: 12000
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
