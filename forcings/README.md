# Custom Forcing Files

Place your custom forcing files here in CSV format.

## Format

```csv
time,forcing
0,0.0
50,0.15
100,0.35
...
```

**Columns:**
- `time`: Years from 1750 (so 1750=0, 2020=270, 2100=350)
- `forcing`: Effective radiative forcing in W/m²

## Usage

```bash
erucakra run --forcing ./forcings/my_forcing.csv
```
