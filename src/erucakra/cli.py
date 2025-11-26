"""
Command-line interface for erucakra.

Usage:
    erucakra run --scenario ssp245
    erucakra run --all-scenarios
    erucakra run --forcing ./forcing.csv
    erucakra list
    erucakra info ssp126
"""

import sys
from pathlib import Path
from typing import List, Optional
import click

from erucakra import __version__, ClimateModel, SCENARIOS, get_scenario
from erucakra.io.forcing import load_forcing_csv
from erucakra.utils.logging import setup_logging
from erucakra.utils.config import load_config, DEFAULT_CONFIG


@click.group()
@click.version_option(version=__version__, prog_name="erucakra")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--config", type=click.Path(), help="Config file path")
@click.pass_context
def main(ctx, verbose, debug, config):
    """
    erucakra - Climate Tipping Point Dynamics Model
    
    A physically-motivated dynamical systems model for analyzing
    climate tipping points under various SSP scenarios.
    """
    ctx.ensure_object(dict)
    
    # Load config
    ctx.obj["config"] = load_config(config)
    
    # Setup logging
    level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    ctx.obj["logger"] = setup_logging(level=level)


@main.command("run")
@click.option(
    "--scenario", "-s",
    type=click.Choice(["ssp126", "ssp245", "ssp370", "ssp585", "overshoot"]),
    help="Built-in scenario to run",
)
@click.option(
    "--all-scenarios", "-a",
    is_flag=True,
    help="Run all built-in scenarios",
)
@click.option(
    "--forcing", "-f",
    type=click.Path(exists=True),
    help="Custom forcing CSV file (time,forcing in W/m²)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="./outputs",
    help="Output directory (default: ./outputs)",
)
@click.option(
    "--outputs",
    type=click.Choice(["csv", "netcdf", "gif", "png"]),
    multiple=True,
    default=["csv", "netcdf", "gif", "png"],
    help="Output formats (default: all)",
)
@click.option(
    "--t-end",
    type=float,
    default=150.0,
    help="End time (normalized). Default: 150.0 (~120 years)",
)
@click.option(
    "--n-points",
    type=int,
    default=12000,
    help="Number of output points. Default: 12000",
)
@click.option(
    "--no-noise",
    is_flag=True,
    help="Disable climate noise",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--log-dir",
    type=click.Path(),
    default="./logs",
    help="Log directory",
)
@click.pass_context
def run(ctx, scenario, all_scenarios, forcing, output_dir, outputs, t_end, n_points, no_noise, seed, log_dir):
    """Run climate tipping point simulation."""
    from tqdm import tqdm
    
    config = ctx.obj["config"]
    
    # Setup logging to file
    logger = setup_logging(
        level="DEBUG",
        log_dir=log_dir,
        include_timestamp=True,
        format_style="detailed",
    )
    
    output_dir = Path(output_dir)
    outputs = list(outputs) if outputs else ["csv", "netcdf", "gif", "png"]
    
    # Create output subdirectories
    subdirs = {fmt: output_dir / fmt for fmt in outputs}
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("  ERUCAKRA - Climate Tipping Point Dynamics")
    logger.info("=" * 70)
    
    # Initialize model
    model = ClimateModel(
        c=config["model"]["damping"],
        epsilon=config["model"]["epsilon"],
        beta=config["model"]["beta"],
        z_crit=config["model"]["z_critical"],
    )
    
    click.echo(f"\nModel Parameters:")
    click.echo(f"  Damping (c)     = {model.params['c']}")
    click.echo(f"  Timescale (ε)   = {model.params['epsilon']}")
    click.echo(f"  Feedback (β)    = {model.params['beta']}")
    click.echo(f"  Threshold       = {model.params['z_crit']}")
    
    # Determine scenarios to run
    scenarios_to_run = []
    
    if all_scenarios:
        scenarios_to_run = list(SCENARIOS.keys())
        click.echo(f"\nRunning all {len(scenarios_to_run)} scenarios")
    elif scenario:
        scenarios_to_run = [scenario]
        click.echo(f"\nRunning scenario: {scenario}")
    elif forcing:
        scenarios_to_run = ["custom"]
        click.echo(f"\nUsing custom forcing: {forcing}")
    else:
        # Default scenario
        default_scenario = config["scenarios"]["default"]
        scenarios_to_run = [default_scenario]
        click.echo(f"\nRunning default scenario: {default_scenario}")
    
    # Run simulations
    for scenario_key in tqdm(scenarios_to_run, desc="Processing scenarios"):
        click.echo(f"\n{'─' * 60}")
        
        if scenario_key == "custom":
            click.echo(f"  Processing: Custom Forcing")
            times, values = load_forcing_csv(forcing)
            results = model.run(
                forcing=values,
                forcing_times=times,
                t_end=t_end,
                n_points=n_points,
                add_noise=not no_noise,
                seed=seed,
                show_progress=True,
            )
            base_name = Path(forcing).stem
        else:
            info = get_scenario(scenario_key)
            click.echo(f"  Processing: {info['name']}")
            click.echo(f"  {info['subtitle']}")
            
            results = model.run(
                scenario=scenario_key,
                t_end=t_end,
                n_points=n_points,
                add_noise=not no_noise,
                seed=seed,
                show_progress=True,
            )
            base_name = scenario_key
        
        click.echo("─" * 60)
        
        # Generate outputs
        click.echo("  Generating outputs...")
        
        if "csv" in outputs:
            csv_path = subdirs["csv"] / f"{base_name}_data.csv"
            results.to_csv(csv_path)
            click.echo(f"    ✓ CSV: {csv_path}")
        
        if "netcdf" in outputs:
            nc_path = subdirs["netcdf"] / f"{base_name}_data.nc"
            results.to_netcdf(nc_path)
            click.echo(f"    ✓ NetCDF: {nc_path}")
        
        if "png" in outputs:
            png_path = subdirs["png"] / f"{base_name}_timeseries.png"
            results.to_png(png_path)
            click.echo(f"    ✓ PNG: {png_path}")
        
        if "gif" in outputs:
            gif_path = subdirs["gif"] / f"{base_name}_phase_space.gif"
            results.to_gif(gif_path)
            click.echo(f"    ✓ GIF: {gif_path}")
        
        # Summary
        summary = results.summary()
        click.echo(f"\n  Results Summary:")
        click.echo(f"    Max z: {summary['max_z']:.3f}")
        click.echo(f"    Final z: {summary['final_z']:.3f}")
        click.echo(f"    Crossed threshold: {summary['crossed_threshold']}")
        click.echo(f"    Time above threshold: {summary['time_above_threshold_pct']:.1f}%")
    
    click.echo(f"\n{'=' * 70}")
    click.echo("  COMPLETE - All outputs generated successfully!")
    click.echo(f"{'=' * 70}")
    click.echo(f"\nOutput Directory: {output_dir}")
    click.echo(f"Log Directory: {log_dir}\n")


@main.command("list")
def list_scenarios():
    """List available scenarios."""
    click.echo("\nAvailable Scenarios:")
    click.echo("─" * 60)
    
    for key, info in SCENARIOS.items():
        click.echo(f"\n  {key}:")
        click.echo(f"    Name: {info['name']}")
        click.echo(f"    Subtitle: {info['subtitle']}")
        click.echo(f"    Expected: {info['expected_outcome']}")
        click.echo(f"    Description: {info['description']}")
    
    click.echo()


@main.command("info")
@click.argument("scenario")
def info(scenario):
    """Show detailed information about a scenario."""
    try:
        scenario_info = get_scenario(scenario)
    except KeyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    click.echo(f"\n{scenario_info['name']}")
    click.echo("=" * 60)
    click.echo(f"Subtitle: {scenario_info['subtitle']}")
    click.echo(f"Expected Outcome: {scenario_info['expected_outcome']}")
    click.echo(f"Description: {scenario_info['description']}")
    click.echo(f"\nVisualization:")
    click.echo(f"  Primary Color: {scenario_info['color_primary']}")
    click.echo(f"  Secondary Color: {scenario_info['color_secondary']}")
    click.echo(f"  Colormap: {scenario_info['cmap_name']}")
    click.echo()


if __name__ == "__main__":
    main()
