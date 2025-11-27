"""
Command-line interface for erucakra.

Usage:
    erucakra run --scenario ssp245
    erucakra run --scenario ssp245 --threshold-fraction 0.6
    erucakra run --all-scenarios
    erucakra run --forcing ./forcing.csv
    erucakra list
    erucakra info ssp126
    erucakra sensitivity --scenario ssp245 --tf-range 0.5 0.9
"""

import sys
from pathlib import Path
from typing import List, Optional
import click

from erucakra import __version__, ClimateModel, SCENARIOS, get_scenario
from erucakra.io.forcing import load_forcing_csv
from erucakra.utils.logging import (
    setup_logging,
    start_step,
    end_step,
    log_error,
    get_timing_logger,
)
from erucakra.utils.config import load_config, DEFAULT_CONFIG


@click.group()
@click.version_option(version=__version__, prog_name="erucakra")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--config", type=click.Path(), help="Config file path")
@click.pass_context
def main(ctx, verbose, debug, config):
    """
    erucakra - Climate Tipping Point Dynamics Toy Model
    
    A physically-motivated dynamical system toy model for analyzing
    climate tipping points under various SSP scenarios.
    
    The critical threshold z_crit is computed from forcing data,
    not prescribed per scenario.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@main.command("run")
@click.option(
    "--scenario", "-s",
    type=click.Choice(["ssp126", "ssp245", "ssp370", "ssp585"]),
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
    "--threshold-fraction", "-tf",
    type=float,
    default=0.7,
    help="Fraction of max forcing for z_crit (default: 0.7)",
)
@click.option(
    "--t-end",
    type=float,
    default=600.0,
    help="End time (normalized). Default: 600.0",
)
@click.option(
    "--n-points",
    type=int,
    default=48000,
    help="Number of output points. Default: 48000",
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
@click.option(
    "--experiment-name", "-e",
    type=str,
    default=None,
    help="Experiment name for log file",
)
@click.pass_context
def run(ctx, scenario, all_scenarios, forcing, output_dir, outputs, threshold_fraction,
        t_end, n_points, no_noise, seed, log_dir, experiment_name):
    """Run climate tipping point simulation."""
    from tqdm import tqdm
    import traceback
    
    config = ctx.obj["config"]
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    output_dir = Path(output_dir)
    outputs = list(outputs) if outputs else ["csv", "netcdf", "gif", "png"]
    
    if experiment_name is None:
        if all_scenarios:
            experiment_name = "all_scenarios"
        elif scenario:
            experiment_name = scenario
        elif forcing:
            experiment_name = Path(forcing).stem
        else:
            experiment_name = config["scenarios"]["default"]
    
    level = "DEBUG" if debug else ("INFO" if verbose else "INFO")
    logger = setup_logging(
        level=level,
        log_dir=log_dir,
        experiment_name=experiment_name,
        format_style="detailed",
        always_save=True,
        include_timestamp=False,
    )
    
    try:
        # Initialize Output Directories
        start_step("Initialize output directories")
        
        subdirs = {fmt: output_dir / fmt for fmt in outputs}
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        end_step(success=True)
        
        # Initialize Model
        start_step("Initialize model")
        
        model = ClimateModel(
            c=config["model"]["damping"],
            epsilon=config["model"]["epsilon"],
            beta=config["model"]["beta"],
            threshold_fraction=threshold_fraction,
        )
        
        click.echo(f"\n{'═' * 60}")
        click.echo(f"  Model Parameters:")
        click.echo(f"{'─' * 60}")
        click.echo(f"  Damping (c)          = {model.params['c']}")
        click.echo(f"  Timescale (ε)        = {model.params['epsilon']}")
        click.echo(f"  Feedback (β)         = {model.params['beta']}")
        click.echo(f"  Threshold fraction   = {threshold_fraction}")
        click.echo(f"  (z_crit computed from forcing data)")
        click.echo(f"{'═' * 60}")
        
        end_step(success=True)
        
        # Determine Scenarios
        start_step("Determine scenarios to run")
        
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
            default_scenario = config["scenarios"]["default"]
            scenarios_to_run = [default_scenario]
            click.echo(f"\nRunning default scenario: {default_scenario}")
        
        end_step(success=True)
        
        # Run Simulations
        for scenario_key in tqdm(scenarios_to_run, desc="Processing scenarios"):
            
            start_step(f"Scenario: {scenario_key}")
            
            try:
                click.echo(f"\n{'─' * 60}")
                
                if scenario_key == "custom":
                    click.echo(f"  Processing: Custom Forcing")
                    
                    start_step("Load custom forcing")
                    times, values = load_forcing_csv(forcing)
                    end_step(success=True)
                    
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
                
                # Show computed parameters
                click.echo(f"  Computed: z_crit = {results.z_crit:.3f}, A_scale = {results.A_scale:.2f} W/m²")
                click.echo("─" * 60)
                
                # Generate Outputs
                start_step(f"Generate outputs: {base_name}")
                
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
                
                end_step(success=True)
                
                # Summary - binary regime
                summary = results.summary()
                click.echo(f"\n  Results Summary:")
                click.echo(f"    z_crit (computed): {summary['z_crit']:.3f}")
                click.echo(f"    Max z: {summary['max_z']:.3f}")
                click.echo(f"    Final z: {summary['final_z']:.3f}")
                click.echo(f"    Tipped: {summary['tipped']}")
                click.echo(f"    Time above threshold: {summary['time_above_threshold_pct']:.1f}%")
                if summary['first_crossing_year']:
                    click.echo(f"    First crossing: Year {int(summary['first_crossing_year'])}")
                
                end_step(success=True)
                
            except Exception as e:
                log_error(e, f"Scenario {scenario_key}")
                end_step(success=False)
                click.echo(f"\n  ✗ ERROR in scenario {scenario_key}: {e}", err=True)
                continue
        
        click.echo(f"\n{'═' * 70}")
        click.echo("  COMPLETE - All outputs generated successfully!")
        click.echo(f"{'═' * 70}")
        click.echo(f"\nOutput Directory: {output_dir}")
        click.echo(f"Log Directory: {log_dir}")
        
        timing_logger = get_timing_logger()
        if timing_logger:
            click.echo(timing_logger.get_summary())
        
        click.echo()
        
    except Exception as e:
        log_error(e, "Main execution")
        click.echo(f"\n{'!' * 70}", err=True)
        click.echo(f"  FATAL ERROR: {e}", err=True)
        click.echo(f"{'!' * 70}", err=True)
        click.echo(f"\nCheck log file in: {log_dir}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@main.command("list")
def list_scenarios():
    """List available scenarios."""
    click.echo("\nAvailable Scenarios:")
    click.echo("─" * 70)
    
    for key, info in SCENARIOS.items():
        click.echo(f"\n  {key}:")
        click.echo(f"    Name: {info['name']}")
        click.echo(f"    Subtitle: {info['subtitle']}")
        click.echo(f"    Description: {info['description']}")
    
    click.echo("\n" + "─" * 70)
    click.echo("\nz_crit is computed from forcing data, controlled by --threshold-fraction")
    click.echo("  erucakra run --scenario ssp245 --threshold-fraction 0.6")
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
    click.echo(f"Description: {scenario_info['description']}")
    click.echo(f"\nVisualization:")
    click.echo(f"  Primary Color: {scenario_info['color_primary']}")
    click.echo(f"  Secondary Color: {scenario_info['color_secondary']}")
    click.echo(f"  Colormap: {scenario_info['cmap_name']}")
    click.echo(f"\nNote: z_crit is computed from forcing data at runtime.")
    click.echo(f"Use --threshold-fraction to control sensitivity (default: 0.7)")
    click.echo()


@main.command("sensitivity")
@click.option(
    "--scenario", "-s",
    type=click.Choice(["ssp126", "ssp245", "ssp370", "ssp585"]),
    required=True,
    help="Scenario to analyze",
)
@click.option(
    "--tf-min",
    type=float,
    default=0.5,
    help="Minimum threshold_fraction value",
)
@click.option(
    "--tf-max",
    type=float,
    default=0.9,
    help="Maximum threshold_fraction value",
)
@click.option(
    "--n-samples",
    type=int,
    default=10,
    help="Number of threshold_fraction values to test",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="./sensitivity",
    help="Output directory",
)
@click.option(
    "--log-dir",
    type=click.Path(),
    default="./logs",
    help="Log directory",
)
@click.pass_context
def sensitivity(ctx, scenario, tf_min, tf_max, n_samples, output_dir, log_dir):
    """Run sensitivity analysis over threshold_fraction values."""
    import numpy as np
    import pandas as pd
    import traceback
    
    config = ctx.obj["config"]
    
    logger = setup_logging(
        level="INFO",
        log_dir=log_dir,
        experiment_name=f"sensitivity_{scenario}",
        format_style="detailed",
        always_save=True,
        include_timestamp=False,
    )
    
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        start_step("Initialize sensitivity analysis")
        
        model = ClimateModel(
            c=config["model"]["damping"],
            epsilon=config["model"]["epsilon"],
            beta=config["model"]["beta"],
        )
        
        click.echo(f"\nSensitivity Analysis: {scenario}")
        click.echo(f"threshold_fraction range: [{tf_min}, {tf_max}]")
        click.echo(f"Samples: {n_samples}")
        click.echo("─" * 50)
        
        end_step(success=True)
        
        start_step("Run sensitivity samples")
        
        results_list = model.sensitivity_analysis(
            scenario=scenario,
            threshold_fraction_range=(tf_min, tf_max),
            n_samples=n_samples,
            add_noise=False,
        )
        
        end_step(success=True)
        
        # Compile results
        start_step("Compile results")
        
        summary_data = []
        for tf, results in results_list:
            if results is not None:
                summary = results.summary()
                summary_data.append({
                    "threshold_fraction": tf,
                    "z_crit": summary["z_crit"],
                    "max_z": summary["max_z"],
                    "final_z": summary["final_z"],
                    "tipped": summary["tipped"],
                    "time_above_pct": summary["time_above_threshold_pct"],
                    "first_crossing": summary["first_crossing_year"],
                })
            else:
                summary_data.append({
                    "threshold_fraction": tf,
                    "z_crit": np.nan,
                    "max_z": np.nan,
                    "final_z": np.nan,
                    "tipped": None,
                    "time_above_pct": np.nan,
                    "first_crossing": None,
                })
        
        df = pd.DataFrame(summary_data)
        
        csv_path = output_dir / f"{scenario}_sensitivity.csv"
        df.to_csv(csv_path, index=False)
        click.echo(f"\nResults saved to: {csv_path}")
        
        click.echo("\nResults:")
        click.echo(df.to_string(index=False))
        
        # Find critical threshold_fraction where behavior changes
        tipped = df[df["tipped"] == True]
        not_tipped = df[df["tipped"] == False]
        
        if len(tipped) > 0 and len(not_tipped) > 0:
            critical_tf = (tipped["threshold_fraction"].max() + not_tipped["threshold_fraction"].min()) / 2
            click.echo(f"\nCritical threshold_fraction (transition): ~{critical_tf:.3f}")
        
        end_step(success=True)
        
        timing_logger = get_timing_logger()
        if timing_logger:
            click.echo(timing_logger.get_summary())
        
        click.echo()
        
    except Exception as e:
        log_error(e, "Sensitivity analysis")
        click.echo(f"\nFATAL ERROR: {e}", err=True)
        click.echo(f"Check log file in: {log_dir}", err=True)
        click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
