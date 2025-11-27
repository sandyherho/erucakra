"""
Command-line interface for erucakra.

Usage:
    erucakra run --config ./configs/custom_config.yaml
    erucakra run --scenario ssp245
    erucakra run --scenario ssp245 --threshold-fraction 0.6
    erucakra run --all-scenarios
    erucakra run --forcing ./custom_forcing.csv
    erucakra list
    erucakra info ssp126
    erucakra sensitivity --scenario ssp245 --tf-min 0.5 --tf-max 0.9
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
def main():
    """
    erucakra - Climate Tipping Point Dynamics Toy Model
    
    A physically-motivated dynamical system toy model for analyzing
    climate tipping points under various SSP scenarios.
    
    The critical threshold z_crit is computed from forcing data,
    controlled by --threshold-fraction.
    """
    pass


@main.command("run")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Config YAML file (can specify forcing_file, model params, etc.)",
)
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
    default=None,
    help="Output directory (default: from config or ./outputs)",
)
@click.option(
    "--outputs",
    type=click.Choice(["csv", "netcdf", "gif", "png"]),
    multiple=True,
    default=None,
    help="Output formats (default: from config)",
)
@click.option(
    "--threshold-fraction", "-tf",
    type=float,
    default=None,
    help="Fraction of max forcing for z_crit computation. Lower = tips earlier.",
)
@click.option(
    "--t-end",
    type=float,
    default=None,
    help="End time (normalized).",
)
@click.option(
    "--n-points",
    type=int,
    default=None,
    help="Number of output points.",
)
@click.option(
    "--no-noise",
    is_flag=True,
    default=False,
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
    default=None,
    help="Log directory",
)
@click.option(
    "--experiment-name", "-e",
    type=str,
    default=None,
    help="Experiment name for log file",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug output",
)
def run(config, scenario, all_scenarios, forcing, output_dir, outputs, threshold_fraction,
        t_end, n_points, no_noise, seed, log_dir, experiment_name, verbose, debug):
    """Run climate tipping point simulation."""
    from tqdm import tqdm
    import traceback
    
    # Load config file if provided, otherwise use defaults
    if config:
        cfg = load_config(config)
        config_path = Path(config)
    else:
        cfg = DEFAULT_CONFIG.copy()
        config_path = None
    
    # =========================================================================
    # Resolve parameters: CLI args override config, config overrides defaults
    # =========================================================================
    
    # Forcing: CLI --forcing > config forcing_file > CLI --scenario > config default scenario
    forcing_file = forcing  # CLI takes priority
    if forcing_file is None:
        forcing_file = cfg.get("forcing_file")  # Then config
    
    # Output directory
    if output_dir is None:
        output_dir = cfg.get("outputs", {}).get("base_dir", "./outputs")
    output_dir = Path(output_dir)
    
    # Output formats
    if not outputs:
        outputs = cfg.get("outputs", {}).get("formats", ["csv", "netcdf", "gif", "png"])
    else:
        outputs = list(outputs)
    
    # Model parameters from config
    model_config = cfg.get("model", {})
    sim_config = cfg.get("simulation", {})
    
    # Threshold fraction: CLI > config > default
    if threshold_fraction is None:
        threshold_fraction = model_config.get("threshold_fraction", 0.7)
    
    # Time end: CLI > config > default
    if t_end is None:
        t_end = sim_config.get("t_end", 600.0)
    
    # Number of points: CLI > config > default
    if n_points is None:
        n_points = sim_config.get("n_points", 48000)
    
    # Noise: CLI --no-noise > config > default
    if no_noise:
        add_noise = False
    else:
        add_noise = sim_config.get("add_noise", True)
    
    # Log directory
    if log_dir is None:
        log_dir = cfg.get("logging", {}).get("log_dir", "./logs")
    
    # Get scenario metadata from config (for custom forcings)
    scenario_meta = cfg.get("scenario", {})
    
    # Determine experiment name
    if experiment_name is None:
        if all_scenarios:
            experiment_name = "all_scenarios"
        elif scenario:
            experiment_name = scenario
        elif forcing_file:
            experiment_name = Path(forcing_file).stem
        elif scenario_meta.get("key"):
            experiment_name = scenario_meta["key"]
        else:
            experiment_name = cfg.get("scenarios", {}).get("default", "ssp245")
    
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
            c=model_config.get("damping", 0.2),
            epsilon=model_config.get("epsilon", 0.02),
            beta=model_config.get("beta", 0.8),
            threshold_fraction=threshold_fraction,
        )
        
        click.echo(f"\n{'═' * 60}")
        click.echo(f"  Model Parameters:")
        click.echo(f"{'─' * 60}")
        click.echo(f"  Damping (c)          = {model.params['c']}")
        click.echo(f"  Timescale (ε)        = {model.params['epsilon']}")
        click.echo(f"  Feedback (β)         = {model.params['beta']}")
        click.echo(f"  threshold_fraction   = {threshold_fraction}")
        click.echo(f"  (z_crit computed as threshold_fraction × max(A_normalized))")
        click.echo(f"{'═' * 60}")
        
        end_step(success=True)
        
        # Determine what to run
        start_step("Determine scenarios to run")
        
        scenarios_to_run = []
        use_custom_forcing = False
        
        if all_scenarios:
            scenarios_to_run = list(SCENARIOS.keys())
            click.echo(f"\nRunning all {len(scenarios_to_run)} scenarios")
        elif scenario:
            scenarios_to_run = [scenario]
            click.echo(f"\nRunning scenario: {scenario}")
        elif forcing_file:
            # Custom forcing from CLI or config
            scenarios_to_run = ["custom"]
            use_custom_forcing = True
            click.echo(f"\nUsing custom forcing: {forcing_file}")
        else:
            default_scenario = cfg.get("scenarios", {}).get("default", "ssp245")
            scenarios_to_run = [default_scenario]
            click.echo(f"\nRunning default scenario: {default_scenario}")
        
        end_step(success=True)
        
        # Get simulation parameters from config
        initial_state = sim_config.get("initial_state", [0.05, 0.0, 0.3])
        noise_level = sim_config.get("noise_level", 0.03)
        noise_smoothing = sim_config.get("noise_smoothing", 15.0)
        rtol = sim_config.get("rtol", 1e-10)
        atol = sim_config.get("atol", 1e-12)
        method = sim_config.get("method", "RK45")
        
        # Run Simulations
        for scenario_key in tqdm(scenarios_to_run, desc="Processing scenarios"):
            
            start_step(f"Scenario: {scenario_key}")
            
            try:
                click.echo(f"\n{'─' * 60}")
                
                if scenario_key == "custom" and use_custom_forcing:
                    # Use custom forcing file
                    custom_name = scenario_meta.get("name", "Custom Forcing")
                    custom_subtitle = scenario_meta.get("subtitle", "")
                    click.echo(f"  Processing: {custom_name}")
                    if custom_subtitle:
                        click.echo(f"  {custom_subtitle}")
                    
                    start_step("Load custom forcing")
                    times, values = load_forcing_csv(forcing_file)
                    end_step(success=True)
                    
                    results = model.run(
                        forcing=values,
                        forcing_times=times,
                        t_end=t_end,
                        n_points=n_points,
                        initial_state=tuple(initial_state),
                        add_noise=add_noise,
                        noise_level=noise_level,
                        noise_smoothing=noise_smoothing,
                        rtol=rtol,
                        atol=atol,
                        method=method,
                        seed=seed,
                        show_progress=True,
                    )
                    
                    # Inject custom scenario metadata into results
                    if scenario_meta.get("name"):
                        viz_config = cfg.get("visualization", {})
                        results.scenario_info = {
                            "name": scenario_meta.get("name", "Custom"),
                            "subtitle": scenario_meta.get("subtitle", ""),
                            "description": scenario_meta.get("description", ""),
                            "expected_outcome": scenario_meta.get("expected_outcome", "N/A"),
                            "color_primary": viz_config.get("color_primary", "#00FFFF"),
                            "color_secondary": viz_config.get("color_secondary", "#FFFFFF"),
                            "color_trajectory": viz_config.get("color_trajectory", "#00FFFF"),
                            "color_forcing": viz_config.get("color_forcing", "#88FF00"),
                            "cmap_name": viz_config.get("cmap_name", "viridis"),
                        }
                        results.scenario_key = scenario_meta.get("key", "custom")
                    
                    base_name = scenario_meta.get("key") or Path(forcing_file).stem
                    
                else:
                    # Use built-in scenario
                    info = get_scenario(scenario_key)
                    
                    click.echo(f"  Processing: {info['name']}")
                    click.echo(f"  {info['subtitle']}")
                    
                    results = model.run(
                        scenario=scenario_key,
                        t_end=t_end,
                        n_points=n_points,
                        initial_state=tuple(initial_state),
                        add_noise=add_noise,
                        noise_level=noise_level,
                        noise_smoothing=noise_smoothing,
                        rtol=rtol,
                        atol=atol,
                        method=method,
                        seed=seed,
                        show_progress=True,
                    )
                    base_name = scenario_key
                
                # Show computed parameters
                click.echo(f"\n  Computed threshold: z_crit = {results.z_crit:.3f}")
                click.echo(f"  Forcing scale: A_scale = {results.A_scale:.2f} W/m²")
                click.echo(f"  Max normalized forcing: {results.A_normalized.max():.3f}")
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
                click.echo(f"    threshold_fraction: {summary['threshold_fraction']:.2f}")
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
        click.echo(f"    Expected outcome: {info['expected_outcome']}")
    
    click.echo("\n" + "─" * 70)
    click.echo("\nz_crit is computed from forcing data, controlled by --threshold-fraction")
    click.echo("  Lower values = tips earlier (more sensitive)")
    click.echo("  Higher values = tips later (less sensitive)")
    click.echo("\nExamples:")
    click.echo("  erucakra run --scenario ssp245 --threshold-fraction 0.6  # More sensitive")
    click.echo("  erucakra run --scenario ssp245 --threshold-fraction 0.85 # Less sensitive")
    click.echo("  erucakra run --config ./configs/custom.yaml              # Use config file")
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
    click.echo(f"Expected outcome: {scenario_info['expected_outcome']}")
    click.echo(f"\nVisualization:")
    click.echo(f"  Primary Color: {scenario_info['color_primary']}")
    click.echo(f"  Secondary Color: {scenario_info['color_secondary']}")
    click.echo(f"  Colormap: {scenario_info['cmap_name']}")
    click.echo(f"\nNote: z_crit is computed at runtime as:")
    click.echo(f"  z_crit = threshold_fraction × max(A_normalized)")
    click.echo(f"\nUse --threshold-fraction to control sensitivity (default: 0.7)")
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
def sensitivity(scenario, tf_min, tf_max, n_samples, output_dir, log_dir):
    """Run sensitivity analysis over threshold_fraction values."""
    import numpy as np
    import pandas as pd
    import traceback
    
    config = DEFAULT_CONFIG.copy()
    
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
            click.echo(f"  Below {critical_tf:.2f}: System tips")
            click.echo(f"  Above {critical_tf:.2f}: System remains stable")
        elif len(tipped) == len(df):
            click.echo(f"\nSystem tips for all tested threshold_fraction values")
        elif len(not_tipped) == len(df):
            click.echo(f"\nSystem remains stable for all tested threshold_fraction values")
        
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
