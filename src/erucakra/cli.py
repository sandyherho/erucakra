"""
Command-line interface for erucakra.

Usage:
    erucakra run --scenario ssp245
    erucakra run --scenario ssp245 --z-crit 0.85
    erucakra run --all-scenarios
    erucakra run --forcing ./forcing.csv --z-crit 0.80 --a-scale 5.0
    erucakra list
    erucakra info ssp126
    erucakra sensitivity --scenario ssp245 --z-crit-range 0.5 1.2
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
from erucakra.core.dynamics import DEFAULT_Z_CRIT, FORCING_SCALES


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
    
    The model exhibits a pitchfork bifurcation when the slow variable z
    crosses z_crit. Users can tune z_crit to explore different tipping
    threshold sensitivities.
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
    "--z-crit", "-z",
    type=float,
    default=None,
    help="Critical threshold for tipping (default: scenario-specific)",
)
@click.option(
    "--a-scale",
    type=float,
    default=None,
    help="Forcing normalization scale in W/m² (default: scenario-specific)",
)
@click.option(
    "--t-end",
    type=float,
    default=600.0,
    help="End time (normalized). Default: 600.0 (~480 years to 2500)",
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
def run(ctx, scenario, all_scenarios, forcing, output_dir, outputs, z_crit, a_scale,
        t_end, n_points, no_noise, seed, log_dir, experiment_name):
    """Run climate tipping point simulation with comprehensive logging."""
    from tqdm import tqdm
    import traceback
    
    config = ctx.obj["config"]
    verbose = ctx.obj.get("verbose", False)
    debug = ctx.obj.get("debug", False)
    
    output_dir = Path(output_dir)
    outputs = list(outputs) if outputs else ["csv", "netcdf", "gif", "png"]
    
    # Experiment name
    if experiment_name is None:
        if all_scenarios:
            experiment_name = "all_scenarios"
        elif scenario:
            experiment_name = scenario
        elif forcing:
            experiment_name = Path(forcing).stem
        else:
            experiment_name = config["scenarios"]["default"]
    
    # Setup logging FIRST - this ensures errors are captured
    # NO timestamp in filename
    level = "DEBUG" if debug else ("INFO" if verbose else "INFO")
    logger = setup_logging(
        level=level,
        log_dir=log_dir,
        experiment_name=experiment_name,
        format_style="detailed",
        always_save=True,
        include_timestamp=False,  # No timestamp in filename
    )
    
    # Wrap everything in try-except to ensure log is saved on error
    try:
        # =====================================================================
        # STEP: Initialize Output Directories
        # =====================================================================
        start_step("Initialize output directories")
        
        subdirs = {fmt: output_dir / fmt for fmt in outputs}
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {subdir}")
        
        end_step(success=True)
        
        # =====================================================================
        # STEP: Initialize Model
        # =====================================================================
        start_step("Initialize model")
        
        model = ClimateModel(
            c=config["model"]["damping"],
            epsilon=config["model"]["epsilon"],
            beta=config["model"]["beta"],
            z_crit=z_crit,
        )
        
        click.echo(f"\n{'═' * 60}")
        click.echo(f"  Model Parameters:")
        click.echo(f"{'─' * 60}")
        click.echo(f"  Damping (c)     = {model.params['c']}")
        click.echo(f"  Timescale (ε)   = {model.params['epsilon']}")
        click.echo(f"  Feedback (β)    = {model.params['beta']}")
        
        if z_crit is not None:
            click.echo(f"  z_crit          = {z_crit} (user-specified)")
        else:
            click.echo(f"  z_crit          = scenario-specific (see below)")
        
        if a_scale is not None:
            click.echo(f"  A_scale         = {a_scale} W/m² (user-specified)")
        else:
            click.echo(f"  A_scale         = scenario-specific (see below)")
        click.echo(f"{'═' * 60}")
        
        end_step(success=True)
        
        # =====================================================================
        # STEP: Determine Scenarios
        # =====================================================================
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
        
        # Show default thresholds
        click.echo(f"\n  Default z_crit values by scenario:")
        for key, val in DEFAULT_Z_CRIT.items():
            scale = FORCING_SCALES.get(key, 5.0)
            click.echo(f"    {key}: z_crit={val:.2f}, A_scale={scale:.1f} W/m²")
        click.echo()
        
        logger.info(f"Scenarios to run: {scenarios_to_run}")
        
        end_step(success=True)
        
        # =====================================================================
        # Run Simulations
        # =====================================================================
        for scenario_key in tqdm(scenarios_to_run, desc="Processing scenarios"):
            
            start_step(f"Scenario: {scenario_key}")
            
            try:
                click.echo(f"\n{'─' * 60}")
                
                if scenario_key == "custom":
                    click.echo(f"  Processing: Custom Forcing")
                    
                    start_step("Load custom forcing")
                    times, values = load_forcing_csv(forcing)
                    end_step(success=True)
                    
                    effective_z_crit = z_crit if z_crit else DEFAULT_Z_CRIT["custom"]
                    effective_a_scale = a_scale if a_scale else FORCING_SCALES["custom"]
                    
                    results = model.run(
                        forcing=values,
                        forcing_times=times,
                        t_end=t_end,
                        n_points=n_points,
                        add_noise=not no_noise,
                        seed=seed,
                        show_progress=True,
                        z_crit_override=effective_z_crit,
                        A_scale_override=effective_a_scale,
                    )
                    base_name = Path(forcing).stem
                else:
                    info = get_scenario(scenario_key)
                    effective_z_crit = z_crit if z_crit else DEFAULT_Z_CRIT.get(scenario_key, 0.8)
                    effective_a_scale = a_scale if a_scale else FORCING_SCALES.get(scenario_key, 5.0)
                    
                    click.echo(f"  Processing: {info['name']}")
                    click.echo(f"  {info['subtitle']}")
                    click.echo(f"  Expected: {info['expected_outcome']}")
                    click.echo(f"  z_crit = {effective_z_crit:.2f}, A_scale = {effective_a_scale:.1f} W/m²")
                    
                    results = model.run(
                        scenario=scenario_key,
                        t_end=t_end,
                        n_points=n_points,
                        add_noise=not no_noise,
                        seed=seed,
                        show_progress=True,
                        z_crit_override=z_crit,
                        A_scale_override=a_scale,
                    )
                    base_name = scenario_key
                
                click.echo("─" * 60)
                
                # =============================================================
                # Generate Outputs
                # =============================================================
                start_step(f"Generate outputs: {base_name}")
                
                click.echo("  Generating outputs...")
                
                if "csv" in outputs:
                    start_step(f"Export CSV: {base_name}")
                    csv_path = subdirs["csv"] / f"{base_name}_data.csv"
                    results.to_csv(csv_path)
                    click.echo(f"    ✓ CSV: {csv_path}")
                    end_step(success=True)
                
                if "netcdf" in outputs:
                    start_step(f"Export NetCDF: {base_name}")
                    nc_path = subdirs["netcdf"] / f"{base_name}_data.nc"
                    results.to_netcdf(nc_path)
                    click.echo(f"    ✓ NetCDF: {nc_path}")
                    end_step(success=True)
                
                if "png" in outputs:
                    start_step(f"Export PNG: {base_name}")
                    png_path = subdirs["png"] / f"{base_name}_timeseries.png"
                    results.to_png(png_path)
                    click.echo(f"    ✓ PNG: {png_path}")
                    end_step(success=True)
                
                if "gif" in outputs:
                    start_step(f"Export GIF: {base_name}")
                    gif_path = subdirs["gif"] / f"{base_name}_phase_space.gif"
                    results.to_gif(gif_path)
                    click.echo(f"    ✓ GIF: {gif_path}")
                    end_step(success=True)
                
                end_step(success=True)  # End generate outputs
                
                # Summary
                summary = results.summary()
                click.echo(f"\n  Results Summary:")
                click.echo(f"    z_crit: {summary['z_crit']:.2f}")
                click.echo(f"    Max z: {summary['max_z']:.3f}")
                click.echo(f"    Final z: {summary['final_z']:.3f}")
                click.echo(f"    Crossed threshold: {summary['crossed_threshold']}")
                click.echo(f"    Time above threshold: {summary['time_above_threshold_pct']:.1f}%")
                if summary['first_crossing_year']:
                    click.echo(f"    First crossing: Year {int(summary['first_crossing_year'])}")
                
                logger.info(f"Scenario {scenario_key} completed successfully")
                end_step(success=True)  # End scenario
                
            except Exception as e:
                log_error(e, f"Scenario {scenario_key}")
                end_step(success=False)
                click.echo(f"\n  ✗ ERROR in scenario {scenario_key}: {e}", err=True)
                # Continue with other scenarios
                continue
        
        click.echo(f"\n{'═' * 70}")
        click.echo("  COMPLETE - All outputs generated successfully!")
        click.echo(f"{'═' * 70}")
        click.echo(f"\nOutput Directory: {output_dir}")
        click.echo(f"Log Directory: {log_dir}")
        
        # Print timing summary to console
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
    """List available scenarios with their default thresholds."""
    click.echo("\nAvailable Scenarios:")
    click.echo("─" * 70)
    
    for key, info in SCENARIOS.items():
        z_crit = DEFAULT_Z_CRIT.get(key, 0.8)
        a_scale = FORCING_SCALES.get(key, 5.0)
        
        click.echo(f"\n  {key}:")
        click.echo(f"    Name: {info['name']}")
        click.echo(f"    Subtitle: {info['subtitle']}")
        click.echo(f"    Expected Outcome: {info['expected_outcome']}")
        click.echo(f"    Default z_crit: {z_crit:.2f}")
        click.echo(f"    Default A_scale: {a_scale:.1f} W/m²")
        click.echo(f"    Description: {info['description']}")
    
    click.echo("\n" + "─" * 70)
    click.echo("\nTo override defaults, use --z-crit and --a-scale options:")
    click.echo("  erucakra run --scenario ssp245 --z-crit 0.85")
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
    
    z_crit = DEFAULT_Z_CRIT.get(scenario, 0.8)
    a_scale = FORCING_SCALES.get(scenario, 5.0)
    
    click.echo(f"\n{scenario_info['name']}")
    click.echo("=" * 60)
    click.echo(f"Subtitle: {scenario_info['subtitle']}")
    click.echo(f"Expected Outcome: {scenario_info['expected_outcome']}")
    click.echo(f"Description: {scenario_info['description']}")
    click.echo(f"\nThreshold Settings:")
    click.echo(f"  Default z_crit: {z_crit:.2f}")
    click.echo(f"  Default A_scale: {a_scale:.1f} W/m²")
    click.echo(f"\nVisualization:")
    click.echo(f"  Primary Color: {scenario_info['color_primary']}")
    click.echo(f"  Secondary Color: {scenario_info['color_secondary']}")
    click.echo(f"  Colormap: {scenario_info['cmap_name']}")
    click.echo()


@main.command("sensitivity")
@click.option(
    "--scenario", "-s",
    type=click.Choice(["ssp126", "ssp245", "ssp370", "ssp585"]),
    required=True,
    help="Scenario to analyze",
)
@click.option(
    "--z-crit-min",
    type=float,
    default=0.5,
    help="Minimum z_crit value",
)
@click.option(
    "--z-crit-max",
    type=float,
    default=1.2,
    help="Maximum z_crit value",
)
@click.option(
    "--n-samples",
    type=int,
    default=10,
    help="Number of z_crit values to test",
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
def sensitivity(ctx, scenario, z_crit_min, z_crit_max, n_samples, output_dir, log_dir):
    """Run sensitivity analysis over z_crit values."""
    import numpy as np
    import pandas as pd
    import traceback
    
    config = ctx.obj["config"]
    
    # Setup logging - NO timestamp
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
        click.echo(f"z_crit range: [{z_crit_min}, {z_crit_max}]")
        click.echo(f"Samples: {n_samples}")
        click.echo("─" * 50)
        
        end_step(success=True)
        
        start_step("Run sensitivity samples")
        
        results_list = model.sensitivity_analysis(
            scenario=scenario,
            z_crit_range=(z_crit_min, z_crit_max),
            n_samples=n_samples,
            add_noise=False,
        )
        
        end_step(success=True)
        
        # Compile results
        start_step("Compile results")
        
        summary_data = []
        for z_crit, results in results_list:
            if results is not None:
                summary = results.summary()
                summary_data.append({
                    "z_crit": z_crit,
                    "max_z": summary["max_z"],
                    "final_z": summary["final_z"],
                    "crossed": summary["crossed_threshold"],
                    "time_above_pct": summary["time_above_threshold_pct"],
                    "first_crossing": summary["first_crossing_year"],
                })
            else:
                summary_data.append({
                    "z_crit": z_crit,
                    "max_z": np.nan,
                    "final_z": np.nan,
                    "crossed": None,
                    "time_above_pct": np.nan,
                    "first_crossing": None,
                })
        
        df = pd.DataFrame(summary_data)
        
        # Save results
        csv_path = output_dir / f"{scenario}_sensitivity.csv"
        df.to_csv(csv_path, index=False)
        click.echo(f"\nResults saved to: {csv_path}")
        
        # Print summary
        click.echo("\nResults:")
        click.echo(df.to_string(index=False))
        
        # Find critical z_crit where behavior changes
        crossed = df[df["crossed"] == True]
        not_crossed = df[df["crossed"] == False]
        
        if len(crossed) > 0 and len(not_crossed) > 0:
            critical_z = (crossed["z_crit"].max() + not_crossed["z_crit"].min()) / 2
            click.echo(f"\nCritical z_crit (transition): ~{critical_z:.3f}")
        
        end_step(success=True)
        
        # Print timing summary
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
