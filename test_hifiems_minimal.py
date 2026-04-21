"""
Minimal test case following official TopFarm HiFiEMS example
"""
from hydesign.examples import examples_filepath
from hydesign.assembly.hpp_assembly_hifi import hpp_model
import os

# Use official example approach
sim_pars_fn = os.path.join(examples_filepath, "Europe/hpp_pars_HiFiEMS.yml")

print(f"Using config: {sim_pars_fn}")
print(f"Config exists: {os.path.exists(sim_pars_fn)}")

hpp = hpp_model(
    sim_pars_fn=sim_pars_fn,
    latitude=43.012,      # Golfe_du_Lion
    longitude=4.294,
    altitude=100,
)

# Test with simple inputs
inputs = dict(
    wind_MW=495,      # Golfe_du_Lion: 33 turbines * 15 MW
    solar_MW=12.18,   # Golfe_du_Lion
    b_P=0,            # Zero battery
    b_E_h=0,
)

print(f"\nEvaluating with inputs: {inputs}")
res = hpp.evaluate(**inputs)

print("\n=== Design Configuration ===")
hpp.print_design()

# Get results using evaluation_in_df method
eval_df = hpp.evaluation_in_df(None, res)
print("\n=== Evaluation Results ===")
for col in ['NPV [MEuro]', 'Revenues [MEuro]', 'CAPEX [MEuro]', 'OPEX [MEuro]', 'IRR', 'LCOE [Euro/MWh]', 'Mean Annual Electricity Sold [GWh]']:
    if col in eval_df.columns:
        print(f"{col}: {eval_df[col].iloc[0]}")
    else:
        print(f"{col}: NOT FOUND")
