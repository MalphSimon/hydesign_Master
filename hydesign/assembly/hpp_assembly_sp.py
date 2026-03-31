"""SP-based assembly variant.

This module keeps the original assembly untouched while swapping the PV
component to the SP-driven implementation.
"""

import hydesign.assembly.hpp_assembly as base_assembly
from hydesign.pv.pv_sp import pvp_sp_comp

# Replace only the base PV component used by hpp_model construction.
base_assembly.pvp = pvp_sp_comp

# Re-export the same model class name used by existing callers.
hpp_model = base_assembly.hpp_model
