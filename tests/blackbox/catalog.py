"""Case inventory; motion fixtures remain independently importable."""
from motion_cases import CASES as MOTION_CASES
from render_cases import CASES as RENDER_CASES
from mask_cases import CASES as MASK_CASES
from flow_cases import CASES as FLOW_CASES
from interpolation_cases import CASES as INTERPOLATION_CASES
from depan_cases import CASES as DEPAN_CASES
from estimate_cases import CASES as ESTIMATE_CASES
from stabilise_cases import CASES as STABILISE_CASES

CASES = (MOTION_CASES + RENDER_CASES + MASK_CASES + FLOW_CASES + INTERPOLATION_CASES +
         DEPAN_CASES + ESTIMATE_CASES + STABILISE_CASES)
BY_ID = {item["id"]: item for item in CASES}
if len(BY_ID) != len(CASES):
    raise ValueError("duplicate black-box case ID")
