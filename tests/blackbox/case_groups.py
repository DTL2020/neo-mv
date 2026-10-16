"""Functional case metadata; never mutate the hashed fixture definitions."""

GROUP_OPERATIONS = {
    "motion": ("super", "analyse", "many", "recalculate_chain", "recalculate_public",
               "scene_chain", "scene_public"),
    "render": ("compensate", "degrain"),
    "mask": ("VectorLengthMask", "SADMask", "OcclusionMask"),
    "flow": ("Flow",),
    "interpolation": ("FlowInter", "FlowFPS", "FlowBlur"),
    "depan": ("DepanAnalyse", "DepanCompensate"),
    "estimate": ("DepanEstimate",),
    "stabilise": ("DepanStabilise",),
}
GROUP_BY_OPERATION = {operation: group for group, operations in GROUP_OPERATIONS.items()
                      for operation in operations}
LEGACY_PHASE_GROUPS = {
    1: ("motion",), 2: ("render",), 3: ("mask", "flow"), 4: ("interpolation",),
    5: ("depan",), 6: ("estimate",), 7: ("stabilise",),
}

# These groups prepare their source before creating a filter and can therefore
# record creation failures without losing the source observations.
PREPARED_GROUPS = frozenset(GROUP_OPERATIONS) - {"motion"}
# Color properties and the complete output property-name inventory are observed.
EXTENDED_OBSERVATION_GROUPS = PREPARED_GROUPS - {"render"}
DEPAN_GROUPS = frozenset(("depan", "estimate", "stabilise"))
# Estimate deliberately records an explicit empty auxiliary inventory.
AUXILIARY_GROUPS = PREPARED_GROUPS - {"mask"}


def group_for(spec):
    operation = spec.get("operation")
    try:
        return GROUP_BY_OPERATION[operation]
    except KeyError:
        raise ValueError(f"unknown black-box operation: {operation!r}") from None
