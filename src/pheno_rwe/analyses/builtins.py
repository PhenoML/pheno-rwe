"""Built-in analysis instances, imported lazily by the registry."""

from pheno_rwe.analyses.causal_effect import ANALYSIS as causal_effect
from pheno_rwe.analyses.cohort_compare import ANALYSIS as cohort_compare
from pheno_rwe.analyses.incidence_rate import ANALYSIS as incidence_rate
from pheno_rwe.analyses.patient_signature import ANALYSIS as patient_signature
from pheno_rwe.analyses.survival import ANALYSIS as survival
from pheno_rwe.analyses.table_one import ANALYSIS as table_one
from pheno_rwe.analyses.trajectory import ANALYSIS as trajectory
from pheno_rwe.analyses.treatment_pathways import ANALYSIS as treatment_pathways

BUILTIN_ANALYSES = (
    table_one,
    cohort_compare,
    survival,
    incidence_rate,
    treatment_pathways,
    trajectory,
    patient_signature,
    causal_effect,
)
