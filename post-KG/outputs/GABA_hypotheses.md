# Hypothesis candidates from GABA_graph.json

Method:
1. Scan the KG for two-step paths: source -> bridge -> outcome.
2. Keep paths near the requested focus terms, if any were provided.
3. Penalize generic nodes and very common bridge nodes.
4. Score each path for mechanism, evidence, novelty, testability, and specificity.
5. Return hypotheses with the evidence sentences that support each edge.

## 1. ASD may influence GABAB receptor through arbaclofen.

Score: 0.802
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.65, specificity=1.00

Study idea: Test whether changing or measuring arbaclofen alters the relationship between ASD and GABAB receptor.

KG path:
- ASD --[increases]--> arbaclofen (weight=1, sources=1)
  Evidence: - In ASD, since arbaclofen increases repetition suppression, this could indicate that postsynaptic GABAB receptor mechanisms are altered at baseline; but further experimental work in animal models will be needed to test this concept.
  Paper: 37852957 - Exploratory evidence for differences in GABAergic regulation of auditory processing in autism spectrum disorder.pdf
- arbaclofen --[increases]--> GABAB receptor (weight=1, sources=1)
  Evidence: - In ASD, since arbaclofen increases repetition suppression, this could indicate that postsynaptic GABAB receptor mechanisms are altered at baseline; but further experimental work in animal models will be needed to test this concept.
  Paper: 37852957 - Exploratory evidence for differences in GABAergic regulation of auditory processing in autism spectrum disorder.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: increases followed by increases.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- arbaclofen looks measurable or perturbable enough for follow-up work.

## 2. CBD may influence ASD through GABA+.

Score: 0.802
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.65, specificity=1.00

Study idea: Test whether changing or measuring GABA+ alters the relationship between CBD and ASD.

KG path:
- CBD --[regulates]--> GABA+ (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf
- GABA+ --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: regulates followed by regulates.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- GABA+ looks measurable or perturbable enough for follow-up work.

## 3. GABA-A receptor may influence ASD through CYFIP1.

Score: 0.786
Components: mechanism=1.00, evidence=0.23, novelty=1.00, testability=0.75, specificity=1.00

Study idea: Test whether changing or measuring CYFIP1 alters the relationship between GABA-A receptor and ASD.

KG path:
- GABA-A receptor --[decreases]--> CYFIP1 (weight=1, sources=1)
  Evidence: While a decrease in GABA-A receptor subunits has been observed in the cortex of FXS knockout mice [72], a direct connection between CYFIP1 expression and inhibitory synaptic structure and function is just beginning to be explored [73].
  Paper: 31198525 - CYFIP1 overexpression increases fear response in mice but does not affect social or repetitive behavioral phenotypes.pdf
- CYFIP1 --[causes]--> ASD (weight=1, sources=1)
  Evidence: The evidence from two mouse lines overexpressing human CYFIP1 does not support that CYFIP1 overexpression leads to ASD-like behaviors in this mouse model.
  Paper: 31198525 - CYFIP1 overexpression increases fear response in mice but does not affect social or repetitive behavioral phenotypes.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: decreases followed by causes.
- Evidence support is 0.23 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- CYFIP1 looks measurable or perturbable enough for follow-up work.

## 4. ASD may influence Kim through GABA+.

Score: 0.780
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.65, specificity=0.82

Study idea: Test whether changing or measuring GABA+ alters the relationship between ASD and Kim.

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[decreases]--> Kim (weight=1, sources=1)
  Evidence: - Anodal transcranial direct current stimulation (a-tDCS) was reported to affect neurotransmitter levels and reduce GABA+ levels in the cerebral cortex compared with those before the stimulation (Kim et al.
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: decreases followed by decreases.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- GABA+ looks measurable or perturbable enough for follow-up work.

## 5. ASD may influence Stagg through GABA+.

Score: 0.780
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.65, specificity=0.82

Study idea: Test whether changing or measuring GABA+ alters the relationship between ASD and Stagg.

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[decreases]--> Stagg (weight=1, sources=1)
  Evidence: - For example, participants with a lower ratio of GABA+/NAA in M1 tended to show shorter reaction times in a visually cued sequence task performed with four fingers (Stagg et al.
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: decreases followed by decreases.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- GABA+ looks measurable or perturbable enough for follow-up work.

## 6. AtDCS may influence ASD through GABA+.

Score: 0.780
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.65, specificity=0.82

Study idea: Test whether changing or measuring GABA+ alters the relationship between AtDCS and ASD.

KG path:
- AtDCS --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: To summarize, we find that higher GABA+ levels were associated with faster response times on the tasks, AtDCS significantly reduces GABA+ and is associated with increased brain activation in the DLPFC as compared to sham stimulation.
  Paper: 36316421 - Non-invasive brain stimulation modulates GABAergic activity in neurofibromatosis 1.pdf
- GABA+ --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: decreases followed by regulates.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- GABA+ looks measurable or perturbable enough for follow-up work.

## 7. anodal transcranial direct current stimulation may influence ASD through GABA+.

Score: 0.743
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.40, specificity=0.88

Study idea: Test whether changing or measuring GABA+ alters the relationship between anodal transcranial direct current stimulation and ASD.

KG path:
- anodal transcranial direct current stimulation --[causes]--> GABA+ (weight=1, sources=1)
  Evidence: - Anodal transcranial direct current stimulation (a-tDCS) was reported to affect neurotransmitter levels and reduce GABA+ levels in the cerebral cortex compared with those before the stimulation (Kim et al.
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: causes followed by regulates.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.

## 8. ASD may influence PV through GABAergic.

Score: 0.735
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.40, specificity=0.82

Study idea: Test whether changing or measuring GABAergic alters the relationship between ASD and PV.

KG path:
- ASD --[promotes]--> GABAergic (weight=1, sources=1)
  Evidence: Besides, brain organoids derived from induced pluripotent stem cells of patients with ASD facilitate the production of GABAergic inhibitory neurons (Mariani et al. 2015).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABAergic --[increases]--> PV (weight=1, sources=1)
  Evidence: While ASDrelated alterations at glutamatergic synapses have been comprehensively investigated (Galineau et al, 2023; Moretto et al, 2018; Nisar et al, 2022; Ramaswami and Geschwind, 2018), GABAergic pathologies are typically attributed to global deficits, s...
  Paper: 41862769 - Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: promotes followed by increases.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.

## 9. ASD may influence dopaminergic system through GABAA receptor.

Score: 0.722
Components: mechanism=0.65, evidence=0.39, novelty=1.00, testability=0.75, specificity=1.00

Study idea: Test whether changing or measuring GABAA receptor alters the relationship between ASD and dopaminergic system.

KG path:
- ASD --[regulates]--> GABAA receptor (weight=1, sources=1)
  Evidence: We previously suggested that ALLO regulates episodes of ASD-like behavior by positively modulating the function of GABAA receptors linked to the dopaminergic system [19].
  Paper: 30703109 - Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism.pdf
- GABAA receptor --[associated with]--> dopaminergic system (weight=1, sources=1)
  Evidence: We previously suggested that ALLO regulates episodes of ASD-like behavior by positively modulating the function of GABAA receptors linked to the dopaminergic system [19].
  Paper: 30703109 - Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: regulates followed by associated with.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- GABAA receptor looks measurable or perturbable enough for follow-up work.

## 10. ASD may influence M1 through GABA+.

Score: 0.712
Components: mechanism=1.00, evidence=0.39, novelty=0.55, testability=0.65, specificity=1.00

Study idea: Test whether changing or measuring GABA+ alters the relationship between ASD and M1.

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[decreases]--> M1 (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
Direct connection: ASD --[decreases]--> M1 (weight=1)
Why this is interesting:
- Mechanistic path: decreases followed by decreases.
- Evidence support is 0.39 based on edge weights and source counts.
- A direct endpoint connection exists, so novelty is lower than an indirect-only path.
- GABA+ looks measurable or perturbable enough for follow-up work.
