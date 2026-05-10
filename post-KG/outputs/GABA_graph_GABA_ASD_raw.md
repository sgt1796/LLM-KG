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

## 11. ASD may influence Grm5 through GABAA receptor.

Score: 0.700
Components: mechanism=0.65, evidence=0.39, novelty=1.00, testability=0.75, specificity=0.82

Study idea: Test whether changing or measuring GABAA receptor alters the relationship between ASD and Grm5.

KG path:
- ASD --[regulates]--> GABAA receptor (weight=1, sources=1)
  Evidence: We previously suggested that ALLO regulates episodes of ASD-like behavior by positively modulating the function of GABAA receptors linked to the dopaminergic system [19].
  Paper: 30703109 - Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism.pdf
- GABAA receptor --[encodes]--> Grm5 (weight=1, sources=1)
  Evidence: - Gabrb3 encodes a subunit of the GABAA receptor [72], and Grm5 and Grm7 encode metabotropic glutamate receptors (mGluR5 and mGluR7).
  Paper: 38263132 - TrkB-dependent regulation of molecular signaling across septal cell types.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: regulates followed by encodes.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- GABAA receptor looks measurable or perturbable enough for follow-up work.

## 12. ASDs may influence GABA through MDGA1.

Score: 0.700
Components: mechanism=0.65, evidence=0.39, novelty=1.00, testability=0.75, specificity=0.82

Study idea: Test whether changing or measuring MDGA1 alters the relationship between ASDs and GABA.

KG path:
- ASDs --[associated with]--> MDGA1 (weight=1, sources=1)
  Evidence: Moreover, GABAergic synaptic inhibition is essential for controlling the window of the critical period of plasticity (Andrade-Talavera et al, 2023) and its precocious closure is linked to ASDs (Berger et al, 2013; LeBlanc and Fagiolini, 2011), giving rise t...
  Paper: 41862769 - Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice.pdf
- MDGA1 --[inhibits]--> GABA (weight=1, sources=1)
  Evidence: Abstract MDGA1 reportedly suppresses GABAergic synaptic inhibition and may be associated with schizophrenia.
  Paper: 41862769 - Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: associated with followed by inhibits.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
- MDGA1 looks measurable or perturbable enough for follow-up work.

## 13. GABA may influence ASD through ADHD.

Score: 0.694
Components: mechanism=1.00, evidence=0.39, novelty=0.14, testability=1.00, specificity=1.00

Study idea: Test whether changing or measuring ADHD alters the relationship between GABA and ASD.

KG path:
- GABA --[decreases]--> ADHD (weight=1, sources=1)
  Evidence: The levels of serum glutamate were two times higher and that of GABA were lower in children with ADHD [71, 72].
  Paper: 35524181 - The electroretinogram b-wave amplitude a differential physiological measure for Attention Deficit Hyperactivity Disorder and Autism Spectrum Disorder.pdf
- ADHD --[regulates]--> ASD (weight=3, sources=3)
  Evidence: Although b-timeto-peak and PhNR p72 amplitudes also differentiated ADHD from the ASD and control groups with statistical significance, the b-wave amplitude provided the greatest discrimination, at two flash strengths.
  Paper: 35524181 - The electroretinogram b-wave amplitude a differential physiological measure for Attention Deficit Hyperactivity Disorder and Autism Spectrum Disorder.pdf
Direct connection: GABA --[decreases]--> ASD (weight=6)
Why this is interesting:
- Mechanistic path: decreases followed by regulates.
- Evidence support is 0.39 based on edge weights and source counts.
- A direct endpoint connection exists, so novelty is lower than an indirect-only path.
- ADHD looks measurable or perturbable enough for follow-up work.
- This may be more confirmatory than novel because the endpoint edge is already strong.

## 14. GABA may influence ASD through AMY.

Score: 0.694
Components: mechanism=1.00, evidence=0.39, novelty=0.14, testability=1.00, specificity=1.00

Study idea: Test whether changing or measuring AMY alters the relationship between GABA and ASD.

KG path:
- GABA --[decreases]--> AMY (weight=1, sources=1)
  Evidence: These data corroborate with our previous studies in which BTBR housed in a semi-natural environment showed a decrease in GABA levels in AMY [94].
  Paper: 38632257 - Amygdalar neurotransmission alterations in the BTBR mice model of idiopathic autism.pdf
- AMY --[decreases]--> ASD (weight=1, sources=1)
  Evidence: In regards to ACh content, we found a decrease in PFC and AMY according to neurochemical alterations in the cholinergic pathway observed in a postmortem study involving ASD patients [77].
  Paper: 38632257 - Amygdalar neurotransmission alterations in the BTBR mice model of idiopathic autism.pdf
Direct connection: GABA --[decreases]--> ASD (weight=6)
Why this is interesting:
- Mechanistic path: decreases followed by decreases.
- Evidence support is 0.39 based on edge weights and source counts.
- A direct endpoint connection exists, so novelty is lower than an indirect-only path.
- AMY looks measurable or perturbable enough for follow-up work.
- This may be more confirmatory than novel because the endpoint edge is already strong.

## 15. GABA may influence ASD through KCC2.

Score: 0.694
Components: mechanism=1.00, evidence=0.39, novelty=0.14, testability=1.00, specificity=1.00

Study idea: Test whether changing or measuring KCC2 alters the relationship between GABA and ASD.

KG path:
- GABA --[regulates]--> KCC2 (weight=2, sources=2)
  Evidence: - 1 Percent change of group means relative to neurotypical controls for plasma GABA, KCC2, and C1 (ASD overall; mild-moderate) Table 3 Within-group spearman correlations in boys with ASD and neurotypical controls Sign P value ρ (Correlation) Pair Group Pᵃ 0...
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf
- KCC2 --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - 1 Percent change of group means relative to neurotypical controls for plasma GABA, KCC2, and C1 (ASD overall; mild-moderate) Table 3 Within-group spearman correlations in boys with ASD and neurotypical controls Sign P value ρ (Correlation) Pair Group Pᵃ 0...
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf
Direct connection: GABA --[decreases]--> ASD (weight=6)
Why this is interesting:
- Mechanistic path: regulates followed by regulates.
- Evidence support is 0.39 based on edge weights and source counts.
- A direct endpoint connection exists, so novelty is lower than an indirect-only path.
- KCC2 looks measurable or perturbable enough for follow-up work.
- This may be more confirmatory than novel because the endpoint edge is already strong.

## 16. GABA may influence ASD through NAA.

Score: 0.694
Components: mechanism=1.00, evidence=0.39, novelty=0.14, testability=1.00, specificity=1.00

Study idea: Test whether changing or measuring NAA alters the relationship between GABA and ASD.

KG path:
- GABA --[decreases]--> NAA (weight=1, sources=1)
  Evidence: Lower GABA and NAA concentrations are frequently observed, suggesting that altered brain metabolism, particularly regarding neuronal integrity and excitation/inhibition balance, may be implicated in the pathophysiology of ASD34.
  Paper: 41107264 - Neurometabolic profiles of autism spectrum disorder patients with genetic variants in specific neurotransmission and synaptic genes.pdf
- NAA --[decreases]--> ASD (weight=1, sources=1)
  Evidence: Lower GABA and NAA concentrations are frequently observed, suggesting that altered brain metabolism, particularly regarding neuronal integrity and excitation/inhibition balance, may be implicated in the pathophysiology of ASD34.
  Paper: 41107264 - Neurometabolic profiles of autism spectrum disorder patients with genetic variants in specific neurotransmission and synaptic genes.pdf
Direct connection: GABA --[decreases]--> ASD (weight=6)
Why this is interesting:
- Mechanistic path: decreases followed by decreases.
- Evidence support is 0.39 based on edge weights and source counts.
- A direct endpoint connection exists, so novelty is lower than an indirect-only path.
- NAA looks measurable or perturbable enough for follow-up work.
- This may be more confirmatory than novel because the endpoint edge is already strong.

## 17. GABA may influence ASD through NKCC1.

Score: 0.694
Components: mechanism=1.00, evidence=0.39, novelty=0.14, testability=1.00, specificity=1.00

Study idea: Test whether changing or measuring NKCC1 alters the relationship between GABA and ASD.

KG path:
- GABA --[regulates]--> NKCC1 (weight=1, sources=1)
  Evidence: - 1 Percent change of group means relative to neurotypical controls for plasma GABA, KCC2, and C1 (ASD overall; mild-moderate) Table 3 Within-group spearman correlations in boys with ASD and neurotypical controls Sign P value ρ (Correlation) Pair Group Pᵃ 0...
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf
- NKCC1 --[decreases]--> ASD (weight=1, sources=1)
  Evidence: - Discussion Plasma GABA, KCC2, and NKCC1 were significantly lower in individuals with ASD compared with controls (Table 1), and values tracked clinical severity, with the lowest levels in the severe subgroup.
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf
Direct connection: GABA --[decreases]--> ASD (weight=6)
Why this is interesting:
- Mechanistic path: regulates followed by decreases.
- Evidence support is 0.39 based on edge weights and source counts.
- A direct endpoint connection exists, so novelty is lower than an indirect-only path.
- NKCC1 looks measurable or perturbable enough for follow-up work.
- This may be more confirmatory than novel because the endpoint edge is already strong.

## 18. ASD may influence pharmacoresistance through GABAergic.

Score: 0.690
Components: mechanism=1.00, evidence=0.39, novelty=1.00, testability=0.15, specificity=0.82

Study idea: Test whether changing or measuring GABAergic alters the relationship between ASD and pharmacoresistance.

KG path:
- ASD --[promotes]--> GABAergic (weight=1, sources=1)
  Evidence: Besides, brain organoids derived from induced pluripotent stem cells of patients with ASD facilitate the production of GABAergic inhibitory neurons (Mariani et al. 2015).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABAergic --[promotes]--> pharmacoresistance (weight=1, sources=1)
  Evidence: - Altered expression or function of KCC2 and NKCC1 can destabilize the excitation–inhibition balance, diminish the effect of GABAergic medications, and promote pharmacoresistance.
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: promotes followed by promotes.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.

## 19. ASD may influence DLPFC through GABA+.

Score: 0.659
Components: mechanism=0.65, evidence=0.39, novelty=1.00, testability=0.40, specificity=1.00

Study idea: Test whether changing or measuring GABA+ alters the relationship between ASD and DLPFC.

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[associated with]--> DLPFC (weight=1, sources=1)
  Evidence: To summarize, we find that higher GABA+ levels were associated with faster response times on the tasks, AtDCS significantly reduces GABA+ and is associated with increased brain activation in the DLPFC as compared to sham stimulation.
  Paper: 36316421 - Non-invasive brain stimulation modulates GABAergic activity in neurofibromatosis 1.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: decreases followed by associated with.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.

## 20. ASD may influence empathy concern scale through GABA+.

Score: 0.659
Components: mechanism=0.65, evidence=0.39, novelty=1.00, testability=0.40, specificity=1.00

Study idea: Test whether changing or measuring GABA+ alters the relationship between ASD and empathy concern scale.

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[associated with]--> empathy concern scale (weight=1, sources=1)
  Evidence: - In the current study, we found that the AI GABA+ concentration was associated with the empathy concern scale as well as the personal distress scale, suggesting that the cerebral GABA system might be involved in empathy.
  Paper: 25419976 - Anterior insula GABA levels correlate with emotional aspects of empathy a proton magnetic resonance spectroscopy study.pdf
Direct connection: none found in this graph.
Why this is interesting:
- Mechanistic path: decreases followed by associated with.
- Evidence support is 0.39 based on edge weights and source counts.
- No direct endpoint connection was found, which makes the bridge worth checking.
