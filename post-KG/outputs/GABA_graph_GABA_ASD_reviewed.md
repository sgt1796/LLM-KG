# Reviewed Hypotheses from GABA_graph.json

Review method:
1. Read each raw KG path as a candidate, not as a conclusion.
2. Reject obvious artifacts such as author names, control labels, or unsupported negated claims.
3. Rewrite awkward graph direction into a biologically plausible study hypothesis.
4. Classify the idea as intervention, biomarker, mechanism, subtype, or artifact.
5. Assign a priority and concrete next actions.

## 1. arbaclofen response may reveal altered GABAB receptor biology in ASD.

Decision: advance
Priority: high (0.854)
Category: intervention
Raw: ASD may influence GABAB receptor through arbaclofen. (score=0.802)

Study design: Compare arbaclofen-responsive and non-responsive samples, then test whether GABAB receptor or ASD-relevant phenotypes change with the intervention.

Measurements:
- Primary bridge measure: arbaclofen
- Endpoint measure: GABAB receptor
- Exposure or response measure for arbaclofen
- Manual evidence audit of both KG edges

KG path:
- ASD --[increases]--> arbaclofen (weight=1, sources=1)
  Evidence: - In ASD, since arbaclofen increases repetition suppression, this could indicate that postsynaptic GABAB receptor mechanisms are altered at baseline; but further experimental work in animal models will be needed to test this concept.
  Paper: 37852957 - Exploratory evidence for differences in GABAergic regulation of auditory processing in autism spectrum disorder.pdf
- arbaclofen --[increases]--> GABAB receptor (weight=1, sources=1)
  Evidence: - In ASD, since arbaclofen increases repetition suppression, this could indicate that postsynaptic GABAB receptor mechanisms are altered at baseline; but further experimental work in animal models will be needed to test this concept.
  Paper: 37852957 - Exploratory evidence for differences in GABAergic regulation of auditory processing in autism spectrum disorder.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "arbaclofen response may reveal altered GABAB receptor biology in ASD."
- Search: "ASD" "arbaclofen" "GABAB receptor" autism
- Search: "ASD" "arbaclofen" "increases"
- Search: "arbaclofen" "GABAB receptor" "increases"
- Search: "Exploratory evidence for differences in GABAergic regulation of auditory processing in autism spectrum disorder"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 2. CBD may affect ASD-relevant biology through GABA+.

Decision: advance
Priority: high (0.854)
Category: intervention
Raw: CBD may influence ASD through GABA+. (score=0.802)

Study design: Compare GABA+-responsive and non-responsive samples, then test whether ASD or ASD-relevant phenotypes change with the intervention.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: ASD
- Exposure or response measure for CBD
- Manual evidence audit of both KG edges

KG path:
- CBD --[regulates]--> GABA+ (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf
- GABA+ --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "CBD may affect ASD-relevant biology through GABA+."
- Search: "CBD" "GABA+" "ASD" autism
- Search: "CBD" "GABA+" "regulates"
- Search: "GABA+" "ASD" "regulates"
- Search: "Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 3. AtDCS may affect ASD-relevant biology through GABA+.

Decision: advance
Priority: high (0.832)
Category: intervention
Raw: AtDCS may influence ASD through GABA+. (score=0.780)

Study design: Compare GABA+-responsive and non-responsive samples, then test whether ASD or ASD-relevant phenotypes change with the intervention.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: ASD
- Exposure or response measure for AtDCS
- Manual evidence audit of both KG edges

KG path:
- AtDCS --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: To summarize, we find that higher GABA+ levels were associated with faster response times on the tasks, AtDCS significantly reduces GABA+ and is associated with increased brain activation in the DLPFC as compared to sham stimulation.
  Paper: 36316421 - Non-invasive brain stimulation modulates GABAergic activity in neurofibromatosis 1.pdf
- GABA+ --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "AtDCS may affect ASD-relevant biology through GABA+."
- Search: "AtDCS" "GABA+" "ASD" autism
- Search: "AtDCS" "GABA+" "decreases"
- Search: "GABA+" "ASD" "regulates"
- Search: "Non-invasive brain stimulation modulates GABAergic activity in neurofibromatosis 1"
- Search: "Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 4. GABAA receptor may be a measurable bridge between ASD and dopaminergic system.

Decision: advance
Priority: high (0.782)
Category: biomarker
Raw: ASD may influence dopaminergic system through GABAA receptor. (score=0.722)

Study design: Measure GABAA receptor in independent ASD and control cohorts, then model whether it explains variation between ASD and dopaminergic system.

Measurements:
- Primary bridge measure: GABAA receptor
- Endpoint measure: dopaminergic system
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- ASD --[regulates]--> GABAA receptor (weight=1, sources=1)
  Evidence: We previously suggested that ALLO regulates episodes of ASD-like behavior by positively modulating the function of GABAA receptors linked to the dopaminergic system [19].
  Paper: 30703109 - Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism.pdf
- GABAA receptor --[associated with]--> dopaminergic system (weight=1, sources=1)
  Evidence: We previously suggested that ALLO regulates episodes of ASD-like behavior by positively modulating the function of GABAA receptors linked to the dopaminergic system [19].
  Paper: 30703109 - Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "GABAA receptor may be a measurable bridge between ASD and dopaminergic system."
- Search: "ASD" "GABAA receptor" "dopaminergic system" autism
- Search: "ASD" "GABAA receptor" "regulates"
- Search: "GABAA receptor" "dopaminergic system" "associated with"
- Search: "Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 5. anodal transcranial direct current stimulation may affect ASD-relevant biology through GABA+.

Decision: advance
Priority: medium (0.775)
Category: intervention
Raw: anodal transcranial direct current stimulation may influence ASD through GABA+. (score=0.743)

Study design: Compare GABA+-responsive and non-responsive samples, then test whether ASD or ASD-relevant phenotypes change with the intervention.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: ASD
- Exposure or response measure for anodal transcranial direct current stimulation
- Manual evidence audit of both KG edges

KG path:
- anodal transcranial direct current stimulation --[causes]--> GABA+ (weight=1, sources=1)
  Evidence: - Anodal transcranial direct current stimulation (a-tDCS) was reported to affect neurotransmitter levels and reduce GABA+ levels in the cerebral cortex compared with those before the stimulation (Kim et al.
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - Speciﬁcally, both in prefrontal and subcortical regions, CBD increased GABA+ in the controls but decreased GABA+ in ASD.
  Paper: 30758329 - Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.
- The path is not obviously measurable or perturbable.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "anodal transcranial direct current stimulation may affect ASD-relevant biology through GABA+."
- Search: "anodal transcranial direct current stimulation" "GABA+" "ASD" autism
- Search: "anodal transcranial direct current stimulation" "GABA+" "causes"
- Search: "GABA+" "ASD" "regulates"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Search: "Effects of cannabidiol on brain excitation and inhibition systems; a randomised placebo-controlled single dose trial during magnetic resonance spectroscopy in adults wit"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 6. KCC2 may be a measurable bridge between GABA and ASD.

Decision: advance
Priority: medium (0.774)
Category: biomarker
Raw: GABA may influence ASD through KCC2. (score=0.694)

Study design: Measure KCC2 in independent ASD and control cohorts, then model whether it explains variation between GABA and ASD.

Measurements:
- Primary bridge measure: KCC2
- Endpoint measure: ASD
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- GABA --[regulates]--> KCC2 (weight=2, sources=2)
  Evidence: - 1 Percent change of group means relative to neurotypical controls for plasma GABA, KCC2, and C1 (ASD overall; mild-moderate) Table 3 Within-group spearman correlations in boys with ASD and neurotypical controls Sign P value ρ (Correlation) Pair Group Pᵃ 0...
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf
- KCC2 --[regulates]--> ASD (weight=1, sources=1)
  Evidence: - 1 Percent change of group means relative to neurotypical controls for plasma GABA, KCC2, and C1 (ASD overall; mild-moderate) Table 3 Within-group spearman correlations in boys with ASD and neurotypical controls Sign P value ρ (Correlation) Pair Group Pᵃ 0...
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "KCC2 may be a measurable bridge between GABA and ASD."
- Search: "GABA" "KCC2" "ASD" autism
- Search: "GABA" "KCC2" "regulates"
- Search: "KCC2" "ASD" "regulates"
- Search: "Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 7. NKCC1 may be a measurable bridge between GABA and ASD.

Decision: advance
Priority: medium (0.774)
Category: biomarker
Raw: GABA may influence ASD through NKCC1. (score=0.694)

Study design: Measure NKCC1 in independent ASD and control cohorts, then model whether it explains variation between GABA and ASD.

Measurements:
- Primary bridge measure: NKCC1
- Endpoint measure: ASD
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- GABA --[regulates]--> NKCC1 (weight=1, sources=1)
  Evidence: - 1 Percent change of group means relative to neurotypical controls for plasma GABA, KCC2, and C1 (ASD overall; mild-moderate) Table 3 Within-group spearman correlations in boys with ASD and neurotypical controls Sign P value ρ (Correlation) Pair Group Pᵃ 0...
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf
- NKCC1 --[decreases]--> ASD (weight=1, sources=1)
  Evidence: - Discussion Plasma GABA, KCC2, and NKCC1 were significantly lower in individuals with ASD compared with controls (Table 1), and values tracked clinical severity, with the lowest levels in the severe subgroup.
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "NKCC1 may be a measurable bridge between GABA and ASD."
- Search: "GABA" "NKCC1" "ASD" autism
- Search: "GABA" "NKCC1" "regulates"
- Search: "NKCC1" "ASD" "decreases"
- Search: "Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 8. GABA+ may be a measurable bridge between ASD and M1.

Decision: advance
Priority: medium (0.764)
Category: biomarker
Raw: ASD may influence M1 through GABA+. (score=0.712)

Study design: Measure GABA+ in independent ASD and control cohorts, then model whether it explains variation between ASD and M1.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: M1
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[decreases]--> M1 (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "GABA+ may be a measurable bridge between ASD and M1."
- Search: "ASD" "GABA+" "M1" autism
- Search: "ASD" "GABA+" "decreases"
- Search: "GABA+" "M1" "decreases"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 9. GABAA receptor may be a measurable bridge between ASD and Grm5.

Decision: advance
Priority: medium (0.760)
Category: biomarker
Raw: ASD may influence Grm5 through GABAA receptor. (score=0.700)

Study design: Measure GABAA receptor in independent ASD and control cohorts, then model whether it explains variation between ASD and Grm5.

Measurements:
- Primary bridge measure: GABAA receptor
- Endpoint measure: Grm5
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- ASD --[regulates]--> GABAA receptor (weight=1, sources=1)
  Evidence: We previously suggested that ALLO regulates episodes of ASD-like behavior by positively modulating the function of GABAA receptors linked to the dopaminergic system [19].
  Paper: 30703109 - Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism.pdf
- GABAA receptor --[encodes]--> Grm5 (weight=1, sources=1)
  Evidence: - Gabrb3 encodes a subunit of the GABAA receptor [72], and Grm5 and Grm7 encode metabotropic glutamate receptors (mGluR5 and mGluR7).
  Paper: 38263132 - TrkB-dependent regulation of molecular signaling across septal cell types.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "GABAA receptor may be a measurable bridge between ASD and Grm5."
- Search: "ASD" "GABAA receptor" "Grm5" autism
- Search: "ASD" "GABAA receptor" "regulates"
- Search: "GABAA receptor" "Grm5" "encodes"
- Search: "Kami-shoyo-san improves ASD-like behaviors caused by decreasing allopregnanolone biosynthesis in an SKF mouse model of autism"
- Search: "TrkB-dependent regulation of molecular signaling across septal cell types"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 10. MDGA1 may be a measurable bridge between ASDs and GABA.

Decision: advance
Priority: medium (0.760)
Category: biomarker
Raw: ASDs may influence GABA through MDGA1. (score=0.700)

Study design: Measure MDGA1 in independent ASD and control cohorts, then model whether it explains variation between ASDs and GABA.

Measurements:
- Primary bridge measure: MDGA1
- Endpoint measure: GABA
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- ASDs --[associated with]--> MDGA1 (weight=1, sources=1)
  Evidence: Moreover, GABAergic synaptic inhibition is essential for controlling the window of the critical period of plasticity (Andrade-Talavera et al, 2023) and its precocious closure is linked to ASDs (Berger et al, 2013; LeBlanc and Fagiolini, 2011), giving rise t...
  Paper: 41862769 - Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice.pdf
- MDGA1 --[inhibits]--> GABA (weight=1, sources=1)
  Evidence: Abstract MDGA1 reportedly suppresses GABAergic synaptic inhibition and may be associated with schizophrenia.
  Paper: 41862769 - Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "MDGA1 may be a measurable bridge between ASDs and GABA."
- Search: "ASDs" "MDGA1" "GABA" autism
- Search: "ASDs" "MDGA1" "associated with"
- Search: "MDGA1" "GABA" "inhibits"
- Search: "Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 11. ADHD may mediate the relationship between GABA and ASD.

Decision: advance
Priority: medium (0.694)
Category: mechanism
Raw: GABA may influence ASD through ADHD. (score=0.694)

Study design: Perturb or stratify by ADHD, then measure whether the GABA to ASD relationship changes.

Measurements:
- Primary bridge measure: ADHD
- Endpoint measure: ASD
- Manual evidence audit of both KG edges

KG path:
- GABA --[decreases]--> ADHD (weight=1, sources=1)
  Evidence: The levels of serum glutamate were two times higher and that of GABA were lower in children with ADHD [71, 72].
  Paper: 35524181 - The electroretinogram b-wave amplitude a differential physiological measure for Attention Deficit Hyperactivity Disorder and Autism Spectrum Disorder.pdf
- ADHD --[regulates]--> ASD (weight=3, sources=3)
  Evidence: Although b-timeto-peak and PhNR p72 amplitudes also differentiated ADHD from the ASD and control groups with statistical significance, the b-wave amplitude provided the greatest discrimination, at two flash strengths.
  Paper: 35524181 - The electroretinogram b-wave amplitude a differential physiological measure for Attention Deficit Hyperactivity Disorder and Autism Spectrum Disorder.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.

Agent web audit:
- Search: "ADHD may mediate the relationship between GABA and ASD."
- Search: "GABA" "ADHD" "ASD" autism
- Search: "GABA" "ADHD" "decreases"
- Search: "ADHD" "ASD" "regulates"
- Search: "The electroretinogram b-wave amplitude a differential physiological measure for Attention Deficit Hyperactivity Disorder and Autism Spectrum Disorder"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 12. AMY may mediate the relationship between GABA and ASD.

Decision: advance
Priority: medium (0.694)
Category: mechanism
Raw: GABA may influence ASD through AMY. (score=0.694)

Study design: Perturb or stratify by AMY, then measure whether the GABA to ASD relationship changes.

Measurements:
- Primary bridge measure: AMY
- Endpoint measure: ASD
- Manual evidence audit of both KG edges

KG path:
- GABA --[decreases]--> AMY (weight=1, sources=1)
  Evidence: These data corroborate with our previous studies in which BTBR housed in a semi-natural environment showed a decrease in GABA levels in AMY [94].
  Paper: 38632257 - Amygdalar neurotransmission alterations in the BTBR mice model of idiopathic autism.pdf
- AMY --[decreases]--> ASD (weight=1, sources=1)
  Evidence: In regards to ACh content, we found a decrease in PFC and AMY according to neurochemical alterations in the cholinergic pathway observed in a postmortem study involving ASD patients [77].
  Paper: 38632257 - Amygdalar neurotransmission alterations in the BTBR mice model of idiopathic autism.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.

Agent web audit:
- Search: "AMY may mediate the relationship between GABA and ASD."
- Search: "GABA" "AMY" "ASD" autism
- Search: "GABA" "AMY" "decreases"
- Search: "AMY" "ASD" "decreases"
- Search: "Amygdalar neurotransmission alterations in the BTBR mice model of idiopathic autism"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 13. NAA may mediate the relationship between GABA and ASD.

Decision: advance
Priority: medium (0.694)
Category: mechanism
Raw: GABA may influence ASD through NAA. (score=0.694)

Study design: Perturb or stratify by NAA, then measure whether the GABA to ASD relationship changes.

Measurements:
- Primary bridge measure: NAA
- Endpoint measure: ASD
- Manual evidence audit of both KG edges

KG path:
- GABA --[decreases]--> NAA (weight=1, sources=1)
  Evidence: Lower GABA and NAA concentrations are frequently observed, suggesting that altered brain metabolism, particularly regarding neuronal integrity and excitation/inhibition balance, may be implicated in the pathophysiology of ASD34.
  Paper: 41107264 - Neurometabolic profiles of autism spectrum disorder patients with genetic variants in specific neurotransmission and synaptic genes.pdf
- NAA --[decreases]--> ASD (weight=1, sources=1)
  Evidence: Lower GABA and NAA concentrations are frequently observed, suggesting that altered brain metabolism, particularly regarding neuronal integrity and excitation/inhibition balance, may be implicated in the pathophysiology of ASD34.
  Paper: 41107264 - Neurometabolic profiles of autism spectrum disorder patients with genetic variants in specific neurotransmission and synaptic genes.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.

Agent web audit:
- Search: "NAA may mediate the relationship between GABA and ASD."
- Search: "GABA" "NAA" "ASD" autism
- Search: "GABA" "NAA" "decreases"
- Search: "NAA" "ASD" "decreases"
- Search: "Neurometabolic profiles of autism spectrum disorder patients with genetic variants in specific neurotransmission and synaptic genes"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 14. GABA+ may be a measurable bridge between ASD and DLPFC.

Decision: advance
Priority: medium (0.691)
Category: biomarker
Raw: ASD may influence DLPFC through GABA+. (score=0.659)

Study design: Measure GABA+ in independent ASD and control cohorts, then model whether it explains variation between ASD and DLPFC.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: DLPFC
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[associated with]--> DLPFC (weight=1, sources=1)
  Evidence: To summarize, we find that higher GABA+ levels were associated with faster response times on the tasks, AtDCS significantly reduces GABA+ and is associated with increased brain activation in the DLPFC as compared to sham stimulation.
  Paper: 36316421 - Non-invasive brain stimulation modulates GABAergic activity in neurofibromatosis 1.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.
- The path is not obviously measurable or perturbable.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "GABA+ may be a measurable bridge between ASD and DLPFC."
- Search: "ASD" "GABA+" "DLPFC" autism
- Search: "ASD" "GABA+" "decreases"
- Search: "GABA+" "DLPFC" "associated with"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Search: "Non-invasive brain stimulation modulates GABAergic activity in neurofibromatosis 1"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 15. GABA+ may be a measurable bridge between ASD and empathy concern scale.

Decision: advance
Priority: medium (0.691)
Category: biomarker
Raw: ASD may influence empathy concern scale through GABA+. (score=0.659)

Study design: Measure GABA+ in independent ASD and control cohorts, then model whether it explains variation between ASD and empathy concern scale.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: empathy concern scale
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[associated with]--> empathy concern scale (weight=1, sources=1)
  Evidence: - In the current study, we found that the AI GABA+ concentration was associated with the empathy concern scale as well as the personal distress scale, suggesting that the cerebral GABA system might be involved in empathy.
  Paper: 25419976 - Anterior insula GABA levels correlate with emotional aspects of empathy a proton magnetic resonance spectroscopy study.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.
- The path is not obviously measurable or perturbable.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "GABA+ may be a measurable bridge between ASD and empathy concern scale."
- Search: "ASD" "GABA+" "empathy concern scale" autism
- Search: "ASD" "GABA+" "decreases"
- Search: "GABA+" "empathy concern scale" "associated with"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Search: "Anterior insula GABA levels correlate with emotional aspects of empathy a proton magnetic resonance spectroscopy study"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 16. GABAergic may mediate the relationship between ASD and PV.

Decision: advance
Priority: medium (0.687)
Category: mechanism
Raw: ASD may influence PV through GABAergic. (score=0.735)

Study design: Perturb or stratify by GABAergic, then measure whether the ASD to PV relationship changes.

Measurements:
- Primary bridge measure: GABAergic
- Endpoint measure: PV
- Manual evidence audit of both KG edges

KG path:
- ASD --[promotes]--> GABAergic (weight=1, sources=1)
  Evidence: Besides, brain organoids derived from induced pluripotent stem cells of patients with ASD facilitate the production of GABAergic inhibitory neurons (Mariani et al. 2015).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABAergic --[increases]--> PV (weight=1, sources=1)
  Evidence: While ASDrelated alterations at glutamatergic synapses have been comprehensively investigated (Galineau et al, 2023; Moretto et al, 2018; Nisar et al, 2022; Ramaswami and Geschwind, 2018), GABAergic pathologies are typically attributed to global deficits, s...
  Paper: 41862769 - Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.
- The path is not obviously measurable or perturbable.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.

Agent web audit:
- Search: "GABAergic may mediate the relationship between ASD and PV."
- Search: "ASD" "GABAergic" "PV" autism
- Search: "ASD" "GABAergic" "promotes"
- Search: "GABAergic" "PV" "increases"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Search: "Bazedoxifene reverses sexually dimorphic autistic-like abnormalities in biallelic MDGA1-mutant mice"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 17. CYFIP1 may be a measurable bridge between GABA-A receptor and ASD.

Decision: needs manual evidence check
Priority: medium (0.666)
Category: biomarker
Raw: GABA-A receptor may influence ASD through CYFIP1. (score=0.786)

Study design: Measure CYFIP1 in independent ASD and control cohorts, then model whether it explains variation between GABA-A receptor and ASD.

Measurements:
- Primary bridge measure: CYFIP1
- Endpoint measure: ASD
- Replication in an independent cohort or model system
- Manual evidence audit of both KG edges

KG path:
- GABA-A receptor --[decreases]--> CYFIP1 (weight=1, sources=1)
  Evidence: While a decrease in GABA-A receptor subunits has been observed in the cortex of FXS knockout mice [72], a direct connection between CYFIP1 expression and inhibitory synaptic structure and function is just beginning to be explored [73].
  Paper: 31198525 - CYFIP1 overexpression increases fear response in mice but does not affect social or repetitive behavioral phenotypes.pdf
- CYFIP1 --[causes]--> ASD (weight=1, sources=1)
  Evidence: The evidence from two mouse lines overexpressing human CYFIP1 does not support that CYFIP1 overexpression leads to ASD-like behaviors in this mouse model.
  Paper: 31198525 - CYFIP1 overexpression increases fear response in mice but does not affect social or repetitive behavioral phenotypes.pdf

Concerns:
- One evidence sentence appears negated or explicitly unsupported.
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.
- Prioritize this for a short manual literature review because it is relatively testable.

Agent web audit:
- Search: "CYFIP1 may be a measurable bridge between GABA-A receptor and ASD."
- Search: "GABA-A receptor" "CYFIP1" "ASD" autism
- Search: "GABA-A receptor" "CYFIP1" "decreases"
- Search: "CYFIP1" "ASD" "causes"
- Search: "CYFIP1 overexpression increases fear response in mice but does not affect social or repetitive behavioral phenotypes"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 18. GABAergic may mediate the relationship between ASD and pharmacoresistance.

Decision: advance
Priority: medium (0.622)
Category: mechanism
Raw: ASD may influence pharmacoresistance through GABAergic. (score=0.690)

Study design: Perturb or stratify by GABAergic, then measure whether the ASD to pharmacoresistance relationship changes.

Measurements:
- Primary bridge measure: GABAergic
- Endpoint measure: pharmacoresistance
- Manual evidence audit of both KG edges

KG path:
- ASD --[promotes]--> GABAergic (weight=1, sources=1)
  Evidence: Besides, brain organoids derived from induced pluripotent stem cells of patients with ASD facilitate the production of GABAergic inhibitory neurons (Mariani et al. 2015).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABAergic --[promotes]--> pharmacoresistance (weight=1, sources=1)
  Evidence: - Altered expression or function of KCC2 and NKCC1 can destabilize the excitation–inhibition balance, diminish the effect of GABAergic medications, and promote pharmacoresistance.
  Paper: 41642412 - Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis.pdf

Concerns:
- Evidence support is thin; inspect the source paper before prioritizing.
- The path is not obviously measurable or perturbable.

Next actions:
- Open the source evidence sentences and verify that the relation direction is correct.
- Search for direct literature on the rewritten hypothesis to estimate novelty.
- Convert the idea into an experimental contrast with controls and measurable endpoints.

Agent web audit:
- Search: "GABAergic may mediate the relationship between ASD and pharmacoresistance."
- Search: "ASD" "GABAergic" "pharmacoresistance" autism
- Search: "ASD" "GABAergic" "promotes"
- Search: "GABAergic" "pharmacoresistance" "promotes"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Search: "Plasma KCC2, NKCC1, and GABA as peripheral biomarkers in autism spectrum disorder a combined ROC analysis"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 19. Reject or manually inspect the path involving ASD, GABA+, and Kim.

Decision: reject
Priority: rejected (0.000)
Category: artifact
Raw: ASD may influence Kim through GABA+. (score=0.780)

Study design: Do not design a study yet; first verify that all nodes are valid biomedical concepts.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: Kim
- Manual evidence audit of both KG edges

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[decreases]--> Kim (weight=1, sources=1)
  Evidence: - Anodal transcranial direct current stimulation (a-tDCS) was reported to affect neurotransmitter levels and reduce GABA+ levels in the cerebral cortex compared with those before the stimulation (Kim et al.
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf

Concerns:
- Kim looks like an artifact or study-group label.
- Kim looks like an author name rather than a biomedical endpoint.
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Add the offending node to the miner artifact filters if it recurs.
- Do not advance this candidate until the extracted entities are corrected.

Agent web audit:
- Search: "Reject or manually inspect the path involving ASD, GABA+, and Kim."
- Search: "ASD" "GABA+" "Kim" autism
- Search: "ASD" "GABA+" "decreases"
- Search: "GABA+" "Kim" "decreases"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.

## 20. Reject or manually inspect the path involving ASD, GABA+, and Stagg.

Decision: reject
Priority: rejected (0.000)
Category: artifact
Raw: ASD may influence Stagg through GABA+. (score=0.780)

Study design: Do not design a study yet; first verify that all nodes are valid biomedical concepts.

Measurements:
- Primary bridge measure: GABA+
- Endpoint measure: Stagg
- Manual evidence audit of both KG edges

KG path:
- ASD --[decreases]--> GABA+ (weight=1, sources=1)
  Evidence: Previous studies have reported that individuals with ASD have lower GABA+ concentrations in M1 (Gaetz et al. 2014; Puts et al. 2017).
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf
- GABA+ --[decreases]--> Stagg (weight=1, sources=1)
  Evidence: - For example, participants with a lower ratio of GABA+/NAA in M1 tended to show shorter reaction times in a visually cued sequence task performed with four fingers (Stagg et al.
  Paper: 31997060 - Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder.pdf

Concerns:
- Stagg looks like an artifact or study-group label.
- Stagg looks like an author name rather than a biomedical endpoint.
- Evidence support is thin; inspect the source paper before prioritizing.

Next actions:
- Add the offending node to the miner artifact filters if it recurs.
- Do not advance this candidate until the extracted entities are corrected.

Agent web audit:
- Search: "Reject or manually inspect the path involving ASD, GABA+, and Stagg."
- Search: "ASD" "GABA+" "Stagg" autism
- Search: "ASD" "GABA+" "decreases"
- Search: "GABA+" "Stagg" "decreases"
- Search: "Altered GABA Concentration in Brain Motor Area Is Associated with the Severity of Motor Disabilities in Individuals with Autism Spectrum Disorder"
- Use `perplexity_search` for discovery; snapshot only the useful URLs that search returns.
