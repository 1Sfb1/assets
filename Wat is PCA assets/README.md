# 📈 Wiskunde Lab Assets

In deze map staan alle plots die gemaakt zijn doormiddel van Python voor de cursus "Wiskunde Lab".

Hier volgt de gebruikte dataset, packages en een korte beschrijving van elk figuur:

Libraries:

NumPy
matplotlib
Sklearn
SciPy

Dataset:

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

| Figuur | Titel | Beschrijving |
|--------|-------|-------------|
| Figuur 1 | Drie datasets met verschillende covariantiematrices | Drie scatterplots van 2D-data met respectievelijk sterk positieve (ρ = 0.95), sterk negatieve (ρ = −0.95) en bijna nul (ρ = 0.03) correlatie. Toont visueel het effect van de buitendiagonaalelementen van Σ op de vorm van de puntenwolk. |
| Figuur 2 | Tweedimensionale dataset met twee principale componenten | Links: originele 2D-data met de eigenvectoren v₁ (PC1) en v₂ (PC2) van Σ ingetekend. Midden en rechts: histogrammen van de projecties op respectievelijk v₁ (grote spreiding, λ₁ = 9.2) en v₂ (kleine spreiding, λ₂ = 0.2). |
| Figuur 3 | Scree-plot en EVR-plot | Links: scree-plot met eigenwaarden λ₁ ≥ … ≥ λ₁₀, met een duidelijke elleboog bij k = 3. Rechts: cumulatieve Explained Variance Ratio die bij k = 3 al 88.3% bereikt en bij k ≈ 7 de 95%-drempel passeert. |
| Figuur 4 | Tumordata in de originele ruimte ℝ² | Scatterplot van de fictieve tumordata (n = 300) met celgrootte op de x-as en biomarkerconcentratie op de y-as. Kleur geeft klasse aan (benigne vs. maligne). De klassen overlappen op de x-as maar scheiden duidelijk op de y-as. PC1-richting is horizontaal ingetekend. |
| Figuur 5 | Covariantiematrix Σ van de tumordata | Heatmap van de 2×2 covariantiematrix: Var(celgrootte) = 128.67 domineert, Var(biomarker) = 0.67, covarianties ≈ 0.46. Visualiseert het enorme verschil in variantie tussen de twee features. |
| Figuur 6 | Projectieverdelingen per PC | Twee dichtheidsplots naast elkaar. Links: projectie op PC1 (celgrootte) — de twee klassen overlappen volledig. Rechts: projectie op PC2 (biomarker) — de klassen zijn duidelijk gescheiden. Toont dat hoge variantie ≠ discriminatief vermogen. |
| Figuur 7 | EVR per principale component (fictief scenario) | Staafdiagram met EVR₁ ≈ 99.5% en EVR₂ ≈ 0.5%. Illustreert dat de EVR-metriek PCA als "succesvol" bestempelt, terwijl de classificatie-informatie juist in PC2 zit. |
| Figuur 8 | Swiss Roll: geodetische vs. Euclidische afstand | 3D-plot van de Swiss Roll met twee gemarkeerde punten A en B. Een blauwe stippellijn toont de korte Euclidische afstand dwars door de rol; een rode lijn toont de veel langere geodetische afstand langs het oppervlak. |
| Figuur 9 | PCA-projectie van de Swiss Roll | Links: originele Swiss Roll in ℝ³ (kleur = positie op de rol). Rechts: PCA-projectie op PC1 en PC2 — de kleuren zijn door elkaar gemengd, wat aantoont dat PCA de manifold-structuur niet kan "uitrollen". Eigenwaarden λ₁ = 52.6, λ₂ = 51.2; EVR₂ ≈ 69.8%. |
| Figuur 10 | PCA op de Wisconsin Breast Cancer Dataset | Links: scatterplot van PC1 vs. PC2 (n = 569, d = 30), gekleurd naar diagnose — klassen zijn redelijk lineair gescheiden. Rechts: EVR-plot met per-component en cumulatieve variantie; k = 10 bereikt ≈ 95%. |
| Figuur 11 | Contrastfiguur: fictief scenario vs. WBCD | Twee dichtheidsplots naast elkaar. Links (Sectie 3.1): PC1-projectie van de fictieve data — klassen overlappen volledig. Rechts (Sectie 4): PC1-projectie van de WBCD — klassen scheiden goed. Directe visuele vergelijking van wanneer PCA faalt vs. slaagt. |
