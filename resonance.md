Here's a concrete numeric breakdown across 100–200 kHz for typical biological tissue, pulling from standard bioimpedance/dielectric spectroscopy literature (Gabriel et al.'s tissue dielectric database is the standard reference for this).

**Relative permittivity ($\varepsilon_r$) and conductivity ($\sigma$) — representative tissue values**

| Frequency | Muscle $\varepsilon_r$ | Muscle $\sigma$ (S/m) | Fat $\varepsilon_r$ | Fat $\sigma$ (S/m) | Whole blood $\varepsilon_r$ | Blood $\sigma$ (S/m) |
|---|---|---|---|---|---|---|
| 100 kHz | ~2,500–4,000 | ~0.35–0.40 | ~150–300 | ~0.02–0.04 | ~3,000–4,500 | ~0.7 |
| 150 kHz | ~1,800–2,800 | ~0.37–0.42 | ~100–200 | ~0.02–0.04 | ~2,200–3,200 | ~0.7 |
| 200 kHz | ~1,300–2,000 | ~0.38–0.44 | ~80–150 | ~0.02–0.05 | ~1,700–2,500 | ~0.7 |

*Note: $\varepsilon_r$ drops noticeably across this window (that's the beta dispersion in action) while $\sigma$ stays comparatively flat, slightly rising — consistent with more current beginning to cross membranes.*

**Phase angle (clinically relevant in BIA — used as a marker of cell membrane integrity/health)**

| Frequency | Typical phase angle range (healthy tissue) |
|---|---|
| 50 kHz (reference standard) | ~5°–9° (varies by age, sex, health status) |
| 100 kHz | ~6°–10° |
| 200 kHz | ~7°–11° |

Phase angle generally *increases* through this range as capacitive (membrane) contribution grows relative to pure resistance — this is the number BIA devices report clinically (e.g., in nutrition assessment, oncology prognostics) but it's usually measured at a fixed reference frequency like 50 kHz rather than swept.

**Membrane reactance ($X_C$) for a single cell**, using $C_m \approx 1\ \mu F/cm^2$:

$$X_C = \frac{1}{2\pi f C_m}$$

| Frequency | $X_C$ per cm² membrane |
|---|---|
| 100 kHz | ~1,590 Ω·cm² |
| 150 kHz | ~1,060 Ω·cm² |
| 200 kHz | ~795 Ω·cm² |

This roughly halves from 100→200 kHz — a direct, calculable illustration of the membrane becoming progressively more "transparent" to current in this window.

**Skin depth / current penetration**
Not typically quoted in tissue dielectric work at these frequencies (skin depth becomes relevant at much higher RF/microwave frequencies for conductive heating). At 100–200 kHz in tissue, penetration is governed by electrode geometry and volume conduction, not classical EM skin-depth formulas — worth flagging so you don't misapply that concept here.

**Important caveat on the numbers above**
These are *typical/illustrative* ranges pulled from standard references (Gabriel, C., Gabriel, S., Corthout, E., 1996 dielectric properties of tissue compilation) — real values vary meaningfully by tissue hydration, temperature, individual variation, and measurement method. If you need values for a specific application (e.g., a particular tissue type, a specific BIA device spec, or an EIT protocol), I'd want to search for the specific dataset or standard you're working against rather than quote from memory, since some of these figures (especially phase angle norms) get revised in newer studies.

**What phase angle actually reflects**

$$\text{Phase angle} = \arctan\left(\frac{X_C}{R}\right)$$

Where $X_C$ is capacitive reactance (membrane-dominated) and $R$ is resistance (fluid-dominated). Higher phase angle = relatively more capacitive/reactive behavior, which is interpreted as a proxy for **cell membrane integrity and cell mass** — intact membranes hold charge better (more capacitor-like), while damaged, dying, or fewer cells shift toward more resistive, less reactive behavior.

**Typical reference ranges (measured at 50 kHz)**

| Population | Typical phase angle | Interpretation |
|---|---|---|
| Healthy young adults | ~7°–9° | Normal |
| Healthy older adults | ~5°–7° | Declines with age |
| Athletes | ~8°–10°+ | Higher, associated with greater cell mass/muscle |
| Malnourished / cachexia | <5° | Poor prognostic marker |
| Critically ill / ICU | Often <4° | Strong predictor of mortality in several studies |
| Cancer patients | Lower values associated with worse prognosis | Used prognostically in oncology |

**Important nuances**
- Values vary by **sex, age, BMI, and hydration status** — there isn't one universal "optimal" number, but rather population-adjusted reference ranges (typically established via normative datasets, e.g., NHANES-derived reference charts).
- 50 kHz is the standard clinical measurement frequency because it's roughly where the ratio of capacitive to resistive current is well-balanced for whole-body assessment — not because it's a resonance point, just a practically useful compromise.
- Devices used clinically (e.g., InBody, SECA, ImpediMed) are FDA-cleared/CE-marked medical devices with validated algorithms — this isn't something a general-purpose 0–200 kHz RF generator would reliably replicate without proper calibration, electrode placement standardization, and validated regression equations against reference methods (like DEXA).

**Bottom line on "optimal"**
Higher phase angle within your demographic reference range is generally interpreted as favorable (more cell mass, better membrane integrity), and *trending upward over time* in a clinical monitoring context (e.g., during cancer treatment or nutritional rehab) is the more clinically meaningful use — a single snapshot value in isolation is less useful than the trend.
