# synthreason-source/physics — Component Shopping List by Project

The repo (`github.com/synthreason-source/physics`) is a flat, ~250-file dump with no
folders and an empty README. Below, every file is grouped into an inferred project.
Each project has a **Physical Components** subsection listing what you'd actually buy.

A general caveat: several designs here (ratchet motor, ZPE sensor, plasmonic engine)
describe mechanisms not supported by mainstream physics as continuous energy sources.
This list is the parts called for in the docs, not a claim the devices work as described.

---

## 1. Quantum Ratchet Motor (H₂O/NH₃ proton tunneling)
**Files:** `Quantum_motor.md`, `DIY_Quantum_Ratchet_Motor_Checklist.txt`,
`quantum_ratchet_motor.svg`, `watermill.py`, `nano_watermill_simulation.png`, `parts_list.png`

### Physical Components
- Liquid helium dewar (1–2 L) + cryogenic transfer tube
- Small borosilicate/Pyrex sample vial (5–10 mL)
- Vacuum jacket or Styrofoam cooler (outer insulation)
- PT-100 platinum RTD thermometer
- Cryogenic gloves
- Distilled water + ammonia (NH₃) source
- Ratchet/sawtooth rotor mechanical assembly
- Output electrode leads + femtowatt-scale readout electronics

### Software / Non-Physical
- `watermill.py` (simulation), `nano_watermill_simulation.png` (rendered output)

---

## 2. Brain–Computer Interface / Wetware Interface
**Files:** `BCIprototype.py`, `wetwareInterfacePOC.ino`, `Tryptophan_Qubit.md`,
`tryptophan-sensor.png`, `electron_seismograph.md`

### Physical Components
- Arduino Uno/Nano (the `.ino` reads analog pin A0 over serial)
- Simple analog sensor/electrode wired to A0 (EEG/EMG front-end or photodiode, depending on sensor)
- Ag/AgCl electrodes or conductive gel pads (if biosignal use)
- USB serial cable
- Breadboard, jumper wires, filtering resistors/capacitors
- UV LED or laser diode (220–290 nm) — for the tryptophan sensor variant
- Circular polarizer/waveplate rated for UV
- Photodiode + transimpedance amplifier
- Tryptophan powder (lab-grade amino acid)
- Electron gun / low-energy e-beam source + thin metal foil target + vacuum chamber
  (for the "electron tunneling seismograph" variant)
- Swivel/torsion mount with angular sensor

### Software / Non-Physical
- `BCIprototype.py` (serial read + ML classifier script)

---

## 3. Plasmonic Photovoltaic ("Silver Nanocluster Heat") Engine
**Files:** `plasmonic_photovoltaic_engine.md`

### Physical Components
- Silver nanoparticle solution or evaporated silver nanocluster film
- Diffraction grating substrate
- Multilayer graphene sheet
- Solar simulator or focused light source (400–500 nm band)
- Thermocouple or IR camera for heat measurement

---

## 4. Grid/Lattice Optical Simulator (Mach-Zehnder + Peltier + RF stack)
**Files:** `grid_simulator.py`, `grid_simulator.md`, `matrix_simulator.py`,
`lattice_sim.py`, `lattice-metrics.md`, `subset_grid_simulator.py`

### Physical Components
- 1550 nm laser diode + single-mode fiber
- LiNbO₃ electro-optic waveguide chip (Thorlabs / iXblue)
- 12-bit DAC
- InGaAs photodiode + transimpedance amplifier
- Bi₂Te₃ Peltier module (e.g. Laird Nextreme UT8-12)
- PT100 RTD or NTC thermistor
- LM393 voltage comparator
- RF signal generator + resonant cavity hardware

### Software / Non-Physical
- `grid_simulator.py`, `matrix_simulator.py`, `lattice_sim.py`, `subset_grid_simulator.py`

---

## 5. CNT-Epoxy Mesh
**Files:** `CNT_mesh.md`

### Physical Components
- MWCNT or SWCNT powder, 0.2–0.5 g (Sigma-Aldrich / Cheap Tubes)
- Low-viscosity epoxy resin (e.g. West System 105 + hardener)
- SDS or Triton X-100 surfactant
- HCl or acetone (coagulant bath)
- Ultrasonic probe sonicator (~100 W)
- Syringe pump or peristaltic pump
- Spinning needle/capillary (20–30 gauge)
- Hot plate/stirrer, small oven (up to 165°C)
- Tweezers, gloves, petri dishes
- 3D-printed or cardboard weaving frame/loom

---

## 6. ECDF Crypto Miner (firmware + ASIC path)
**Files:** `ecdf_miner.py`, `ecdf_miner.cpp`, `ecdf_miner.h`, `ecdf_miner_asic.cpp`,
`ecdf_miner_asic.h`, `ecdf_asic_realtime_dashboard.html`, `ecdf_src.zip`,
`stratum.cpp`, `stratum.h`, `miner.cpp`, `miner.h`, `firmware.bin`, `spiffs.bin`,
`monero_pspace_extended_images.pdf`

### Physical Components
- FPGA or ASIC dev board (target for the custom ASIC firmware)
- ESP32-class microcontroller (matches the `spiffs.bin`/firmware artifacts)
- Ethernet or Wi-Fi module for stratum/pool connectivity
- Heatsinks + fans for sustained hashing load
- Power supply sized for continuous mining current draw
- USB-to-serial programmer/flasher for firmware upload

### Software / Non-Physical
- `ecdf_miner.py/.cpp/.h`, `ecdf_miner_asic.cpp/.h`, `stratum.cpp/.h`, `miner.cpp/.h`,
  `ecdf_asic_realtime_dashboard.html`, `ecdf_src.zip`, `monero_pspace_extended_images.pdf`

---

## 7. "Spatial Quantum Computer" / QPU Voxel Array
**Files:** `QPU.md`, `QPU.py`, `QPU_sim.py`, `QPU_factor_sim.py`, `QPU.jpg`,
`ballistic_QPU.md`, `ballistic-qpu.html`

### Physical Components
- Array of displacement/photodiode sensors, or one high-speed large-aperture camera
- Entangled/coherent laser source with wide-beam-forming optics
- Etched glass substrate
- Spin-coating equipment (for conductive/reflective layers)
- Conductive trace material (gold sputtering target; the notes also claim plain cardboard "works")
- Adhesive for layering etched panels

### Software / Non-Physical
- `QPU.py`, `QPU_sim.py`, `QPU_factor_sim.py`, `ballistic-qpu.html`

---

## 8. Shor's Algorithm / Factorization
**Files:** `shor.py`, `factor.py`, `factor2.py`, `factor_15_strict.qasm`,
`quantum_prime.py`, `prime.md`, `prime_animation.gif`,
`metamaterial_factorization.png`, `metamaterial_sieve.png`

### Physical Components
- None specified beyond a standard computer to run the code/simulator
- If pursuing the metamaterial "sieve" variant referenced in the images: overlaps
  with Project 4's RF standing-wave cavity hardware (no separate parts list given)

### Software / Non-Physical
- `shor.py`, `factor.py`, `factor2.py`, `factor_15_strict.qasm`, `quantum_prime.py`

---

## 9. Subset-Sum / GPU Solvers
**Files:** `cuda_subset.cu`, `cuda_subset_sum_128.cu`, `cuda_subset_sum_128` (binary),
`knapsack.py`, `subset.py`

### Physical Components
- NVIDIA GPU with CUDA support (any modern consumer GPU, e.g. RTX-series)

### Software / Non-Physical
- `knapsack.py`, `subset.py`, compiled CUDA binary

---

## 10. Crypto/Randomness Utilities
**Files:** `encryption.py`, `SHA256.py`, `rand.py`

### Physical Components
- None — runs on any existing computer

---

## 11. Misc. Simulation Scripts
**Files:** `Schrodinger's_Cat.py`, `CTC.py`, `robot_behavior.py`, `doombox.py`,
`predict.py`, `state_transfer.py`, `realistic_sim.py`, `v18_csns_g_refmodel.py`, `casimir_qc.py`

### Physical Components
- None — matplotlib/numpy-based simulations and diagrams only

---

## 12. Nuclear Isomer "Battery" — not sourced here
**Files:** `nuclear_battery_AI.md`, `nuclear_fission_puzzle.md`

### Physical Components
- Not provided. The pumping methods described (reactor neutron irradiation,
  high-energy photon/laser excitation of isomer-forming nuclides) involve
  controlled nuclear materials and radiation sources, which is outside what
  I'll help source.

---

## 13–19. Theoretical / Conceptual Write-ups — No Physical Components
These are papers, notes, or interpretive frameworks with no hardware description
to shop from:

- **Automodification / Timing:** `automodification.md`, `Automodification, Timing, and the Problem of Unnecessary Intervention.pdf`, `Automodification_Timing_Rendered_Math.pdf`
- **Coherence & Geodesic Resistance:** `Coherence2.md`, `Coherence2.pdf`, `coherence_geodesic_resistance.md`
- **Beam-Phase Correlation Theory:** `beam_phase_correlation_theory.md`, `beam_phase_correlation_theory_full.md`, `beam_phase_correlation_crystal_radiation_defects.md`
- **Retrocausality / Tomography:** `Retrocausal_Xray_Experiment_Final.pdf`, `threshold-tomography.pdf`, `tomography_reconstruction, complex naturalisation.gif`
- **Zero-Point Energy Prediction Sensor:** `ZPE_PRED.md` (proposes a sensing geometry but gives no vendor-level parts list)
- **Wagenknecht Theory Papers:** `George_W.pdf`, `Wagenknecht Interpretation.txt`, `External_Space_Self_Consistency_Law_George_Wagenknecht.pdf`, `eigenvector_methods_paper_george_w.pdf`, `perturbing_the_visual_field_george_wagenknecht.pdf`, `perturbing_the_visual_field_photonic_extension_george_wagenknecht.pdf`, `resonant_displacement_gravity_theory.pdf`
- **Standalone Notes:** `5-D_reconstruction.md`, `EXPSPACE.md`, `RS_Warp_Factor_Effect.md`, `Blandford-Znajek plasma jet.md`, `Quantum_liquid.md`, `Ti.md`, `VEDBNRP.md`, `Wave.md`, `amino_acids.md`, `chiral.md`, `chiral_eyes.md`, `curvature.md`, `dark.md`, `fermion.md`, `fission.md`, `heat_death.md`, `lang.md`, `latest.md`, `lightsaber.md`, `lockbox.md`, `market.md`, `memory.md`, `microvoid.md`, `nomologies_of_motion.md`, `qtt.md`, `resonance.md`, `thermal_photons.md`, `theoretical_guidebook.md`, `truth_table_mental_health.md`, `turing war.md`, `what is the 3d crystal for.md`, `window_vision.md`, `auth.md`, `disclaimer.md`, `MTT.md`, `MTT.png`

---

## 20. Every Individual Image — Reasoned Through for Buildable Components

I opened and inspected each of the 153 image files (batched into labeled contact
sheets for review) rather than just listing filenames. Most are diagrams, tweet
screenshots, simulation output, or philosophical/art content with nothing to build.
A meaningful subset are real hardware schematics or photos, and those get an actual
parts list. A few describe things I'm not going to help source — flagged below.

**Declined items (won't provide components for):**
- The "Diisopropyl Decoxyclonate" synthesis diagram — an unverified chemical synthesis route.
- The "Zinc-Gallium Fusion" and "Lead-Gamma Positronic Reactor" diagrams — nuclear/fusion reactor concepts.
- The "box of doom" 1000W-microwave + electromagnet posts — the source itself calls this a "radiation bomb."
- The thorium-to-uranium-233 transmutation tweet — proliferation-sensitive nuclear material production.

| # | Image | What it shows | Buildable components |
|---|-------|----------------|------------------------|
| 0 | `17ed2e8a-d65b-4274-b898-279ebfbbc905.png` | "Experimental Setup for Exponential Quantum Complexity Growth" lab diagram | Laser source; microwave resonant cavity; superconducting circuit chip; cryostat; oscilloscope/spectrum analyzer |
| 1 | `1967195736640278995-G0ziF9ZaEAEBjYR.png` | Combinatorial logic flowchart (AND/XOR) | None — conceptual diagram |
| 2 | `1967420400041869801-G02uaA7aYAE5lGm.png` | AI pattern-recognition flowchart | None — conceptual diagram |
| 3 | `1968314401917899057-G1DbYzAbIAAf80M.png` | Thermal photon gas -> THz laser -> plasma cone -> thrust diagram | Thermal photon gas generator; THz laser; sealed plasma cone chamber; THz-reflective mirror/waveguide |
| 4 | `1968317271824294281-G1DeIJYbkAA2uGD.jpg` | Math: THz laser force on plasma mirror | None — calculation only |
| 5 | `1968319187308495260-G1Df3oNa0AAZFtn.jpg` | Math: gigawatt laser force calc | None — calculation only |
| 6 | `1968906714452774945-G1L2OAQakAAj4jy.png` | Gambling app screenshot | None — unrelated |
| 7 | `1969029691152519306-G1NmEBDbgAACp2w.png` | Thermal photon gas -> asymmetric plasma mirror cavity -> thrust diagram | Thermal photon gas generator; asymmetric plasma mirror cavity; thrust measurement rig |
| 8 | `1970060842189443291-G1cOJKEacAAUH-y.jpg` | Photo: clear silicone/plastic cube | Clear silicone or optical-grade resin cube (material sample) |
| 9 | `1970844883931074689-G1nY9SbbAAA8AnN.jpg` | "Bent Graphene" interference computing photo | Graphene sheet; clear silicone; syringe (for casting); Arduino board; paperweight/mass |
| 10 | `1970844883931074689-G1nY9VNbEAALEcF.jpg` | "Interference computing in bent graphene" diagram | None new — conceptual companion to prior photo |
| 11 | `1970844943808860444-G1nZBvsasAA9xyE.png` | Full-adder logic circuit diagram | 74-series logic gate ICs (XOR/AND/OR); breadboard; jumper wires |
| 12 | `1972210249093599660-G16ywgEb0AQixz1.jpg` | "Electrically Controlled 60Hz ITO Glasses" product photo | ITO-coated lens/glass; 60Hz driver circuit; glasses frame; battery |
| 13 | `1972304946545762676-G18I5aPbEAE2hWI.png` | "Two-Stage Electron Acceleration in Vacuum Flask" diagram | Evacuated glass flask; cathode/anode electrodes; high-voltage supply; hydrogen gas source; spectral detector |
| 14 | `1973336917916917793-G2KxmGJaIAEEJP7.jpg` | "Photonically Sensitive Charge-Emitting Quantum Liquid" diagram | Laser diode; optical cavity mirrors; diffuser; transimpedance amplifier circuits; Arduino |
| 15 | `1974076259446763838-G2VT3slbMAA_wf6.jpg` | Photo: futuristic syringe/injector device | Syringe body/plunger assembly; housing (generic, no spec given) |
| 16 | `1974794079897886852-G2fgga2agAAvVPg.png` | Flowchart: ITO grating -> SLM -> single-photon detector -> entangled beam | ITO diffraction grating; spatial light modulator (SLM); single-photon detector; entangled photon source |
| 17 | `1977005402803036391-G2-755vbgAAReCP.jpg` | "Silicone-Graphene wire" diagram with gold electrodes + weight | Graphene wire/sheet; silicone substrate; gold electrodes; support layer material; heavy weight/mass |
| 18 | `1978410394944217416-G3S5vbea0AA-SnB.png` | Societal/organizational flowchart | None — unrelated social diagram |
| 19 | `1978678862339871135-G3WsYWxaoAA6H4r.jpg` | Lab photo: oxy torch heating glass pressure vat + optics bench | Oxy-acetylene torch; glass pressure vessel; hydrogen gas cylinder; optical bench (lenses/mirrors/photodetector) |
| 20 | `1979094279344787725-G3cnb50WwAAv84q.jpg` | Thermal photon gas -> silicon-graphene amplifier -> hot electrons diagram | Silicon-graphene photodetector/amplifier stack; optical cavity mirrors; thermal photon source |
| 21 | `1979515380730405012-G3imonvWMAARvLI.jpg` | "Graphene Hot-Electron Photodetector with Timing Dynamics" flowchart | Graphene photodetector chip; collector electrode; timing/readout electronics |
| 22 | `1979813599460930005-G3m19YZWcAA7e0B.png` | Laser -> crystal -> 2.5GHz piezoelectric plate -> interference screen diagram | Laser source; nonlinear/piezoelectric crystal; 2.5GHz piezoelectric plate; interference screen/photodetector |
| 23 | `1979814380842934643-G3m2oTPXEAAJKlK.png` | Duplicate of prior laser/crystal diagram | Same as previous entry |
| 24 | `1979824073401307641-G3m_fz_WwAAThgz.png` | Thermoelectric/heat-pump process flow diagram | None clearly specified — too generic/illegible |
| 25 | `1984589343802155251-G4qtfa8bQAINq_v.png` | "Retrocausal tester" flowchart | Optical cavity; optical isolator; beamsplitter; photonic clock/accumulator; laser source; electronic measurement unit |
| 26 | `1985093643733189010-G4x4JlOboAAcrPs.png` | "Retrocausal Tester" v2 diagram | Electro-optic polymer plate; beamsplitter; phosphor screen; single/entangled-photon source |
| 27 | `1986555311017369636-G5GphkgbIAMJLv3.png` | "Machine Consciousness" AI architecture flowchart | None — software concept |
| 28 | `1986589854441087471-G5HI8LwacAEmgkW.png` | Crypto trading/mining-hardware purchasing workflow | Generic ASIC/GPU mining hardware only — no specific parts (business process diagram) |
| 29 | `1987716783990767623-G5XJ4aEbQAA8Uya.jpg` | "Relativistic entangled photons / c-slowing medium" diagram | Entangled photon source; slow-light medium; beamsplitter; electronic detector material; readout electronics |
| 30 | `1987792069944021305-G5YOWmibAAAk8AX.jpg` | Photo: Arduino Uno + two barrel-style sensors | Arduino Uno; 2x inductive/proximity sensors (M12 barrel type); connecting cables |
| 31 | `1987834150221750587-G5Y0dEQbgAEC580.jpg` | Duplicate relativistic photon diagram | Same as prior relativistic photon entry |
| 32 | `1988739842013089856-G5lsVyibwAA7mPU.jpg` | Small control-loop schematic | None clearly specified — too generic/illegible |
| 33 | `1990370560640266644-G583ePtakAAbGpK.png` | "Early Stop Signal Detection" graph | None — simulation graph |
| 34 | `1990535159238840799-G5_NJlFbMAEAtyS.png` | Duplicate of prior stop-signal graph | None — simulation graph |
| 35 | `1993517477805146301-G6pllNibwAM5FfL.jpg` | Reference table of quantum logic gates | None — educational reference |
| 36 | `1994079180154139062-G6xkcSjbQAAWXma.jpg` | "Device Architecture Explained": acoustoelectric amplifier stack | InGaAs 2DEG heterostructure chip; LiNbO3 piezoelectric layer; quartz block (phononic crystal); heat source; RF readout electronics |
| 37 | `1996091697353425110-G7OKueJb0AAjM-2.png` | Flowchart: polarizer film -> beamsplitter -> PVDF plate -> entangled laser | Polarizer film; beamsplitter; PVDF piezoelectric plate; entangled laser source |
| 38 | `1996132482673197545-G7Ov6f2bgAA-tVB.jpg` | Spacetime interval equation text | None — pure physics/math |
| 39 | `1997274550304948262-G7e-jy8akAA9uU9.png` | Flowchart: entangled laser <-> cube beamsplitter / DMD / black phosphorus | Entangled laser source; cube beamsplitter; digital micromirror device (DMD); black phosphorus flake; polarizer plate |
| 40 | `1998981869858336777-G73PaRNaMAIubxM.jpg` | Photo: small IR/flame sensor breakout board | IR photodiode/phototransistor sensor module (LM393-based comparator board) |
| 41 | `1999258749652074816-G77LPBQbsAArz8C.jpg` | Table: distance vs hashes-at-1MH/s | None — data reference for ECDF miner |
| 42 | `2000165115845558434-G8IDkUfakAAVFPk.jpg` | Photo: black powder sample in petri dish | CNT/graphite powder sample (ties to CNT mesh project) |
| 43 | `2001426674710204444-G8Z-8tBaMAAEi6F.png` | Flowchart: laser diode, double slit, CO2 mirror, wormhole metamaterial | Laser diode; beamsplitter; polarizer; double-slit apparatus; CO2 mirror; metamaterial sheet |
| 44 | `2001536794773590338-G8bjGqRasAACMN6.png` | "Analog Optical Prime Factorization" infographic | Coherent light source (vector optical beams); interference/diffraction setup |
| 45 | `2001902694135738582-G8gv2BBakAMz7Al.png` | Behavioral flowchart (Unawareness/Optical Reward Function) | None — psychological model |
| 46 | `2001922025741521379-G8hBb7UbkAAeiVZ.png` | Duplicate behavioral flowchart | None — psychological model |
| 47 | `2002165320807821714-G8keL23bQAAVmog.jpg` | "Photon Measurement Methods" infographic | Silicon avalanche photodiode (SPAD) module + readout electronics; (resonance method: scintillator + gamma detector) |
| 48 | `2002324581647093957-G8mvleTbMAAg-Ep.png` | "Which Way Experiment with Gamma Probe" | Proton source/accelerator (specialized equipment, not detailed here); silicon beamsplitter crystal; aluminum foil target; germanium detector; SPAD array |
| 49 | `2002352317262426426-G8nIySxbMAEDwjz.jpg` | Duplicate "Photon Measurement Methods" | Same as prior entry |
| 50 | `2020281450168480005-HAl7RFfbcAAia8M.jpg` | Linguistics ambiguity illustration | None — unrelated |
| 51 | `2021660310319641021-HA5hVUca8AEVXKM.png` | AI/material-reasoning flowchart | None clearly specified — illegible at scale |
| 52 | `2022589384680452345-HBGuS0bbcAAYpz0.jpg` | Photo: photoresistor labeled "Light Dependent Polarisation Resistor" | Standard CdS photoresistor (LDR) |
| 53 | `2024017501324300443-HBbBK85bcAES2fp.jpg` | 3-panel sci-fi art (piezo/molten iron/spacetime warp) | None — illustrative art, not a build spec |
| 54 | `2024723916338655507-HBlDqpkbsAAtrMP.png` | Crypto trading/mining automation flowchart | None — business workflow, generic mining hardware only |
| 55 | `2026077188597100972-HB4SdOuasAAxf6_.png` | "Synthesis of Diisopropyl Decoxyclonate" chemical diagram | DECLINED — chemical synthesis route for an unverified compound; not sourcing reagents for this |
| 56 | `2026405780078104688-HB89TqNaMAgZcGS.jpg` | "Abelard Declares" philosophy-of-physics chart | None — philosophical |
| 57 | `2026972514988229099-HCFAoWraYAAemvl.jpg` | Piezo-Modulated Casimir Cavity / ZPE infographic | Parallel-plate Casimir cavity (nanometer-gap conductive plates); piezoelectric modulator; metamaterial extraction circuit board; non-reciprocal resonator device (named parts only — no energy-extraction claim endorsed) |
| 58 | `2027174968140935611-HCH425pbsAAzqyK.jpg` | "Causal Paradox: CP Violations" diagram | None — theoretical particle physics |
| 59 | `2029309799154991207-HCmOgTjawAIW6GT.png` | Robot control flowchart | None specific — generic control-loop diagram |
| 60 | `2029394592916013059-HCnbmr_aYAAYnkw.jpg` | "Quantum Tunneling Ratchet" infographic | Same as Ratchet Motor project — no new items |
| 61 | `2029503925859361136-HCo-TGGagAAtRqn.jpg` | "PT-Symmetric Exceptional Point Laser" infographic | Gain-cavity laser medium; loss-cavity element; coupling waveguide; pump laser |
| 62 | `2030437340607340716-HC2P9JSbIAAmu2-.png` | "Cell Design (Prototype)" battery table | Vitreous carbon foam (positive electrode); zinc sheet (negative electrode); Nafion 117 membrane; NaClO2 aqueous electrolyte; organic solvent ClO2 trap; cell housing (named materials only, not concentrations/assembly steps — ClO2 chemistry is a hazardous oxidizer) |
| 63 | `2030437340607340716-HC2P9MkWEAA_yac.png` | "The Core Chemistry" (chlorite/ClO2 couple) | Same battery components as prior entry — no new items |
| 64 | `2030442226585944423-HC2UMiIbMAAdKpx.png` | Table: zinc layers vs voltage/current/power | None — data reference for battery project |
| 65 | `2030445289522479150-HC2XOfGaQAEdlDu.png` | Table: charging performance of 100g battery | None — data reference |
| 66 | `2030488255695896731-HC2-TUUagAAmCLn.jpg` | Bullet list comparing drone battery capacities | None — contextual reference |
| 67 | `2031317862829404484-HDCw030aMAIKQBL.jpg` | "Nano-Watermill Casimir Metamaterial Experiment" infographic | Vacuum pump; gold-coated plates; CNT-modified plate surface; dielectric fluid; gap chamber/rotor assembly |
| 68 | `2032911925152198779-HDZanhwacAABF2n.jpg` | Duplicate "Photon Measurement Methods" | Same as prior entry |
| 69 | `2032915625920835884-HDZdbavaMAMNIlg.jpg` | Duplicate relativistic entangled photon diagram | Same as prior entry |
| 70 | `2032924659516403722-HDZmJmCagAA0g-l.jpg` | Duplicate relativistic entangled photon diagram | Same as prior entry |
| 71 | `224ead14-ba68-4fdf-8bf7-b945b7427478.png` | "Entropy and the Fate of the Universe" infographic | None — cosmology theory |
| 72 | `3488.jpg` | Twitter screenshot (mostly blacked out) | None |
| 73 | `3493.jpg` | Tweet about metanopsin sensor + red PCB spectral sensor board photo | Spectral/color sensor breakout board (AS7341-style) |
| 74 | `4a984788-81aa-4cfd-bee6-17c5281713b7.png` | "How to Build a Maser-Clocked Microphone Array" infographic | Maser frequency reference; microphone array; timing/clock distribution electronics |
| 75 | `4aea835d-ec91-4479-99be-098c886fc7bf.png` | Physics notation explainer (spin projection) | None |
| 76 | `4e1fb95f-a26d-420a-bd25-f173a2d35753.png` | Duplicate sci-fi art (piezo/molten iron/spacetime) | None — illustrative art |
| 77 | `5373299b-7d8c-493d-b4d8-5b4328551c98.png` | "How to Build a 3D Microwave Photonic Crystal Quantum Cavity" infographic | Dielectric rod lattice (photonic crystal); microwave cavity resonator hardware; particle/photon source |
| 78 | `6277c98f-602e-4e62-8fab-eb8d58dae8ca.png` | Tweet on entropy-less-than-past | None |
| 79 | `628e0d17-1ee5-40b6-b814-838e9332acbf.jpg` | "Zinc-Gallium Fusion Nucleosynthetic Context" diagram | DECLINED — nuclear-fusion/nucleosynthesis theory, not extracting build components |
| 80 | `6a1db90a-5c7a-439d-be0e-d08ce1da4afd.png` | Tweet, "the anticipation formula" | None |
| 81 | `6bb6f964-2090-419b-837c-faf3a585f864.png` | "Quantum-Noise True Random Generator" schematic | Reverse-biased avalanche/noise diode; op-amp amplifier stage; comparator or ADC; resistors/capacitors |
| 82 | `6dc53d8b-fc99-4e22-8a3e-986dc56947a5.png` | "Quantum Light Through Nanocorrugated Gold" infographic | Gold foil/plate; precision drilling or lithography setup; laser source |
| 83 | `725ea20d-4ef2-4300-ae76-12eef655ca36.png` | "Photonic Woodpile Qubit Lattice" infographic | Dielectric rods (ceramic/polymer) for woodpile lattice; assembly jig; photon sensors |
| 84 | `7e99d109-c6bd-497a-9c16-320ba62b8539.png` | Dark art meme ("dopamine vial") | None |
| 85 | `85d8486d-1368-41c5-974c-506910bb7379.jpg` | "Lead-Gamma Positronic Reactor Cycle" diagram | DECLINED — nuclear/high-energy reactor concept |
| 86 | `8a984248-34b9-432e-9b5f-b05a8311e0be.png` | "Enantioselective Photodegradation with CPL and UV" lab infographic | Racemic sample mixture; circularly polarized UV light source; sample vials; polarimeter or chiral analysis equipment |
| 87 | `8b6839ab-6d22-498f-ad92-cee711ad51d4.jpg` | "Entangled-Photon Melanopsin Sensor" schematic | Entangled-photon source; illumination optics; retina-facing sensor/camera; monitoring computer/display |
| 88 | `9b01abe4-60a9-4fa7-b3ec-262ea83d77e3.jpg` | Black phosphorus crystal photo/diagram | Black phosphorus crystal flake; electrodes; current source |
| 89 | `9c46ce2e-ce91-4ca3-ae81-85abfd54b391.png` | "Maghemite Nanoparticles in Alternating Magnetic Field" infographic | Maghemite (gamma-Fe2O3) nanoparticle sample; alternating-magnetic-field coil/electromagnet; function generator/power supply |
| 90 | `AI_slop_war.png` | Joke tweet with pseudo-SQL | None |
| 91 | `Diagram1.png` | Small unreadable flowchart | None clearly specified |
| 92 | `Fermion.jpg` | BQP algorithm shot-count math | None |
| 93 | `HCFA0o-bwAELazk.png` | Duplicate Casimir/ZPE infographic | Same as prior Casimir entry |
| 94 | `HD7L6okaQAIFG4l.png` | "Key Results: T-Symmetry Satisfied" text summary | None |
| 95 | `HD7L7MFb0AEk-yt.jpg` | Simulation dashboard (histograms, scatter plots) | None |
| 96 | `HE0gtSRbcAA5Kim.png` | Interferometer flowchart (polarizer/beamsplitter/1550nm) | 1550nm laser; polarizer; beamsplitter |
| 97 | `HEPlYx7aYAISVTW.jpg` | Duplicate "3D Microwave Photonic Crystal Quantum Cavity" | Same as prior entry |
| 98 | `HESU1KeaUAA-g99.png` | Table: optical reflection zones for nanophotonic crystal | None — reference data |
| 99 | `HESUhxpaQAAeJDc.png` | Table: slow light at band edge | None — reference data |
| 100 | `HESZ7t3aEAACPaZ.png` | Text on cavity-enhanced superfluorescence | None |
| 101 | `HESjQ9AbYAE6L2-.png` | "Easiest route to working maser" step list | Ceramic + Teflon stock (woodpile crystal); pentacene-doped para-terphenyl crystal; 532nm ~500mW green laser; NanoVNA or spectrum analyzer; adjustable coupling iris mechanism |
| 102 | `HEULSVobYAYyR_F.png` | Table: bio-signal detection hierarchy | None — reference table |
| 103 | `HGEDo5GaUAAAsXJ.png` | Tweet on beam entanglement/time union | None |
| 104 | `HJ4ivxlbUAED2Zy.png` | "Air Pressure Detection with LiNbO3 Diaphragm" sensor cross-section | LiNbO3 crystal wafer (diaphragm); aluminum foil electrode; sealed vacuum reference chamber housing; piezoelectric feedback material; high-impedance sense amplifier; voltage-output readout circuit |
| 105 | `HJPMYtZb0AArhYN.jpg` | "Paper-Based Electronics" infographic | Paper substrate; conductive (silver) ink; laminated paper capacitor/memory layers |
| 106 | `HKOV6_RbsAA30gJ.png` | Tweet: Peltier H2O boiler + laser tripwire | Peltier module; small water container/boiler; laser diode + photodetector (tripwire pair) |
| 107 | `HKSGjNUaAAApJpg.png` | Tweet: "Self-Amplified Non-Adiabatic Emission" claim | None — text claim only, no hardware named |
| 108 | `HKSnrihboAAni3c.png` | "OPCPA Step-by-Step Process" infographic | Femtosecond seed laser; pulse stretcher (dispersive glass block); high-power pump laser; nonlinear parametric amplification crystal; pulse compressor (grating pair) |
| 109 | `HLApkTQa8AAhF1T.png` | "AGI consists of..." text | None |
| 110 | `HN3hdfAa4AAC3KE.png` | "Test Superposition & Entanglement" flowchart | Tryptophan liquid; gold foil vial; circular polarizer; beamsplitter; UV laser source; electronic decay-rate control circuit |
| 111 | `HN88rD2aoAAnmKY.png` | Synchrotron radiation discharge flowchart | Electron beam source; beamsplitter; magnetic steering coils; synchrotron radiation source; discharge electrode stage |
| 112 | `HPq7tFYasAANrEi.png` | Flowchart: entangled laser, drilled-hole plate, light valves | Entangled laser source; drilled-hole plate; light-valve glass elements; control electronics |
| 113 | `HPq_ideakAEdbX2.png` | Related timing/displacement diagram | None new — companion diagram |
| 114 | `HQ0BIsAbQAAhvK3.png` | Tweet ("trying to noscope God") | None |
| 115 | `HQ0C2khawAEhFDV.png` | Text card, philosophical formula | None |
| 116 | `HQ0D4PvbcAAUu11.png` | Text card, philosophical/game reference | None |
| 117 | `HQo8FJXa8AE0gL9.png` | Probability math explainer | None |
| 118 | `HQz8qJ9aUAAAs5-.png` | Tweet duplicate | None |
| 119 | `MTT.png` | "Material Time Transience" flowchart | None |
| 120 | `QPU.jpg` | Photonic entanglement/QPU photo (nanocorrugated gold) | Same as Spatial QPU project — no new items |
| 121 | `Screenshot_20260413_165647_Chrome.jpg` | "Box of doom": 1000W microwave + woodpile crystal + electromagnets described as a radiation bomb | DECLINED — source explicitly frames this as a radiological hazard/weapon; not providing components |
| 122 | `Screenshot_20260415_102719_Chrome.jpg` | Duplicate of prior "box of doom" post | DECLINED — same reason as above |
| 123 | `a0401cc2-2545-40d4-9317-fadcb5271bcc.png` | Abstract simulation visualization | None |
| 124 | `a1989681-4a9a-47a2-ba4a-2fab9b42700e.jfif` | Tweet: thorium-to-U-233 transmutation chain | DECLINED — proliferation-sensitive nuclear material production; not providing any components or sourcing |
| 125 | `af63119c.png` | Small unreadable flowchart | None clearly specified |
| 126 | `ayxs3o.gif` | Simulation data plot | None |
| 127 | `c5b58dad-e11e-4fa0-9894-dad90a8a8f49.png` | Meme image (imgflip) | None |
| 128 | `casimir_qiskit_sweep.png` | Simulation graph (Bell fidelity vs Casimir suppression) | None |
| 129 | `ccfb8d66-87b1-4cdc-81d7-64bc5a64f31f.png` | "Quantum-Noise True Random Generator" detailed schematic | Avalanche noise diode; amplifier stages; Arduino (ADC); power supply |
| 130 | `dcc16395-1a3d-4011-b487-888593f610d9.png` | Sci-fi art ("3D Temporal ET") | None |
| 131 | `de0909ae-6d7b-4a5b-81dc-2970d2e8ec22.png` | "Ghosts" poster/philosophy | None |
| 132 | `df058dda-9fe6-4cea-8eaf-457a0659cacd.png` | "Chiral UV degradation" infographic | None clearly buildable — conceptual/biological claim |
| 133 | `df6aec6c-0b43-4bde-bce0-db93f354f632.png` | Math/philosophy notation on "checkable" hypotheses | None |
| 134 | `e2cda810-247e-4b2f-b0d6-70f7a1b069da (1).jpg` | "Building Your Non-Hermitian EM Sensor (PT-Symmetric Transducer)" | YIG (yttrium iron garnet) crystal (active gain medium); gain/loss cavity components; magnetically sensitive material; balanced photodetector pair; magnetic-field coil; polished optical stage; feedback-loop control electronics |
| 135 | `ecc3a97d-1348-4bc5-9904-9d46f5afa385.jpeg` | "Fear of Detachment" art | None |
| 136 | `image.png` | IR polarisation switch/detector: aluminium wire grid on VO2-annealed COP | Aluminum wire-grid polarizer (lithographically patterned); cyclic olefin polymer (COP) substrate; vanadium dioxide (VO2) coating; small annealing oven; IR photodetector |
| 137 | `metamaterial_factorization.png` | Simulation heatmap | None — ties to factorization project |
| 138 | `metamaterial_sieve.png` | Simulation histogram | None |
| 139 | `nano_watermill_simulation.png` | Simulation dashboard | None — ties to Nano-Watermill Casimir project |
| 140 | `omnibeam.png` | "OMNIBEAM EBUC-1: Electron Beam Universal Cutter/Drill" infographic | Electron-beam gun/source; vacuum chamber; multi-axis positioning stage; high-voltage power supply; beam-focusing/deflection coils (standard e-beam machining components; needs proper HV/vacuum safety practice) |
| 141 | `parts_list.png` | Photo: PCB sensor module, glass vial, labeled component "SN75176," resistor/diode, sensor breakout | Small sensor PCB/breakout board; glass sample vial with cap; RS-485 transceiver IC (SN75176) or similar labeled part; axial resistor/diode; additional 5-pin sensor breakout |
| 142 | `prime_animation.gif` | Simulation frame | None |
| 143 | `quantum_random_triode_materials.png` | "Solid-State Quantum Random Triode – Materials" | Silicon wafer substrate; thin-film aluminum (0.3nm, cathode/anode); normal-metal gate material; thin-film deposition equipment (sputter/evaporator); readout electronics |
| 144 | `quantum_ratchet_motor.svg` | Illustration of ratchet motor | None new — matches Ratchet Motor project |
| 145 | `quantum_triangle_chip_illustration.svg` | Illustration of QPU chip | None new — matches Spatial QPU project |
| 146 | `shopping.webp` | Photo: SMD tactile push-button switch | Standard SMD tact switch |
| 147 | `tomography_reconstruction, complex naturalisation.gif` | Simulation graph | None |
| 148 | `tryptophan-sensor.png` | "UV-Sensitive Circular Polarization Sensor" | Nanoscale electrode pair; tryptophan sample; UV circular polarizer; current-readout circuit |
| 149 | `u92nggu92nggu92n.png` | Unclear/noise image | None identifiable |
| 150 | `unnamed (1).png` | Tweet: leek + gold foil dipped in chlorophyll as circular polarisation detector | Leek (plant material); gold foil; chlorophyll extract; UV circularly polarized light source; sensing electrodes |
| 151 | `unnamed (17).jpg` | "Nano-Watermill Casimir Metamaterial Experiment" (adds PZT actuator mount) | Vacuum pump; gold-coated plates; CNT-modified surface; dielectric fluid; PZT (piezo) actuator mount |
| 152 | `wr1o4cwr1o4cwr1o.png` | "Multimodal Cognitive Stabilization Framework: mmWave Biosignature Correlation" | mmWave radar sensor module; biosignal sensors (e.g. heart-rate/EEG); data-acquisition computer |

