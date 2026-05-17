**The beam — a Mach-Zehnder interferometer**

You need:
- A cheap diode laser (~650nm red, £5)
- Two 50/50 beam splitters (half-silvered mirrors)
- Two fully reflective mirrors
- An optical bench or even a flat piece of aluminium with bolt holes

The laser hits the first beam splitter, splits into two paths, both paths reflect off mirrors and recombine at the second beam splitter. A photodetector at the output reads the interference pattern. Phase difference between the two arms determines whether you get a bright or dark output. You control phase by nudging one mirror with a piezoelectric actuator — tiny voltage = tiny movement = phase shift.

This is your "lock mechanism." The phase setting is the secret.

---

**Concealing the phase**

You set the piezo to a voltage (known only to you), which puts the interferometer in a specific phase state. The output at the detector is just bright or dark — that single bit tells an observer nothing about *what phase produced it* unless they already know the baseline.

---

**The which-path trap**

Insert a polariser in one arm of the interferometer. Now each path has a distinct polarisation — horizontal vs vertical. A photon now carries which-path information, and the interference pattern disappears completely. The output goes flat. Anyone trying to "peek" inside the device by sampling the beam mid-path automatically tags the photon with path information and kills the interference. You'd see this as a drop in fringe visibility at the detector — a measurable alarm.

---

**The measurement condition**

The correct measurement basis has to be chosen before measuring. You set up two detectors at the output — one measuring in the +/- basis, one in the ×/÷ basis (like BB84). If the phase was set in the +/- basis and you measure in ×/÷, you get a random result with no signal. The lock doesn't open, and you've learned nothing useful. Worse, your wrong-basis measurement has perturbed the state.

You'd implement this with a half-wave plate on a rotation mount before the detector. Rotate to the wrong angle — noise. Correct angle — clean signal.

---

**The "open" condition**

Wire the photodetector output to a threshold comparator (a simple op-amp circuit). Only when:
1. Fringe visibility is above a threshold (nobody peeked), AND
2. The measured output matches the expected value for the correct phase

...does the comparator trigger a relay or solenoid. That's your physical unlock.

---

**Cost to prototype**

| Component | Rough cost |
|---|---|
| Diode laser | £5–10 |
| Beam splitters x2 | £20–40 |
| Mirrors x2 | £10–20 |
| Piezo actuator | £15–30 |
| Half-wave plate | £20–50 |
| Photodetectors x2 | £10–20 |
| Op-amp comparator circuit | £5 |
| Optical rail/bench | £30–80 |

Rough total: **£115–255** for a functional tabletop prototype. Alignment is the hard part — even vibrations from footsteps will wash out fringes. You'd want to damp the whole thing on a rubber mat or inner tube.

---

The elegant thing is the security proof is physical, not computational. There's no algorithm to crack. The laws of quantum mechanics are doing the enforcing.
