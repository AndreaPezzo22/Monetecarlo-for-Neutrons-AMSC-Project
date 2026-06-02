# Experiment Descriptions

## 1. Experiment: multilayered_wall

This experiment simulates the classic configuration of protective shields used in nuclear laboratories. The goal is to block a directional neutron beam by utilizing three materials with complementary physical properties [1].

The neutron first encounters a thick block of **Iron**, whose purpose is to reduce the kinetic energy of the fastest neutrons through hard inelastic collisions. It then enters **Water** (the moderator par excellence), where collisions with hydrogen nuclei slow the particle down to thermal energies. Finally, the now-slow neutrons strike **Boron**, which has a very high absorption cross-section and acts as a final “sponge.”

![Multilayered Wall](../images/multilayered_wall.jpg)

## 2. Esperimento: matrioska_cube"

Simulate a point source embedded in the center of a massive solid moderator.

The simulated physical dynamics predict that the neutrons will travel through the initial vacuum without interacting, and then collide with the graphite walls—which are forty units thick—where, due to the high scattering cross section and the very low absorption of carbon, they will begin a process of three-dimensional random diffusion.   

## 3. Experiment: maze

A geometry designed to calculate parasitic radiation streaming along the corridors of accelerators or radiation therapy rooms. Neutrons don't like straight lines; this experiment tests whether they can “turn the corner” [2].

Neutrons are fired into an “L”-shaped tunnel. The **concrete** walls tend to absorb most of the particles, but a percentage bounces off the walls. Only the neutrons that undergo the geometrically perfect sequence of bounces manage to navigate the curve and reach the exit.

## 4. Esperimento: slit_collimator

In this experiment, a microscopic vacuum tunnel is surrounded by **cadmium**. This material has highly absorptive properties, so only neutrons whose trajectory is parallel to the tunnel axis survive.

## 5. Experiment: kernel_pattern

A miniature reproduction of the internal topology of the core of a pressurized water reactor, where the fuel and moderator are arranged in a checkerboard pattern.

Thin “bars” of **Uranium** are separated from one another by channels of **Water**. Neutrons are generated and travel continuously through the two materials: they undergo elastic scattering in the water and absorption in the uranium.


## References

[1] Shultis, John & Faw, Richard. (2005). Radiation shielding technology. Health physics. 88. 587-612. 10.1097/01.HP.0000148615.73825.b1. 
[2] Haghighat, Alireza & Sjoden, Glenn & Yi, Ce. (2009). Analysis and benchmarking of PENTRAN code using the OECD-NEA benchmark problems. 1. 