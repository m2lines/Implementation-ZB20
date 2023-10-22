# Implementation of Zanna-Bolton-2020 mesoscale parameterization to GFDL MOM6 ocean model
This repository describes the implementation of [ZB20](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL088376) parameterization of mesoscale eddies to GFDL [MOM6](https://github.com/NOAA-GFDL/MOM6) ocean model.

The implementation is described in detail in paper "Implementation of a data-driven equation-discovery mesoscale parameterization into an ocean model", Pavel Perezhogin, Cheng Zhang, Alistair Adcroft, Carlos Fernandez-Granda, Laure Zanna, submitted to JAMES.

* See [Figure-plotting](https://github.com/m2lines/Implementation-ZB20/tree/main/Figure-plotting) for notebooks with Figures.
* See [MOM6](https://github.com/m2lines/Implementation-ZB20/tree/main/src) source code used to conduct the research. The implemented ZB20 parameterization is [part](https://github.com/NOAA-GFDL/MOM6/blob/475590dbc4d736fd45a29748577351f2eb58fc57/src/parameterizations/lateral/MOM_Zanna_Bolton.F90) of this source code.
* See [configurations](https://github.com/m2lines/Implementation-ZB20/tree/main/configurations) for files required to run Double Gyre and NeverWorld2 experiments with implemented parameterizations.
