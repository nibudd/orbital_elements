# orbital_elements
A collection of classes for computing dynamics and solutions in different orbital element sets, along with functions for converting between different element sets.


There are a number of different ways to represent orbits and calculate their dynamics, each with their pros and cons. Often it can be cumbersome to convert between the different represenations, so this was my attempt at making those tasks a bit easier.

Some functionalities include:
-conversion between cartesian, keplerian elements, modified equinoctial elements, and some variations thereof.
-the ability to superimpose different dynamical features to create a simulation of a whole system (e.g. including Keplerian two-body dynamics along with a Lyapunov control thruster and zonal gravity to simulate the control of a satellite). Multiple control and dynamics components can be included to create a single system.
