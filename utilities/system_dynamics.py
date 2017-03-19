import numpy as np

__author__ = "Nathan I. Budd"
__email__ = "nibudd@gmail.com"
__copyright__ = "Copyright 2017, LASR Lab"
__license__ = "MIT"
__version__ = "0.1"
__status__ = "Production"
__date__ = "16 Mar 2017"


class SystemDynamics():
    """Combines plant dynamics and multiple perturbations into a single system.

    A callable object that combines other callable objects representing a
    system's model, control, and/or perturbations. Each of these objects must
    take inputs (T, X) where T is an (m, 1) ndarray of sample times and X is an
    (m, n) ndarray of sample states. SystemDynamics is callable with the same
    inputs, and returns the sum of its constituent parts (model, control,
    perturbations) when called.

    Attributes:
        plant : callable
            Represents the unperturbed model. Has the same inputs and outputs
            as SystemDynamics().
        preturbations : callable or list of callables
            Represent perturbations acting on the system. Has the same inputs
            and outputs as SystemDynamics().
    """

    def __init__(self, plant, perturbations):
        """."""
        self.plant = plant
        if not isinstance(perturbations, list):
            perturbations = [perturbations]
        self.perturbations = perturbations

    def __call__(self, T, X):
        """Evaluate the full system dynamics.

        Args:
            T : ndarray
                An (m, 1) array of times.
            X : ndarray
                An (m, n) array of states.

        Returns:
            Xdot : ndarray
                An (m, n) array of state derivatives
        """
        Xdot = self.plant(T, X)

        for perturb in self.perturbations:
            Xdot += perturb(T, X)

        return Xdot
