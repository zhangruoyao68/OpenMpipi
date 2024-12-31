"""
OpenMpipi
===============

A package for coarse-grained simulations using the Mpipi recharged forcefield in OpenMM.

Modules:
- biomolecules: Defines classes for various biomolecules.
- coordinate_building: Provides functions for building/reading initial configurations for individual molecules.
- system_building: Contains functions for building OpenMM System objects.
- model_building: Contains functions for building slab configurations for direct coexistance simulations.

Author:
- Kieran Russell kor20@cam.ac.uk
- Collepardo Lab, University of Cambridge

Version: 0.1.0
"""

PACKAGE_NAME = 'OpenMpipi'
__version__ = "0.1.0"

import json
import os
import numpy as np
import openmm as mm
from openmm import app
from openmm import unit

from .biomolecules import IDP, MDP, RNA
from .constants import PLATFORM, PROPERTIES
from .system_building import get_mpipi_system
from .model_building import *
