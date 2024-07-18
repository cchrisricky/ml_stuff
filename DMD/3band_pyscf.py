#The module to determine the three band Hubbard model
import numpy as np
import utils
from abc import ABC, abstractmethod
import scipy
import pyscf
from pyscf import gto, scf, ci, fci
from pyscf.ci.cisd import tn_addrs_signs

class threeband(ABC):
    @abstractmethod
    def __init__( self, Nsites, Ep, Ed, SgnMat, t, U ):

        print()

def FCI_GS(h, V, Ecore, Norbs, Nele):
	# Subroutine to perform groundstate FCI calculation using pyscf
	if isinstance(Nele, tuple):
	    Nele = sum(Nele)

	# Define pyscf molecule
	mol = gto.M()
	mol.nelectron = Nele
	# this call is necessary to use user defined hamiltonian in fci step
	mol.incore_anyway = True
	# First perform HF calculation
	mf = scf.RHF(mol)
	mf.get_hcore = lambda *args: h
	mf.get_ovlp = lambda *args: np.eye(Norbs)
	mf._eri = ao2mo.restore(8, V, Norbs)
	mf.kernel()
 
	# Perform FCI calculation using HF MOs
	cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
	E_FCI, CIcoeffs = cisolver.kernel()
 
	# Rotate CI coefficients back to basis used in DMET calculations
	CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
		CIcoeffs, Norbs, Nele, utils.adjoint(mf.mo_coeff))
 
	return CIcoeffs