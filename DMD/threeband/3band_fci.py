import numpy as np
import pyscf
from pyscf import gto, scf, ao2mo
import utils

# Customize the Hubbard model solver

mol = gto.M()
Ne = 12
Norbs = 12
N_d = 4 # number of d orbitals, indices start from 0
mol.nelectron = Ne

# this call is necessary to use user defined hamiltonian in fci step
mol.incore_anyway = True

# Define the Hamiltonian
h = np.zeros((Norbs,Norbs))
t = 1.3 / 27.12 # transition energy: 1.3 eV
Ep = t * 3 # p-orbital energy. It's 3-7 times of t in the DMD paper
Delta = Ep # charge transfer energy, Ep-Ed, and Ed is set to be reference zero
Ud = t * 8 # onsite Hubbard interaction, 4-14 times of t in the DMD paper, U_p is 0
V = 0 # p-d density-density interactions, set to be zero

E_orb = np.zeros(Norbs)
for i in range(N_d, Norbs):
    E_orb[i] = Ep

sgn_mat = np.copy(h) # the matrix that defines signs between the p-d transition
sgn_mat[0,4] = sgn_mat[0,5] = sgn_mat[4,0] = sgn_mat[5,0] = -1
sgn_mat[0,6] = sgn_mat[0,7] = sgn_mat[6,0] = sgn_mat[7,0] = 1

sgn_mat[1,5] = sgn_mat[1,11] = sgn_mat[5,1] = sgn_mat[11,1] = 1
sgn_mat[1,6] = sgn_mat[1,10] = sgn_mat[6,1] = sgn_mat[10,1] = -1

sgn_mat[2,4] = sgn_mat[2,9] = sgn_mat[4,2] = sgn_mat[9,2] = 1
sgn_mat[2,7] = sgn_mat[2,8] = sgn_mat[7,2] = sgn_mat[8,2] = -1

sgn_mat[3,8] = sgn_mat[3,10] = sgn_mat[8,3] = sgn_mat[10,3] = 1
sgn_mat[3,9] = sgn_mat[3,11] = sgn_mat[9,3] = sgn_mat[11,3] = -1

h = sgn_mat * t + np.diag(E_orb) # hamiltonian with all 1-electron terms

g2e = np.zeros( (Norbs,)*4 )

for i in range(Norbs):

    if i<4:
        g2e[i,i,i,i] = Ud

    for j in range( i, Norbs ):
        if sgn_mat[i,j] != 0:
            g2e[i,i,j,j] = g2e[j,j,i,i] = V

# Perform HF calculation
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h
mf.get_ovlp = lambda *args: np.eye(Norbs)
mf._eri = ao2mo.restore(8, g2e, Norbs)
mf.kernel()

#
# Second, do the FCI calculation

cisolver = pyscf.fci.FCI(mf, mf.mo_coeff)
E_FCI, CIcoeffs = cisolver.kernel()

print(CIcoeffs)
# Rotate CI coefficients back to basis used in DMET calculations
#CIcoeffs = pyscf.fci.addons.transform_ci_for_orbital_rotation(
#    CIcoeffs, Norbs, Ne, utils.adjoint(mf.mo_coeff) )

#print(CIcoeffs)
print("energy:", E_FCI)
utils.printarray(CIcoeffs, 'CIcoeffs.dat', True)