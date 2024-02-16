# NOTE: Using Nitsche bcs messes up the exact mass conservation
# Solve Stokes with homog bcs u.n = u0 and tangential stress = data
from firedrake import *
from fvm_fd import CentroidDistance
from petsc4py import PETSc

print = PETSc.Sys.Print

from stokes_bdm_fd import Jump, Avg, Tangent, Stabilization


def setup_system(mesh, mu, data):
    '''The linear system'''
    n = FacetNormal(mesh)
    
    penalty = Constant(20*mesh.ufl_cell().geometric_dimension())
    
    cell = mesh.ufl_cell()
    Velm = FiniteElement('BDM', cell, 1)
    Qelm = FiniteElement('DG', cell, 0)
    Welm = MixedElement([Velm, Qelm])

    W = FunctionSpace(mesh, Welm)
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    mu = Constant(mu)

    sigma = lambda u, p: 2*mu*sym(grad(u)) - p*Identity(len(u))
    
    a = (inner(sigma(u, p), sym(grad(v)))*dx
         -inner(q, div(u))*dx)

    # Account for HDiv
    a += Stabilization(mesh, u, v, mu, penalty=penalty)
    
    L = inner(data['f0'], v)*dx
    # Tangent "bcs", normals is handled stongly
    L += sum(inner(Tangent(v, n), traction)*ds(tag)
             for (tag, traction) in data['tangent_tractions'].items())

    bcs = None

    hK = CellDiameter(mesh)
    gamma = penalty
    # Use Nitsche instead
    a += (-inner(dot(n, dot(sigma(u, p), n)), dot(v, n))*ds
          -inner(dot(n, dot(sigma(v, q), n)), dot(u, n))*ds
          +(gamma/hK)*inner(dot(u, n), dot(v, n))*ds)

    L += (-inner(dot(n, dot(sigma(v, q), n)), dot(mms_data['u0'], n))*ds
          +(gamma/hK)*inner(dot(mms_data['u0'], n), dot(v, n))*ds)    

    # Preconditioner
    a_prec = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
              + (1/mu)*inner(p, q)*dx)
    # Account for HDiv
    a_prec += Stabilization(mesh, u, v, mu, penalty=penalty, consistent=False)
    # And the boundary conditions
    a_prec += (#-inner(dot(n, dot(sigma(u, p), n)), dot(v, n))*ds
               #-inner(dot(n, dot(sigma(v, q), n)), dot(u, n))*ds
               +(gamma/hK)*inner(dot(u, n), dot(v, n))*ds)    
                                       
    return a, L, W, bcs, a_prec


solver_params = {
    'ksp_type': 'minres',
    'ksp_rtol': 1E-12,
    'ksp_view': None,
    'ksp_monitor_true_residual': None,
    'ksp_converged_reason': None,
    'pc_type': 'fieldsplit',
    'fieldsplit_0_ksp_type': 'preonly',
    'fieldsplit_0_pc_type': 'lu',
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'lu',
}

# --------------------------------------------------------------------

if __name__ == '__main__':
    from stokes_bdm_fd import setup_geometry, setup_mms
    import tabulate

    dim = 2
    
    mu_value = 1.0

    history = []
    headers = ('dimW', '|eu|_1', '|eu|_div', '|div u|_0', '|ep|_0', 'niters', '|r|')
    for k in range(2, 8):
        ncells = 2**k
        mesh = setup_geometry(ncells, dim=dim)

        mms_data = setup_mms(mesh, mu_value, dim=dim)

        mu = Constant(mu_value)
        u0, p0 = mms_data['u0'], mms_data['p0']
        
        a, L, W, bcs, a_prec = setup_system(mesh, mu=mu, data=mms_data)

        wh = Function(W)
        problem = LinearVariationalProblem(a=a, L=L, u=wh, aP=a_prec, bcs=bcs)

        nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True)])
        solver = LinearVariationalSolver(problem, solver_parameters=solver_params,
                                         nullspace=nullspace)
        
        solver.solve()

        ksp = solver.snes.ksp

        niters = ksp.getIterationNumber()
        rnorm = ksp.getResidualNorm()
        uh, ph = wh.split()

        eu_H1 = errornorm(u0, uh, 'H1')                
        eu_div = errornorm(u0, uh, 'Hdiv')        
        ep_L2 = errornorm(p0, ph, 'L2')

        div_uh = sqrt(abs(assemble(div(uh)**2*dx)))

        history.append((W.dim(), eu_H1, eu_div, div_uh, ep_L2, niters, rnorm))
        print(tabulate.tabulate(history, headers=headers))

        outfile = File("uh.pvd")
        outfile.write(uh)

        outfile = File("ph.pvd")
        outfile.write(ph)                
