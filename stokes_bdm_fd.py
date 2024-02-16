# Solve Stokes with homog bcs u.n = u0 and tangential stress = data
from firedrake import *
from fvm_fd import CentroidDistance
from petsc4py import PETSc

print = PETSc.Sys.Print

# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')
# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: Constant(0.5)*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))
# Action of (1 - n x n)
Tangent = lambda v, n: v - n*dot(v, n)    


# CellDiameter = CentroidDistance   # NOTE: also adjust the penalty parameter


def Stabilization(mesh, u, v, mu, penalty, consistent=True):
    '''Displacement/Flux Stabilization from Krauss et al paper'''
    n, hA = FacetNormal(mesh), avg(CellDiameter(mesh))
    
    D = lambda v: sym(grad(v))

    if consistent:
        return (-inner(Avg(2*mu*D(u), n), Jump(Tangent(v, n)))*dS
                -inner(Avg(2*mu*D(v), n), Jump(Tangent(u, n)))*dS
                + 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS)
    # For preconditioning
    return 2*mu*(penalty/hA)*inner(Jump(Tangent(u, n)), Jump(Tangent(v, n)))*dS


def setup_geometry(ncells, dim=2):
    '''Marked unit square'''
    if dim == 2:
        mesh = UnitSquareMesh(ncells, ncells)
    else:
        mesh = UnitCubeMesh(ncells, ncells, ncells)
    return mesh


def setup_mms(mesh, mu_value=1.0, dim=2):
    '''For a unit square'''
    mu0 = Constant(mu_value)

    if dim == 2:
        x, y = SpatialCoordinate(mesh)
        phi = sin(pi*(x-y))
        p = cos(pi*(x+y))
        
        u = as_vector((phi.dx(1), -phi.dx(0)))
        
        normals = {1: Constant((-1, 0)),
                   2: Constant((1, 0)),
                   3: Constant((0, -1)),
                   4: Constant((0, 1))}
    else:
        x, y, z = SpatialCoordinate(mesh)

        phi = sin(pi*(x-y))
        p = cos(pi*(x+y))
        
        u = as_vector((phi.dx(1), -phi.dx(0), x+y))
        
        normals = {1: Constant((-1, 0, 0)),
                   2: Constant((1, 0, 0)),
                   3: Constant((0, -1, 0)),
                   4: Constant((0, 1, 0)),
                   5: Constant((0, 0, -1)),
                   6: Constant((0, 0, 1))}
        
    sigma = 2*mu0*sym(grad(u)) - p*Identity(len(u))
    f = -div(sigma)

    tangent_traction = lambda n: Tangent(dot(sigma, n), n)

    return {
        'f0': f,
        'u0': u,
        'p0': p,
        'tangent_tractions': {tag: tangent_traction(normals[tag]) for tag in normals}}


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
    a = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx - inner(p, div(v))*dx
         -inner(q, div(u))*dx)

    # Account for HDiv
    a += Stabilization(mesh, u, v, mu, penalty=penalty)
    
    L = inner(data['f0'], v)*dx
    # Tangent "bcs", normals is handled stongly
    L += sum(inner(Tangent(v, n), traction)*ds(tag)
             for (tag, traction) in data['tangent_tractions'].items())

    bcs = DirichletBC(W.sub(0), data['u0'], 'on_boundary')

    # Preconditioner
    a_prec = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
              + (1/mu)*inner(p, q)*dx)
    # Account for HDiv
    a_prec += Stabilization(mesh, u, v, mu, penalty=penalty, consistent=True)
                                       
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
    import tabulate

    dim = 3
    
    mu_value = 1.0

    history = []
    headers = ('dimW', '|eu|_1', '|eu|_div', '|div u|_0', '|ep|_0', 'niters', '|r|')
    for k in range(2, 5):
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

        
        ph_p1 = project(ph, FunctionSpace(mesh, 'CG', 1),
                        solver_parameters={ 
                            'ksp_type': 'cg',
                            'ksp_rtol': 1E-40,
                            'ksp_atol': 1E-13,        
                            'ksp_view': None,
                            'pc_type': 'hypre'})
                        
        # ----
        
        outfile = File("uh.pvd")
        outfile.write(uh)

        outfile = File("ph.pvd")
        outfile.write(ph)

        outfile = File("ph_p1.pvd")
        outfile.write(ph_p1)                        
