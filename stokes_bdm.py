# Solve Stokes with homog bcs u.n = u0 and tangential stress = data
from fvm import CentroidDistance
from dolfin import *
from petsc4py import PETSc
import sympy as sp
import ulfy

parameters['ghost_mode'] = 'shared_facet'

print = PETSc.Sys.Print

# NOTE: these are the jump operators from Krauss, Zikatonov paper.
# Jump is just a difference and it preserves the rank 
Jump = lambda arg: arg('+') - arg('-')
# Average uses dot with normal and AGAIN MINUS; it reduces the rank
Avg = lambda arg, n: Constant(0.5)*(dot(arg('+'), n('+')) - dot(arg('-'), n('-')))
# Action of (1 - n x n)
Tangent = lambda v, n: v - n*dot(v, n)    


def Stabilization(u, v, mu, penalty, consistent=True):
    '''Displacement/Flux Stabilization from Krauss et al paper'''
    mesh = u.ufl_domain().ufl_cargo()
    n, hA = FacetNormal(mesh), avg(CentroidDistance(mesh))
    
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

        subdomains = [CompiledSubDomain('near(x[0], 0)'),
                      CompiledSubDomain('near(x[0], 1)'),
                      CompiledSubDomain('near(x[1], 0)'),
                      CompiledSubDomain('near(x[1], 1)')]
    else:
        mesh = UnitCubeMesh(ncells, ncells, ncells)

        subdomains = [CompiledSubDomain('near(x[0], 0)'),
                      CompiledSubDomain('near(x[0], 1)'),
                      CompiledSubDomain('near(x[1], 0)'),
                      CompiledSubDomain('near(x[1], 1)'),
                      CompiledSubDomain('near(x[2], 0)'),
                      CompiledSubDomain('near(x[2], 1)')]
    # Mark
    boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    [subd.mark(boundaries, tag) for (tag, subd) in enumerate(subdomains)]

    return boundaries


def setup_mms(mu_value=1.0, dim=2):
    '''For a unit square'''
    mu0 = Constant(mu_value)

    if dim == 2:
        x, y = SpatialCoordinate(UnitSquareMesh(MPI.comm_self, 1, 1))

        phi = sin(pi*(x-y))
        p = cos(pi*(x+y))
        
        u = as_vector((phi.dx(1), -phi.dx(0)))
        
        normals = {1: Constant((-1, 0)),
                   2: Constant((1, 0)),
                   3: Constant((0, -1)),
                   4: Constant((0, 1))}
    else:
        x, y, z = SpatialCoordinate(UnitCubeMesh(MPI.comm_self, 1, 1, 1))

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

    subs = {mu0: sp.Symbol('mu')}
    as_expression = lambda v: ulfy.Expression(v, degree=6, subs=subs, mu=mu_value)

    return {
        'f0': as_expression(f),
        'u0': as_expression(u),
        'p0': as_expression(p),
        'tangent_tractions': {tag: as_expression(tangent_traction(normals[tag]))
                              for tag in normals}}


def setup_system(boundaries, mu, data):
    '''The linear system'''
    mesh = boundaries.mesh()
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)

    penalty = Constant(20*mesh.geometry().dim())
    
    cell = mesh.ufl_cell()
    Velm = FiniteElement('BDM', cell, 1)
    Qelm = FiniteElement('DG', cell, 0)
    Relm = FiniteElement('R', cell, 0)
    Welm = MixedElement([Velm, Qelm, Relm])

    W = FunctionSpace(mesh, Welm)
    u, p, r = TrialFunctions(W)
    v, q, dr = TestFunctions(W)

    mu = Constant(mu)
    a = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx - inner(p, div(v))*dx
         -inner(q, div(u))*dx                                             + inner(r, q)*dx
                                                   + inner(p, dr)*dx)
    # Account for HDiv
    a += Stabilization(u, v, mu, penalty=penalty)
    
    L = inner(data['f0'], v)*dx
    # Tangent "bcs", normals is handled stongly
    L += sum(inner(Tangent(v, n), traction)*ds(tag)
             for (tag, traction) in data['tangent_tractions'].items())
    # Don't forget the zero mean
    L += inner(data['p0'], dr)*dx

    bcs = DirichletBC(W.sub(0), data['u0'], 'on_boundary')

    A, b = assemble_system(a, L, bcs)

    # Preconditioner
    a_prec = (inner(2*mu*sym(grad(u)), sym(grad(v)))*dx
              + (1/mu)*inner(p, q)*dx
              + mu*inner(r, dr)*dx)
    # Account for HDiv
    a_prec += Stabilization(u, v, mu, penalty=penalty, consistent=True)

    B, _ = assemble_system(a_prec, L, bcs)    

    return A, b, W, B

# --------------------------------------------------------------------

if __name__ == '__main__':
    import tabulate

    dim = 3
    
    mu_value = 1.0
    mms_data = setup_mms(mu_value, dim=dim)

    mu = Constant(mu_value)
    u0, p0 = mms_data['u0'], mms_data['p0']

    history = []
    headers = ('hmin', 'dimW', '|eu|_1', '|eu|_div', '|div u|_0', '|ep|_0', 'niters', '|r|')
    for k in range(2, 8):
        ncells = 2**k

        boundaries = setup_geometry(ncells, dim=dim)
        A, b, W, B = setup_system(boundaries, mu=mu, data=mms_data)

        wh = Function(W)
        solver = PETScKrylovSolver()
        solver.set_operators(A, B)
        
        ksp = solver.ksp()

        opts = PETSc.Options()
        opts.setValue('ksp_type', 'minres')
        opts.setValue('ksp_rtol', 1E-12)                
        opts.setValue('ksp_view', None)
        opts.setValue('ksp_monitor_true_residual', None)                
        opts.setValue('ksp_converged_reason', None)
        opts.setValue('fieldsplit_0_ksp_type', 'preonly')
        opts.setValue('fieldsplit_0_pc_type', 'lu')
        opts.setValue('fieldsplit_1_ksp_type', 'preonly')
        opts.setValue('fieldsplit_1_pc_type', 'lu')
        opts.setValue('fieldsplit_2_ksp_type', 'preonly')
        opts.setValue('fieldsplit_2_pc_type', 'lu')                

        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.FIELDSPLIT)
        is_V = PETSc.IS().createGeneral(W.sub(0).dofmap().dofs())
        is_Q = PETSc.IS().createGeneral(W.sub(1).dofmap().dofs())
        is_R = PETSc.IS().createGeneral(W.sub(2).dofmap().dofs())        

        pc.setFieldSplitIS(('0', is_V), ('1', is_Q), ('2', is_R))
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE) 

        ksp.setUp()
        pc.setFromOptions()
        ksp.setFromOptions()

        niters = solver.solve(wh.vector(), b)
        rnorm = ksp.getResidualNorm()
        uh, ph = wh.split(deepcopy=True)[:2]

        uh_norm, ph_norm = uh.vector().norm('l2'), ph.vector().norm('l2')
        print(f'|uh| = {uh_norm} |ph| = {ph_norm}')
        
        eu_H1 = errornorm(u0, uh, 'H1')                
        eu_div = errornorm(u0, uh, 'Hdiv')        
        ep_L2 = errornorm(p0, ph, 'L2')

        div_uh = sqrt(abs(assemble(div(uh)**2*dx)))

        history.append((boundaries.mesh().hmin(), W.dim(), eu_H1, eu_div, div_uh, ep_L2, niters, rnorm))
        print(tabulate.tabulate(history, headers=headers))
