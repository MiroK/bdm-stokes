# Functionality related for computing cell distance 
import dolfin as df


def CellCentroid(mesh):
    '''[DG0]^d function that evals on cell to its center of mass'''
    V = df.VectorFunctionSpace(mesh, 'DG', 0)
    v = df.TestFunction(V)

    hK = df.CellVolume(mesh)
    x = df.SpatialCoordinate(mesh)

    c = df.Function(V)
    df.assemble((1/hK)*df.inner(x, v)*df.dx, tensor=c.vector())

    return c


def FacetCentroid(mesh):
    '''[DLT]^d function'''
    assert mesh.topology().dim() > 1
    
    xs = df.SpatialCoordinate(mesh)
    
    V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    v = df.TestFunction(V)
    hF = df.FacetArea(mesh)

    xi_foos = []
    for xi in xs:
        form = (1/df.avg(hF))*df.inner(xi, df.avg(v))*df.dS + (1/hF)*df.inner(xi, v)*df.ds
        xi = df.assemble(form)

        xi_foo = df.Function(V)
        xi_foo.vector()[:] = xi
        xi_foos.append(xi_foo)

    V = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
        
    x = df.Function(V)
    for i, xi in enumerate(xi_foos):
        df.assign(x.sub(i), xi)
    return x


def CentroidVector(mesh):
    '''DLT vector pointing on each facet from one CenterTroid to the other'''
    # Cell-cell distance for the interior facet is defined as a distance 
    # of circumcenters. For exterior it is facet centor to circumcenter
    # For facet centers we use DLT projection
    assert mesh.topology().dim() > 1

    L = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
        
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)

    facet_centers = FacetCentroid(mesh)
    cell_centers = CellCentroid(mesh)
    
    cc = df.Function(L)
    # Finally we assemble magniture of the vector that is determined by the
    # two centers
    df.assemble((1/fK('+'))*df.inner(cell_centers('+')-cell_centers('-'), l('+'))*df.dS +
                (1/fK)*df.inner(cell_centers-facet_centers, l)*df.ds,
                tensor=cc.vector())

    return cc


def CentroidDistance(mesh):
    '''Magnitude of Centervector as a DLT function'''
    assert mesh.topology().dim() > 1

    L = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
        
    fK = df.FacetArea(mesh)
    l = df.TestFunction(L)
    
    cc = CentroidVector(mesh)
    distance = df.Function(L)
    # We use P0 projection
    df.assemble(1/fK('+')*df.inner(df.sqrt(df.dot(cc('+'), cc('+'))), l('+'))*df.dS
                + 1/fK*df.inner(df.sqrt(df.dot(cc, cc)), l)*df.ds,
                tensor=distance.vector())
        
    return distance    
