# # Brick Benchmark

from __future__ import print_function
import underworld as uw
import underworld.function as fn
import numpy as np

# LMR utilities
import UWGeodynamics as GEO
from UWGeodynamics import nd


# In[2]:


u = GEO.u

# Characteristic values of the system
half_rate = 2e-11 * u.meter / u.second
model_length = 40 * u.kilometer
model_height = 10 * u.kilometer
bodyforce = 2700 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = model_length
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM


# In[3]:


nx = 256
ny = 64

minX = nd(-20. * u.kilometer)
maxX = nd(20. * u.kilometer)
minY = nd(0. * u.kilometer)
maxY = nd(10. * u.kilometer)

mesh = uw.mesh.FeMesh_Cartesian(elementType = ("Q1/dQ0"),
                                elementRes  = (nx, ny),
                                minCoord    = (minX, minY),
                                maxCoord    = (maxX, maxY))

swarm  = uw.swarm.Swarm(mesh = mesh)
swarmLayout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm = swarm, particlesPerCell=25)
swarm.populate_using_layout(layout = swarmLayout)


velocityField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=mesh.dim )
pressureField = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )

strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn) 
velocityField.data[...]  = 0.0
pressureField.data[...]  = 0.0



materialIndexField = swarm.add_variable( dataType="int", count=1 )

background = 0
seed = 1


vertices = [(-400  * u.meter,  0.),
            (400. * u.meter,  0.),
            (400. * u.meter, 400. * u.meter),
            (-400. * u.meter, 400. * u.meter)]

vertices = np.array([(GEO.nd(x), GEO.nd(y)) for x,y in vertices])
seed_shape = uw.function.shape.Polygon(vertices)

mat = fn.branching.conditional([(seed_shape, 1), (True, background)])
materialIndexField.data[:] = mat.evaluate(swarm)


densityFn = fn.misc.constant(nd(   2700. * u.kilogram / u.metre**3))
z_hat = ( 0.0, -1.0 )
gravity = nd(10. * u.metre / u.second**2)
buoyancyFn = densityFn * z_hat * gravity


background_viscosity = nd(1e25 * u.pascal * u.second)
seed_viscosity = nd(1e20 * u.pascal * u.second)

#Plane Strain Drucker-Prager
DefaultSRInvariant = nd(1.0e-15 / u.second)
C = GEO.nd(40. * u.megapascal)
Phi = np.radians(15)
YieldStress = C*fn.math.cos(Phi) + pressureField * fn.math.sin(Phi)
eij = fn.branching.conditional([(strainRate_2ndInvariantFn < 1e-20, DefaultSRInvariant),
                                (True, strainRate_2ndInvariantFn)])
mu =  0.5 * YieldStress / eij

    
ViscosityMap = {seed: seed_viscosity,
                background: fn.misc.min(mu, background_viscosity)
}

viscosityFn = fn.branching.map(fn_key = materialIndexField, mapping = ViscosityMap)


# # Boundary conditions


left = mesh.specialSets["MinI_VertexSet"]
right = mesh.specialSets["MaxI_VertexSet"]
base = mesh.specialSets["MinJ_VertexSet"]
top = mesh.specialSets["MaxJ_VertexSet"]

velocityField.data[left.data,0] = nd(-2e-11 * u.meter / u.second) # -1.0
velocityField.data[right.data,0] = nd(2e-11 * u.meter / u.second) # 1.0

stokesBC = uw.conditions.DirichletCondition(
    variable = velocityField, 
    indexSetsPerDof = (left+right, base))


stokes = uw.systems.Stokes(velocityField = velocityField, 
                           pressureField = pressureField,
                           conditions    = stokesBC,
                           fn_viscosity  = viscosityFn, 
                           fn_bodyforce  = buoyancyFn)

solver = uw.systems.Solver( stokes )
solver.set_inner_method("mumps")
solver.set_penalty(1e6)

solver.solve(nonLinearIterate=True, nonLinearMaxIterations=1000, 
             nonLinearMinIterations=1000, 
             nonLinearTolerance=1e-7)


