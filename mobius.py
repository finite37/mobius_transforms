from numpy import linalg as la
import cmath
from scipy.optimize import differential_evolution
import warnings
import numpy as np
from utils import σ_C, stereographic, inverse_stereographic
import pyvista as pv


# Our infinity element in C_inf
inf = complex(float('inf'), float('inf'))

class Mobius:
    """Mobius Transform Class"""

    def __init__(self, a:complex, b:complex, c:complex, d:complex) -> None:

        # Checking transform validity
        sqrt_det = cmath.sqrt(a*d-b*c)
        if sqrt_det == 0:
            raise ValueError("This Mobius transformations is not invertable, ad-bc=0")
        
        # Normalizing our transform matrix (GL(2, C) -> SL(2,C))
        self.a = a/sqrt_det
        self.b = b/sqrt_det
        self.c = c/sqrt_det
        self.d = d/sqrt_det

    @classmethod
    def GL_2_C_to_Mobius(M):
        """GL(2,C) to Mobius transformation (obvious homomorphism)"""
        return Mobius(M[0,0],M[0,1],M[1,0],M[0,0])
    
    def get_coefficients(self):
        return np.array([self.a,self.b,self.c,self.d])

    def is_identity(self, tolerance=0):
        """Returns true if the matrix is the identity, within a numerical tolerance."""
        return max(abs(self.b), abs(self.c), abs(self.a - self.d)) < tolerance

    def get_SL_2_C(self):
        """Returns the SL_2_C_homomorphism as [[a,b],[c,d]]"""
        return np.array([[self.a, self.b],[self.c, self.d]])
    
    def __matmul__(self, other: 'Mobius'):
        """Implements matrix multiplication of two Mobius objects"""
        return Mobius.GL_2_C_to_Mobius(self.get_SL_2_C()@other.get_SL_2_C())

    def apply_mobius(self, z:complex):
        """Basic Mobius Transform on z"""
        with np.errstate(divide='ignore', invalid='ignore'): # Suppresses division by zero warnings
            return (self.a * z + self.b) / (self.c * z + self.d)
    
    def apply_inverse_mobius(self, z:complex):
        """Basic inver Mobius Transform on z"""
        with np.errstate(divide='ignore', invalid='ignore'): # Suppresses division by zero warnings
            return (self.d * z - self.b) / (-self.c * z + self.a)

    def get_conjugacy_class(self):
        """
        Returns conjugacy class of Mobius Transform as (k,is_identity(a,b,c,d)), note:
            k ≠ 1 iff two fixed points, with subclassificiations of:
                k ∈ (0,2)U(2,∞) iff hyperbolic transform,
                |k| = 1 iff eliptic transform, and
                otherwise iff loxodromic transform;
            k = 1 and non-identity iff parabolic and 1 fixed point; and 
            k = 1 and identity
        """
        Tr=np.trace(self.get_SL_2_C()) #trace
        k=(Tr-cmath.sqrt(Tr**2-4))/(Tr+cmath.sqrt(Tr**2-4))
        return np.array([k,self.is_identity()])
    
    def get_fixed_points(self):
        _, eigenvectors = la.eig(self.get_SL_2_C())
        with np.errstate(divide='ignore', invalid='ignore'): # Suppresses division by 0 warnings
            z = eigenvectors[0, :] / eigenvectors[1, :] #[0,0]/[1,0] and [0,1]/[1,1]
        return z #[z1, z2]

    def plot(self, vectors_scaled=True):
    
        fixed_z = self.get_fixed_points()
        fixed_coords = inverse_stereographic(fixed_z)

        sphere = pv.Icosphere(radius=1.0,nsub=4)
        z_points = stereographic(*sphere.points.T)
        w_points = self.apply_mobius(z_points)

        transformed_coords = inverse_stereographic(w_points)
        vectors = transformed_coords - sphere.points

        if not vectors_scaled:
            vectors=1/la.norm(vectors,axis=1).reshape(-1,1)*vectors
        
        if not self.is_identity() and len(fixed_coords) > 0:
            distances = np.linalg.norm(sphere.points[:, None, :] - fixed_coords[None, :, :], axis=2)
            min_distance = np.min(distances, axis=1)
            near_fixed = min_distance < 0.01
            vectors[near_fixed] = 0.0

        sphere["vectors"] = vectors

        plotter = pv.Plotter()
        arrows = sphere.glyph(orient="vectors", factor=0.03)
        plotter.add_mesh(arrows, color="grey")
        plotter.add_title("Möbius Transformation on S2")

        # Adding text
        def fmt(val):
            return f"{val.real:g} + {val.imag:g}j" if val.imag != 0 else f"{val.real:g}"
        info_text = (
            f"Möbius Parameters:\n"
            f"a = {fmt(self.a)}\n"
            f"b = {fmt(self.b)}\n"
            f"c = {fmt(self.c)}\n"
            f"d = {fmt(self.d)}"
        )
        plotter.add_text(
            info_text, 
            position='lower_left', 
            font_size=12, 
            color='black', 
            shadow=True, 
            name='params_label'
        )

        # Adding fixed points to the plot
        if self.is_identity() == False:
            for i, coord in enumerate(fixed_coords):
                point_mesh = pv.Sphere(radius=0.01, center=coord)
                plotter.add_mesh(point_mesh, color="red", label=f"Fixed Point {i+1}" if i==0 else "")
        
        plotter.show()
  
    # Currently Redundant Methods

    def uniform_metric(self,other:'Mobius'):
        """
        Returns the uniform metric between two Mobius Transforms.

            Defined as d(f,g)=sup_{x∈ℂ}(σ(f(z),g(z))),
            where σ(z,w) is the length of the cord between the projection of z and w on the Riemann-sphere
        """

        def neg_dist(params):
            """Helper function to ..."""
            θ, φ = params
            z = 0 if θ == inf else 1/np.tan(θ/2)*np.exp(1j*φ)
            return -σ_C(self.apply_mobius(z),other.apply_mobius(z))
        
        if (self@other).is_identity(): 
            return 0
    
        bounds = [(0, np.pi), (0, 2*np.pi)]
        result = differential_evolution(neg_dist, bounds, seed=42, polish=True)
        return -result.fun