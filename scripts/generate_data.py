import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as splin
import matplotlib.pyplot as plt
from pathlib import Path


class Logistic_solver:
    def __init__(self,viscosity,resources,size_x,size_y):
        self.viscosity = viscosity
        self.resources = resources
        self.max_resources = np.max(resources)
        self.tot_size = size_x * size_y
        self.size_x = size_x
        self.size_y = size_y    
   
    def residual(self,vect):
        """
        From a candidate value function named vect, compute the residual of the logistic equation.
        
        The distribution of the fishermen is given.
        """
        return (- self.viscosity * Laplacian(vect,self.size_x,self.size_y)
                - vect * (self.resources - vect))
    
    def jacobian(self,vect):
        """
        Compute the jacobian of the differential operator applied in residual
        """
        return (-self.viscosity * Laplacian_mat(self.size_x,self.size_y)
                - sp.diags(self.resources - 2 * vect))
    
    def solve_lin_logistic(self,theta,additional_RHS):
        """
        This solves the linearized logistic equation in the direction delta_fisher_distrib
        with an additional_RHS as the right-hand side.
        
        Observe that the logistic equation we solve has 0 on the right-hand side.
        However, it will sometime be usefull to have a nonzero RHS for the linearisation (for the
        Newton method for instance)
        """
        return splin.spsolve(self.jacobian(theta), additional_RHS)
    
    def solve_logistic(self,initial_guess=None,nb_max_newton_iter=100,threshold=1e-8):
        """
        This solves the logistic equation with a given fishermen distribution.
        
        The method used here is a Newton method, using the method solve_lin_logistic.
        """
        if initial_guess is None:
            output = self.max_resources * np.ones(self.tot_size)
        else:
            output = initial_guess.copy()
        for i in range(nb_max_newton_iter):
            residual = self.residual(output)
            norm_residual = np.sqrt(np.mean(residual*residual))
            if norm_residual<threshold:
                return output
            output -= self.solve_lin_logistic(output,residual)
        print('In solve_logistic, no convergence after %d Newton iterations, the residual norm is %.2e'%(nb_max_newton_iter,
                                                                                            norm_residual))
        raise NewtonException()
        return output

    #########################
    ## SOME TOOLS FOR PDEs ##
    #########################


class AllenCahnSolver:
    def __init__(self, epsilon, size_x, size_y):
        self.epsilon = epsilon
        self.tot_size = size_x * size_y
        self.size_x = size_x
        self.size_y = size_y
    
    def residual(self, vect):
        """
        From a candidate value function named vect, compute the residual of the Allen-Cahn equation.
        
        ε² Δu - (u³ - u) = 0
        """
        return ((self.epsilon ** 2) * Laplacian(vect, self.size_x, self.size_y)
                - (vect ** 3 - vect))
    
    def jacobian(self, vect):
        """
        Compute the jacobian of the differential operator applied in residual
        """
        return ((self.epsilon ** 2) * Laplacian_mat(self.size_x, self.size_y)
                - sp.diags(3 * vect ** 2 - 1))
    
    def solve_lin_allen_cahn(self, theta, additional_RHS):
        """
        This solves the linearized Allen-Cahn equation in the direction delta_u
        with an additional_RHS as the right-hand side.
        """
        return splin.spsolve(self.jacobian(theta), additional_RHS)
    
    def solve_allen_cahn(self, initial_guess=None, nb_max_newton_iter=100, threshold=1e-8):
        """
        This solves the Allen-Cahn equation.
        
        The method used here is a Newton method, using the method solve_lin_allen_cahn.
        """
        if initial_guess is None:
            # Do not start from the exact zero state: u=0 is already a valid
            # stationary solution and Newton would terminate immediately.
            # Use a deterministic non-constant periodic profile to target a
            # non-trivial branch.
            x = np.linspace(0.0, 1.0, self.size_x, endpoint=False)
            y = np.linspace(0.0, 1.0, self.size_y, endpoint=False)
            X, Y = np.meshgrid(x, y, indexing="ij")
            profile = np.sin(2.0 * np.pi * X) + 0.5 * np.cos(2.0 * np.pi * Y)
            output = 0.9 * np.tanh(profile / max(self.epsilon, 1e-8))
            output = output.reshape(-1)
        else:
            output = initial_guess.copy()
        for i in range(nb_max_newton_iter):
            residual = self.residual(output)
            norm_residual = np.sqrt(np.mean(residual * residual))
            if norm_residual < threshold:
                return output
            output -= self.solve_lin_allen_cahn(output, residual)
        print('In solve_allen_cahn, no convergence after %d Newton iterations, the residual norm is %.2e' % (nb_max_newton_iter,
                                                                                            norm_residual))
        raise NewtonException()
        return output


def Laplacian(vect,size_x,size_y):
    return ((size_x*size_x)
            * (left_neighboor(vect,size_x,size_y)
               +right_neighboor(vect,size_x,size_y)
               -2*vect)
            + (size_y*size_y)
            * (up_neighboor(vect,size_x,size_y)
               +down_neighboor(vect,size_x,size_y)
               -2*vect)
            )

def Laplacian_mat(size_x,size_y):
    grid = make_grid(size_x,size_y)
    row = np.concatenate([grid]*5, axis = 0)
    col = np.concatenate([grid,
                          left_neighboor(grid,size_x,size_y),
                          right_neighboor(grid,size_x,size_y),
                          down_neighboor(grid,size_x,size_y),
                          up_neighboor(grid,size_x,size_y)],axis=0)
    data = np.array([-2*(size_x*size_x+size_y*size_y)]*(size_x*size_y)
                    +[size_x*size_x]*(2*size_x*size_y)
                    +[size_y*size_y]*(2*size_x*size_y))
    return sp.coo_matrix((data,(row,col)),shape=(size_x*size_y,size_x*size_y))

    ##############################
    ## SOME TOOLS FOR THE TORUS ##
    ##############################

def make_grid(size_x,size_y):
    x,y = np.meshgrid(np.arange(size_x),np.arange(size_y))
    return (size_x*y+x).reshape(-1)

def left_neighboor(vect,size_x,size_y):
    return np.roll(vect.reshape(size_x,size_y),1,axis=1).reshape(-1)

def right_neighboor(vect,size_x,size_y):
    return np.roll(vect.reshape(size_x,size_y),-1,axis=1).reshape(-1)

def down_neighboor(vect,size_x,size_y):
    return np.roll(vect.reshape(size_x,size_y),1,axis=0).reshape(-1)

def up_neighboor(vect,size_x,size_y):
    return np.roll(vect.reshape(size_x,size_y),-1,axis=0).reshape(-1)

def left_neighboor_vect(vect,size_x,size_y):
    return np.roll(vect.reshape(-1,size_x,size_y),1,axis=2).reshape(-1,size_x*size_y)

def right_neighboor_vect(vect,size_x,size_y):
    return np.roll(vect.reshape(-1,size_x,size_y),-1,axis=2).reshape(-1,size_x*size_y)

def down_neighboor_vect(vect,size_x,size_y):
    return np.roll(vect.reshape(-1,size_x,size_y),1,axis=1).reshape(-1,size_x*size_y)

def up_neighboor_vect(vect,size_x,size_y):
    return np.roll(vect.reshape(-1,size_x,size_y),-1,axis=1).reshape(-1,size_x*size_y)

    ###########################################
    ## AN EXAMPLE OF RESOURCES DISTRIBUTIONS ##
    ###########################################

def SmoothCenteredResources(size_x,size_y,coef,radius_1=0.15,radius_2=0.3):
    y,x = np.meshgrid(np.linspace(-0.5,0.5,size_y,endpoint=False),
                      np.linspace(-0.5,0.5,size_x,endpoint=False))
    x += (size_x%2)*(0.5/size_x)
    y += (size_y%2)*(0.5/size_y)
    r = np.sqrt(x*x+y*y).reshape(-1)
    output = np.maximum(r - radius_1,0.)
    output = np.where(output<radius_2-radius_1,
                      ((radius_2-radius_1)**2-(output)**2)**2,
                      0.)
    return (coef /(radius_2-radius_1)**4) * output


    ############################
    ## SOME CUSTOM EXCEPTIONS ##
    ############################

class CustomException(Exception):
    """Raise for custom exception"""
    def __init__(self, msg="Custom Exception", *args, **kwargs):
        super().__init__(msg, *args,**kwargs)
        
class NewtonException(CustomException):
    """Raise when a newton subroutine does not converge"""
    def __init__(self, msg="A Newton method did not converge", *args, **kwargs):
        super().__init__(msg, *args,**kwargs)


if __name__ == "__main__":
    size_x = 50
    size_y = 50
    tot_size = size_x*size_y
    coef_resources = 8./3.
    resources = SmoothCenteredResources(size_x,size_y,coef_resources)

    mu = 0.1
    solver = Logistic_solver(mu,resources,size_x,size_y) 

    theta = solver.solve_logistic()

    # Build grid once 
    x_grid, y_grid = np.meshgrid(
        np.linspace(-0.5, 0.5, size_y, endpoint=False),
        np.linspace(-0.5, 0.5, size_x, endpoint=False),
    )

    fig = plt.figure(figsize=(15,12))
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x_grid, y_grid, theta.reshape(size_x, size_y))


    X = np.stack([x_grid.reshape(-1), y_grid.reshape(-1)], axis=1)
    U = theta.reshape(-1, 1)

    ROOT = Path(__file__).resolve().parents[1]   # repo root (.. from scripts/)
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"mu{mu:.3f}_sx{size_x}_sy{size_y}_coef{coef_resources:.3f}"
    base = out_dir / f"logistic_{tag}"

    plt.savefig(str(base) + ".png", dpi=200)
    plt.close()

    np.savez(
        str(base) + ".npz",
        x=X,
        u=U,
        resources=resources.reshape(-1, 1),
        size_x=size_x,
        size_y=size_y,
        mu=mu,
        coef_resources=coef_resources,
    )

    torch.save(
        {
            "x": torch.tensor(X, dtype=torch.float32),
            "u": torch.tensor(U, dtype=torch.float32),
            "resources": torch.tensor(resources.reshape(-1, 1), dtype=torch.float32),
            "size_x": size_x,
            "size_y": size_y,
            "mu": mu,
            "coef_resources": coef_resources,
        },
        str(base) + ".pt",
    )

    # Generate Allen-Cahn data
    epsilon = 0.1
    allen_cahn_solver = AllenCahnSolver(epsilon, size_x, size_y)
    u_allen_cahn = allen_cahn_solver.solve_allen_cahn()

    fig2 = plt.figure(figsize=(15,12))
    ax2 = fig2.add_subplot(projection='3d')
    ax2.plot_surface(x_grid, y_grid, u_allen_cahn.reshape(size_x, size_y))

    U_ac = u_allen_cahn.reshape(-1, 1)

    tag_ac = f"eps{epsilon:.3f}_sx{size_x}_sy{size_y}"
    base_ac = out_dir / f"allen_cahn_{tag_ac}"

    plt.savefig(str(base_ac) + ".png", dpi=200)
    plt.close()

    np.savez(
        str(base_ac) + ".npz",
        x=X,
        u=U_ac,
        size_x=size_x,
        size_y=size_y,
        epsilon=epsilon,
    )

    torch.save(
        {
            "x": torch.tensor(X, dtype=torch.float32),
            "u": torch.tensor(U_ac, dtype=torch.float32),
            "size_x": size_x,
            "size_y": size_y,
            "epsilon": epsilon,
        },
        str(base_ac) + ".pt",
    )
