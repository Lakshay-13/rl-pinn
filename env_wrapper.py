import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from neurodiffeq import diff
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.generators import BaseGenerator, Generator1D, PredefinedGenerator
from neurodiffeq.conditions import BundleIVP, NoCondition, BundleDirichletBVP
from ordered_set import OrderedSet

from core.device import get_device

logger = logging.getLogger(__name__)

# --- Helper Functions & Classes from CINNS.py ---

def V_or(phi, phim=1.0, phiq=10.0):
    # Ensure phi is float32 to avoid double promotion
    phi = phi.float()
    term1 = 6 * (2*phi)**2 * (8 + 2*(2*phi)**2/(phim)**2 - 3*(2*phi)**4 / phiq )**2 
    term2 = (96 +8*(2*phi)**2 + (2*phi)**4/phim**2 - (2*phi)**6/phiq)**2 
    # Explicitly return float32
    res = (1.0/4.0)*(1.0/768.0)*(term1 - term2)
    return res.float()

class VPhi(nn.Module):
    def __init__(self):
        super(VPhi, self).__init__()
        n_nodes = 16
        self.fc1 = nn.Linear(1,n_nodes )
        self.fc2 = nn.Linear(n_nodes, n_nodes)
        self.fc3 = nn.Linear(n_nodes, n_nodes)
        self.fc4 = nn.Linear(n_nodes, n_nodes)
        self.fc5 = nn.Linear(n_nodes, 1)
        
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        x = torch.nn.SiLU()(self.fc1(x))
        x = torch.nn.SiLU()(self.fc2(x))
        x = torch.nn.SiLU()(self.fc3(x))
        x = torch.nn.SiLU()(self.fc4(x))
        x = self.fc5(x)
        return x.float()


class LocalizedMLP(nn.Module):
    """Small MLP with a learnable Gaussian or compact bump gate."""

    def __init__(
        self,
        n_input_units,
        hidden_units,
        n_output_units=1,
        actv=nn.Softplus,
        coord_index=0,
        coord_min=0.0,
        coord_max=1.0,
        kernel="gaussian",
        initial_sigma=0.25,
        sigma_min=0.05,
        sigma_max=0.75,
    ):
        super().__init__()
        if not hidden_units:
            raise ValueError("hidden_units must be non-empty")
        if kernel not in {"gaussian", "bump"}:
            raise ValueError(f"Unsupported localisation kernel: {kernel}")
        self.coord_index = int(coord_index)
        self.coord_min = float(coord_min)
        self.coord_max = float(coord_max)
        self.kernel = kernel
        self.initial_sigma = float(initial_sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.fc_in = nn.Linear(n_input_units, hidden_units[0])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(hidden_units[index], hidden_units[index + 1])
            for index in range(len(hidden_units) - 1)
        )
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)
        self.actv = actv()
        self.mu = nn.Parameter(torch.linspace(self.coord_min, self.coord_max, hidden_units[0]))
        self.sigma = nn.Parameter(torch.full((hidden_units[0],), self.initial_sigma))

    def reset_localization_parameters(self):
        with torch.no_grad():
            self.mu.copy_(torch.linspace(self.coord_min, self.coord_max, self.mu.numel(), device=self.mu.device))
            self.sigma.fill_(self.initial_sigma)

    def _gate(self, coordinate):
        sigma = self.sigma.abs().clamp(self.sigma_min, self.sigma_max).view(1, -1)
        mu = self.mu.view(1, -1)
        z = (coordinate - mu) / sigma
        if self.kernel == "gaussian":
            return torch.exp(-0.5 * z.square())
        inside = (z.abs() < 1.0).to(z.dtype)
        bump = torch.exp(-1.0 / (1.0 - z.square()).clamp_min(1e-6))
        return inside * bump * float(np.e)

    def forward(self, x):
        coordinate = x[..., self.coord_index : self.coord_index + 1].float()
        out = self.actv(self.fc_in(x.float()))
        out = out * self._gate(coordinate)
        for layer in self.hidden_layers:
            out = self.actv(layer(out))
        return self.fc_out(out).float()


class LocalizedVPhi(LocalizedMLP):
    def __init__(self, kernel="gaussian"):
        super().__init__(
            n_input_units=1,
            hidden_units=(16, 16, 16, 16),
            n_output_units=1,
            actv=nn.SiLU,
            coord_index=0,
            coord_min=0.0,
            coord_max=1.8,
            kernel=kernel,
            initial_sigma=0.35,
            sigma_min=0.08,
            sigma_max=1.0,
        )

class MeshGenerator(BaseGenerator):
    def __init__(self, g1, pg, device=None):
        super(MeshGenerator, self).__init__()
        self.g1 = g1
        self.pg = pg
        self.device = device

    def get_examples(self):
        u = self.g1.get_examples()
        u = u.reshape(-1, 1, 1)

        bundle_params = self.pg.get_examples()
        if isinstance(bundle_params, torch.Tensor):
            bundle_params = (bundle_params,)
        
        n_params = len(bundle_params)
        bundle_params = torch.stack(bundle_params, dim=1)
        bundle_params = bundle_params.reshape(1, -1, n_params)

        uu, bb = torch.broadcast_tensors(u, bundle_params)
        uu = uu[:, :, 0].reshape(-1)
        bb = [bb[:, :, i].reshape(-1) for i in range(n_params)]
        
        if self.device:
            return uu.float().to(self.device), *[b.float().to(self.device) for b in bb]
        return uu.float(), *[b.float() for b in bb]

class SolverWithAdditionalLoss(BundleSolver1D):
    def __init__(self, ode_system, conditions, t_min, t_max,
                 yago_data, optim_state, device, **kwargs):
        super().__init__(ode_system, conditions, t_min, t_max, **kwargs)
        self.yago_data = yago_data
        self.optim_state = optim_state
        self.device = device
        
        # Move Yago data to device immediately
        self.yago_u = torch.tensor(yago_data['u'], dtype=torch.float32, device=device).reshape(-1, 1)
        self.yago_A = torch.tensor(yago_data['A'], dtype=torch.float32, device=device).reshape(-1, 1)
        self.yago_Sigma = torch.tensor(yago_data['Sigma'], dtype=torch.float32, device=device).reshape(-1, 1)
        self.yago_Phi = torch.tensor(yago_data['Phi'], dtype=torch.float32, device=device).reshape(-1, 1)
        
        # Fixed constants for Yago loss (from CINNS.py)
        self.Sigma_uh_fixed = 0.8044 * torch.ones_like(self.yago_u, device=device)
        self.Va_uh_fixed = -10.0369 * torch.ones_like(self.yago_u, device=device)

    def additional_loss(self, residuals, funcs, coords):
        self.optim_state['cnt'] += 1
        
        # Get current model prediction on Yago points
        # Solver1D.get_solution returns a function. We can use self.nets directly or call get_solution.
        # Calling get_solution allows us to pass coords.
        
        # Note: BundleSolver1D works with (t, *params). 
        # Here we want to evaluate at specific points (yago_u) with specific params (Sigma_uh, Va_uh).
        
        solution = self.get_solution(copy=False, best=False)
        
        # Evaluate model at Yago points
        # model(u, Sigma_uh, Va_uh) -> returns (Vs, Va, Vp, Sigma, A, phi)
        # Note: solution function expects flattened inputs usually, or broadcastable.
        
        # We need to flatten our fixed inputs for the solution function
        u_flat = self.yago_u.reshape(-1)
        S_flat = self.Sigma_uh_fixed.reshape(-1)
        V_flat = self.Va_uh_fixed.reshape(-1)
        
        # neurodiffeq solution returns tuple of tensors
        # order from equations: Vs, Va, Vp, Sigma, A, phi
        out = solution(u_flat, S_flat, V_flat)
        # Unpack relevant outputs
        Sigma_u = out[3]
        A_u = out[4]
        phi_u = out[5]
        
        # Ensure shapes match for MSE
        aux_loss_yago = (torch.mean((A_u.reshape(-1, 1) - self.yago_A)**2) 
             + torch.mean((Sigma_u.reshape(-1, 1) - self.yago_Sigma)**2) 
             + torch.mean((phi_u.reshape(-1, 1) - self.yago_Phi)**2))
        
        # Decay factor from CINNS.py
        decay = np.exp(-0.001 * self.optim_state['cnt'])
        return torch.tensor(decay, dtype=torch.float32, device=self.device) * aux_loss_yago

# --- Main Environment Wrapper ---

class TorchHoloEnv:
    def __init__(self, cinns_path=".", device=None, localization_target=None, localization_kernel=None):
        self.cinns_path = Path(cinns_path)
        self.device = device if device else get_device()
        self.logger = logging.getLogger(__name__)
        self.localization_target = str(
            localization_target
            if localization_target is not None
            else os.environ.get("RLPINN_LOCALIZATION_TARGET", "none")
        ).strip().lower()
        self.localization_kernel = str(
            localization_kernel
            if localization_kernel is not None
            else os.environ.get("RLPINN_LOCALIZATION_KERNEL", "gaussian")
        ).strip().lower()
        if self.localization_target not in {"none", "v", "pinn", "both"}:
            raise ValueError(f"Unsupported localisation target: {self.localization_target}")
        if self.localization_kernel not in {"gaussian", "bump"}:
            raise ValueError(f"Unsupported localisation kernel: {self.localization_kernel}")
        
        # Force float32 default to avoid Double promotion from numpy/constants
        torch.set_default_dtype(torch.float32)
        
        # If CUDA, set current device for this process
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        
        # Load Data
        self._load_data()
        
        # Initialize RL state
        self.s_v = np.ones(7, dtype=np.float32)
        self.optim_state = {'cnt': 0}
        self.best_full_state = None
        self.best_full_loss = None
        
        # PDE Setup
        self._setup_pde()
        
    def _load_data(self):
        # Load Curve Data
        curve_path = self.cinns_path / "Data/Curve_sofT_case3of5.csv"
        df_data = pd.read_csv(curve_path, header=None).values
        # From CINNS.py: [45:, 1] for S, [45:, 0] for T
        self.S_true = torch.tensor(df_data[45:, 1], dtype=torch.float32, device=self.device)
        self.T_true = torch.tensor(df_data[45:, 0], dtype=torch.float32, device=self.device)
        
        # Prepare params
        step = 7
        # Use torch.pi (or math.pi) and explicit cast to float32 to avoid Double promotion
        pi = torch.acos(torch.zeros(1)).item() * 2 
        self.Sigma_uh_all = ((self.S_true / pi)**(1/3))[::step].float()
        self.Va_uh_all = (-self.T_true * 4 * pi)[::step].float()
        
        # Load Yago Data
        self.yago_data = {}
        for name in ['A', 'Sigma', 'Phi']:
            path = self.cinns_path / f"Data/Yago_{name}.csv"
            df = pd.read_csv(path, header=None).values
            # col 0 is u, col 1 is value
            self.yago_data['u'] = df[:, 0].astype(np.float32)
            self.yago_data[name] = df[:, 1].astype(np.float32)

    def _setup_pde(self):
        # Constants
        self.curriculum = 1.0
        self.delta = 0.0
        self.n_u_points = 32
        
        # Generators
        # Ensure generators produce data on the correct device
        self.pg = PredefinedGenerator(self.Sigma_uh_all.cpu(), self.Va_uh_all.cpu())
        self.valid_pg = PredefinedGenerator(self.Sigma_uh_all.cpu(), self.Va_uh_all.cpu())
        
        self.g1 = Generator1D(self.n_u_points, 0, self.curriculum, method='chebyshev2')
        self.g2 = Generator1D(16, 0, 1.0, method='equally-spaced')
        
        self.train_generator = MeshGenerator(self.g1, self.pg, device=self.device)
        self.valid_generator = MeshGenerator(self.g2, self.valid_pg, device=self.device)
        
        # Networks. Localised variants are selected explicitly for ablations or
        # through RLPINN_LOCALIZATION_* in detached experiment processes.
        if self.localization_target in {"pinn", "both"}:
            net_factory = lambda: LocalizedMLP(
                n_input_units=3,
                hidden_units=(16, 16),
                n_output_units=1,
                actv=nn.Softplus,
                coord_index=0,
                coord_min=0.0,
                coord_max=1.0,
                kernel=self.localization_kernel,
            )
            self.nets = [net_factory() for _ in range(6)]
        else:
            self.nets = [
                FCNN(n_input_units=3, n_output_units=1, hidden_units=[16, 16], actv=nn.Softplus)
                for _ in range(6)
            ]
        self.V = (
            LocalizedVPhi(kernel=self.localization_kernel)
            if self.localization_target in {"v", "both"}
            else VPhi()
        )
        
        # Move to device and ensure float32
        for net in self.nets:
            net.to(self.device).float()
        self.V.to(self.device).float()
        
        # Conditions
        self.conditions = [
            NoCondition(), # Vs
            BundleIVP(1, None, bundle_param_lookup=dict(u_0=1)), # Va
            BundleIVP(0, 1), # Vp
            BundleDirichletBVP(0, 1, 1, None, bundle_param_lookup=dict(u_1=0)), # Sigma
            BundleDirichletBVP(0, 1, 1, 0), # A
            BundleIVP(0, 0), # phi
        ]
        
        # Optimizer (Default to Adam, will be overwritten by Step)
        params = list(self.V.parameters())
        for net in self.nets:
            params += list(net.parameters())
        
        # Use foreach=False to avoid device/dtype mismatch errors
        self.optimizer = torch.optim.Adam(params, lr=1e-4, foreach=False) 
        
        # Solver
        self.solver = SolverWithAdditionalLoss(
            ode_system=self.equations,
            conditions=self.conditions,
            t_min=self.delta,
            t_max=1.0,
            yago_data=self.yago_data,
            optim_state=self.optim_state,
            device=self.device,
            train_generator=self.train_generator,
            valid_generator=self.valid_generator,
            optimizer=self.optimizer,
            nets=self.nets,
            n_batches_valid=0,
            eq_param_index=()
        )
        
    def equations(self, Vs, Va, Vp, Sigma, A, phi, u):
        # V_or and VPhi are closures or available globally
        # Ensure all inputs are float32
        phi = phi.float()
        u = u.float()
        
        # Derivative of V(phi)
        VF = diff(self.V(phi), phi, shape_check=False)
        
        # Terms
        V_phi_val = self.V(phi)
        V_or_val = V_or(phi)
        
        # Equations
        eq1 = Vs - diff(Sigma, u, order=1)  
        eq2 = Va - diff(A, u, order=1)
        eq3 = Vp - diff(phi, u, order=1)
        eq4 = diff(Vs, u,  order=1) + 2 / 3 * Sigma * Vp ** 2  
        eq5 = ((u ** 2) * Sigma * diff(Va, u, order=1) + 8 / (3) * (V_phi_val) * Sigma
                                    + Va * (3 * u ** 2 * Vs - 5 * Sigma * u)
                                    + A * (8 * Sigma - 6 * u * Vs))

        eq6 = (u ** 2 * Sigma * A * diff(Vp, u, order=1) - Sigma * (VF)
              + Vp * (-3 * u * A * Sigma + u ** 2 * Sigma * Va + 3 * u ** 2 * A * Vs))

        eq7 =  ((u * Vs - Sigma) *
            ( u**2 * Sigma * Va + 2 * A * u**2 * Vs - 4 * u * A * Sigma)
            -(2/3)*(u*Sigma**2)*(u**2 * A* Vp**2 - 2 * (V_phi_val)))
            
        eq_list = [eq1, eq2, eq3, eq4, eq5, eq6, eq7]
        
        weighted_eqs = []
        for i, eq in enumerate(eq_list):
            w = torch.tensor(self.s_v[i], dtype=torch.float32, device=self.device)
            weighted_eqs.append(w * eq)
            
        return weighted_eqs

    def reset(self):
        # Reset Network Weights (Xavier init is standard, or just re-instantiate)
        for net in self.nets:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        
        # Reset VPhi
        for m in self.V.modules():
             if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        for module in [*self.nets, self.V]:
            reset_localization = getattr(module, "reset_localization_parameters", None)
            if reset_localization is not None:
                reset_localization()

        # Reset State
        self.optim_state['cnt'] = 0
        self.s_v = np.ones(7, dtype=np.float32)
        self.best_full_state = None
        self.best_full_loss = None
        
        # Reset Solver History
        self.solver.metrics_history = defaultdict(list, {'r2_loss': [], 'phi_max': [], 'train_loss': [], 'valid_loss': []})
        
        # Run initial fit (warmup) like CINNS_RL does
        # "self.c.opts.scale=2 #just for the first iteration"
        self._fit_with_full_best_tracking(max_epochs=50)
        
        return self.get_residuals()

    def step(self, action_dict):
        # action_dict contains: lr, betas, optimizer_name, s_v, epochs

        lr = float(action_dict.get('lr', 1e-3))
        if not np.isfinite(lr) or lr < 1e-9:
            lr = 1e-9
        betas = action_dict.get('betas', (0.9, 0.999))
        opt_name = action_dict.get('optimizer', 'adam')
        new_s_v = action_dict.get('s_v', np.ones(7))
        epochs = action_dict.get('epochs', 50)

        # Update Weights
        self.s_v = new_s_v

        # Update Optimizer Hyperparameters (Maintain state/momentum)
        # We modify the existing optimizer object instead of re-instantiating it.
        for group in self.optimizer.param_groups:
            group['lr'] = lr
            if opt_name.lower() != 'lbfgs':
                group['betas'] = betas

        # Capture old loss BEFORE fit by computing it directly
        # We compute the UNWEIGHTED loss to provide a consistent baseline for the reward
        old_loss = self._compute_unweighted_loss()

        # Helper to avoid recreating bar
        self._fit_with_full_best_tracking(max_epochs=epochs)

        # Capture new loss AFTER fit (weighted loss for solver tracking)
        new_loss = self._compute_current_loss()

        residuals = self.get_residuals()

        return residuals, new_loss, old_loss

    def _compute_unweighted_loss(self, *, best: bool = False) -> float:
        """Compute the unweighted physics loss (Mean Squared Residuals)."""
        try:
            u, sigma, Va = self.valid_generator.get_examples()
            u = u.to(self.device)
            sigma = sigma.to(self.device)
            Va = Va.to(self.device)

            # Get raw residuals from neurodiffeq (they are unweighted at the output level)
            residuals = self._solver_residuals(u, sigma, Va, best=best)
            if residuals:
                total_mse = 0.0
                for res in residuals:
                    if res is not None:
                        total_mse += float((res ** 2).mean().item())
                return total_mse / len(residuals)
        except Exception as e:
            logger.error(f"Failed to compute unweighted loss: {e}")
        
        return 1e9

    def _compute_current_loss(self) -> float:
        """Compute the current loss directly from the solver's metrics or by evaluating the solution."""
        # First try to get from metrics_history (preferred if available)
        metrics = self.solver.metrics_history

        # Check valid_loss first (validation loss is more meaningful)
        if len(metrics.get('valid_loss', [])) > 0:
            val = metrics['valid_loss'][-1]
            loss = float(val.item() if hasattr(val, 'item') else val)
            if loss > 0 and loss < 1e9:
                return loss

        # Check train_loss
        if len(metrics.get('train_loss', [])) > 0:
            val = metrics['train_loss'][-1]
            loss = float(val.item() if hasattr(val, 'item') else val)
            if loss > 0 and loss < 1e9:
                return loss

        # Check r2_loss
        if len(metrics.get('r2_loss', [])) > 0:
            val = metrics['r2_loss'][-1]
            loss = float(val.item() if hasattr(val, 'item') else val)
            if loss > 0 and loss < 1e9:
                return loss

        # Fallback: compute loss by evaluating the solution
        # Get residuals on validation set and compute MSE
        try:
            u, sigma, Va = self.valid_generator.get_examples()
            u = u.to(self.device)
            sigma = sigma.to(self.device)
            Va = Va.to(self.device)

            residuals = self._solver_residuals(u, sigma, Va, best=False)
            if residuals:
                # Compute MSE of all residual equations
                total_loss = 0.0
                count = 0
                for res in residuals:
                    if res is not None:
                        total_loss += float((res ** 2).mean().item())
                        count += 1
                if count > 0:
                    return total_loss / count
        except Exception:
            pass

        return 1e9  # Default high loss if unable to compute

    def get_residuals(self, best=False):
        # Get residuals on VALIDATION set (like CINNS)
        u, sigma, Va = self.valid_generator.get_examples()
        # Move to device? Neurodiffeq solver usually expects inputs on same device as nets
        u = u.to(self.device)
        sigma = sigma.to(self.device)
        Va = Va.to(self.device)
        
        res = self._solver_residuals(u, sigma, Va, best=best)
        # res is list of tensors
        # Target shape: (7, 16, 41) roughly?
        # CINNS.py: res_eq[i, :,:] =r.cpu().detach().reshape(16, 41)
        # My generator uses 16 points for u (g2), and len(pg) params.
        # len(pg) = len(Sigma_uh_all) / step = (approx 48/7? no)
        # In CINNS.py: step=7. S_true[45:]
        
        # Let's just return what we get, reshaped to (7, -1) or keep as is.
        # RL Agent expects (7, 16, 41).
        # We need to match this shape for the RL agent's CNN.
        
        # If shape mismatch, we might need to pad or resize.
        # But for now, let's trust the logic replicates CINNS.
        
        res_stacked = torch.stack(res) # (7, N)
        # Reshape to (7, 16, 41) - 16 u points, 41 parameter points
        return res_stacked.view(7, 16, 41).detach()

    def get_loss(self):
        # Current loss - check multiple metrics in priority order
        # neurodiffeq may track train_loss, valid_loss, or r2_loss
        metrics = self.solver.metrics_history

        # Prefer valid_loss (most reliable for evaluation)
        if len(metrics.get('valid_loss', [])) > 0:
            val = metrics['valid_loss'][-1]
            return float(val.item() if hasattr(val, 'item') else val)

        # Fall back to train_loss
        if len(metrics.get('train_loss', [])) > 0:
            val = metrics['train_loss'][-1]
            return float(val.item() if hasattr(val, 'item') else val)

        # Fall back to r2_loss
        if len(metrics.get('r2_loss', [])) > 0:
            val = metrics['r2_loss'][-1]
            return float(val.item() if hasattr(val, 'item') else val)

        return 1e9 # High initial loss

    def render(self, save_dir):
        from utils.plotting import plot_loss, plot_residuals, plot_potential
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Plot Loss
        plot_loss(self.solver.metrics_history, save_dir / "loss.png")
        
        # 2. Plot Residuals
        current_residuals = self.get_residuals(best=False).cpu().detach().numpy()
        plot_residuals(current_residuals, save_dir / "residuals.png")
        best_available = self.best_full_state is not None
        if best_available:
            best_residuals = self.get_residuals(best=True).cpu().detach().numpy()
            plot_residuals(best_residuals, save_dir / "residuals_best.png")
        
        # 3. Plot Potential
        phi_np, v_current, v_theory = self.get_potential_curve(best=False)
        _, v_best, _ = self.get_potential_curve(best=True) if best_available else (phi_np, None, v_theory)
        plot_potential(phi_np, v_current, v_theory, save_dir / "potential.png", v_best_vals=v_best)

        current_validation = self._best_metric_value()
        current_unweighted = self._compute_unweighted_loss(best=False)
        best_validation = float(self.best_full_loss) if best_available and self.best_full_loss is not None else None
        best_unweighted = self._compute_unweighted_loss(best=True) if best_available else None
        report = {
            "best_available": bool(best_available),
            "loss": {
                "current_validation": float(current_validation) if current_validation is not None else None,
                "current_unweighted": float(current_unweighted) if np.isfinite(current_unweighted) else None,
                "best_validation": best_validation,
                "best_unweighted": float(best_unweighted) if best_unweighted is not None and np.isfinite(best_unweighted) else None,
            },
            "potential": {
                "phi": np.asarray(phi_np, dtype=np.float32).tolist(),
                "current": np.asarray(v_current, dtype=np.float32).tolist(),
                "best": np.asarray(v_best, dtype=np.float32).tolist() if v_best is not None else None,
                "theory": np.asarray(v_theory, dtype=np.float32).tolist(),
            },
        }
        (save_dir / "render_report.json").write_text(json.dumps(report, indent=2))
        return report

    def _capture_full_state(self):
        return {
            "nets_state": [deepcopy(net.state_dict()) for net in self.nets],
            "v_state": deepcopy(self.V.state_dict()),
            "s_v": np.asarray(self.s_v, dtype=np.float32).copy(),
        }

    def _restore_full_state(self, state):
        for net, net_state in zip(self.nets, state["nets_state"]):
            net.load_state_dict(net_state)
        self.V.load_state_dict(state["v_state"])
        self.s_v = np.asarray(state["s_v"], dtype=np.float32).copy()

    def _best_metric_value(self):
        metrics = self.solver.metrics_history
        for key in ("valid_loss", "train_loss", "r2_loss"):
            values = metrics.get(key, [])
            if values:
                value = values[-1]
                loss = float(value.item() if hasattr(value, "item") else value)
                if np.isfinite(loss):
                    return loss
        return None

    def _track_full_best_state(self, solver):
        current_loss = self._best_metric_value()
        if current_loss is None:
            return
        if self.best_full_loss is None or current_loss < self.best_full_loss:
            self.best_full_loss = current_loss
            self.best_full_state = self._capture_full_state()

    def _fit_with_full_best_tracking(self, max_epochs):
        self.solver.fit(
            max_epochs=max_epochs,
            tqdm_file=None,
            callbacks=[self._track_full_best_state],
        )
        if self.best_full_state is None:
            self.best_full_loss = self._compute_current_loss()
            self.best_full_state = self._capture_full_state()

    def _solver_residuals(self, u, sigma, Va, *, best=False):
        if not best or self.best_full_state is None:
            return self.solver.get_residuals(u, sigma, Va, best=False)

        current_state = self._capture_full_state()
        try:
            self._restore_full_state(self.best_full_state)
            return self.solver.get_residuals(u, sigma, Va, best=False)
        finally:
            self._restore_full_state(current_state)

    def get_potential_curve(self, best=False):
        phi = torch.linspace(0, 1.8, 100, device=self.device, dtype=torch.float32).reshape(-1, 1)
        phi_np = phi.cpu().detach().numpy().flatten()
        v_theory = V_or(phi).cpu().detach().numpy().flatten()

        if best and self.best_full_state is not None:
            current_state = self._capture_full_state()
            try:
                self._restore_full_state(self.best_full_state)
                v_vals = self.V(phi).cpu().detach().numpy().flatten()
            finally:
                self._restore_full_state(current_state)
        else:
            v_vals = self.V(phi).cpu().detach().numpy().flatten()

        return phi_np, v_vals, v_theory

    def to_device(self, device):
        """Moves all environment components to the specified device."""
        self.device = device
        
        # Move Reference Data
        self.S_true = self.S_true.to(device)
        self.T_true = self.T_true.to(device)
        
        # Move Solver Data
        self.solver.device = device
        self.solver.yago_u = self.solver.yago_u.to(device)
        self.solver.yago_A = self.solver.yago_A.to(device)
        self.solver.yago_Sigma = self.solver.yago_Sigma.to(device)
        self.solver.yago_Phi = self.solver.yago_Phi.to(device)
        self.solver.Sigma_uh_fixed = self.solver.Sigma_uh_fixed.to(device)
        self.solver.Va_uh_fixed = self.solver.Va_uh_fixed.to(device)

        # Move Networks
        for net in self.nets:
            net.to(device)
        self.V.to(device)
        
        # Update MeshGenerators
        self.train_generator.device = device
        self.valid_generator.device = device
        
        # Move Optimizer state
        if hasattr(self, 'optimizer'):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        
        logger.info(f"Environment moved to {device}")
