from typing import Callable

import torch
from da_tools.observation.data import ObservationSet
from da_tools.system.state import State
from tensordict import TensorDict


class EnKF:
    """Ensemble Kalman Filter implementation for data assimilation."""

    def __init__(self, n_members, n_variables, initial_ensemble_std=1.0):
        """Initialize the EnKF.

        Args:
            n_members (int): Number of ensemble members
            n_variables (int): Number of state variables
            initial_ensemble_std (float): Standard deviation for ensemble perturbations
        """
        self.n_members = n_members
        self.n_variables = n_variables
        self.initial_ensemble_std = initial_ensemble_std
        self.ensemble = None

    def initialize_ensemble(self, initial_state: State) -> State:
        """Initialize ensemble members with perturbations around initial state.

        Args:
            initial_state (State): Initial state for the ensemble mean

        Returns:
            State: State object with batch dimension = n_ensemble
        """

        # Handle both tensor and State inputs
        assert isinstance(initial_state, (State)), "initial_state must be a State"

        initial_tensor = initial_state.fields["x"].squeeze(0)  # Remove batch dimension if present
        time_axis = initial_state.time_axis

        # Generate all noise perturbations at once using vectorized operations
        # Shape: (n_members, *initial_tensor.shape)
        noise_shape = (self.n_members,) + initial_tensor.shape
        noise_perturbations = torch.normal(mean=0.0, std=self.initial_ensemble_std, size=noise_shape)

        # Create ensemble by broadcasting and adding noise to initial state
        # initial_tensor shape: (time_steps, n_variables)
        # noise_perturbations shape: (n_members, time_steps, n_variables)
        # Result shape: (n_members, time_steps, n_variables)
        ensemble_tensor = initial_tensor.unsqueeze(0) + noise_perturbations

        ensemble_fields = TensorDict(
            x=ensemble_tensor,
            batch_size=(self.n_members, 1),
        )
        self.ensemble = State(ensemble_fields, time_axis=time_axis)
        return self.ensemble

    @staticmethod
    def compute_kalman_gain(x, H, R, no_B=True, no_inverse=True):
        ensemble_size, state_dim = x.shape
        xp = x - x.mean(axis=0).reshape(1, state_dim)

        if no_B:  # B == x_T @ x
            """
            We want to compute (ignoring factor of ensemble_size - 1):
            1) HBHt = H @ Xt @ X @ Ht = (H @ Xt) @ (X @ Ht) = (X @ Ht).T @ (X @ Ht)
            2) BHt = Xt @ (X @ Ht)
            """

            xHt = xp @ H.T  # (N_ensembles, N_obs)
            HBHt = xHt.T @ xHt / (ensemble_size - 1)  # (N_obs, N_obs)
            BHt = xp.T @ xHt / (ensemble_size - 1)  # (state_dim, N_obs)
        else:
            B = xp.T @ xp  # (state_dim, state_dim)
            HBHt = H @ B @ H.T

        M = HBHt + R  # (N_obs, N_obs)
        if no_inverse:
            # K = torch.linalg.solve(M, BHt)  # (state_dim, N_obs)
            K = torch.linalg.solve(M.T, BHt.T).T
        else:
            K = B @ H.T @ torch.linalg.inv(M)

        return K

    def forecast_step(self, m_dyn, dt, dynamics_inputs=None, static_inputs=None):
        """Forecast all ensemble members one time step forward using m_dyn.

        Args:
            m_dyn (Callable): Model dynamics function that handles batched State objects
            dt (float): Time step
            dynamics_inputs (State): Dynamic inputs for the model
            static_inputs (State): Static inputs for the model
        """
        # Extract the current state (last time step) from all ensemble members
        # self.ensemble.fields['x'] shape: (n_ensemble, time_steps, n_variables)
        current_ensemble = self.ensemble.fields["x"][:, -1, :]  # Shape: (n_ensemble, n_variables)

        # Create State object with all ensemble members for the current time step
        # Shape: (n_ensemble, 1, n_variables) - batch, time, variables
        current_state_fields = TensorDict(
            x=current_ensemble.unsqueeze(1),  # Add time dimension: (n_ensemble, 1, n_variables)
            batch_size=(self.n_members, 1),
        )

        last_time = self.ensemble.time_axis[-1:]
        current_state = State(current_state_fields, time_axis=last_time)

        # Use m_dyn to forecast all ensemble members at once
        try:
            forecasted_state = m_dyn(current_state, dt, dynamics_inputs, static_inputs)
            # Extract forecasted ensemble: (n_ensemble, 1, n_variables) -> (n_ensemble, n_variables)
            forecasted_ensemble = forecasted_state.fields["x"][:, -1, :]
            new_time = forecasted_state.time_axis
        except Exception as e:
            print(f"Warning: m_dyn failed: {e}")
            # Fallback: use current state (no forecast)
            forecasted_ensemble = current_ensemble
            new_time = last_time + dt

        # Update time axis
        new_time_axis = torch.cat([self.ensemble.time_axis, new_time])

        # Add forecasted state to ensemble history
        # Shape: (n_ensemble, 1, n_variables) to concatenate along time dimension
        forecasted_tensor = forecasted_ensemble.unsqueeze(1)

        # Concatenate with existing ensemble along time dimension
        updated_ensemble_fields = torch.cat([self.ensemble.fields["x"], forecasted_tensor], dim=1)

        # Update ensemble State object
        ensemble_fields = TensorDict(
            x=updated_ensemble_fields, batch_size=(self.n_members, updated_ensemble_fields.shape[1])
        )

        self.ensemble = State(ensemble_fields, time_axis=new_time_axis)

        return forecasted_ensemble  # Return: (n_ensemble, n_variables)

    def analysis_step(self, H, observations, R, time_step_idx):
        """Perform the analysis (update) step using observations.

        Args:
            H (torch.Tensor): Observation matrix (n_obs x n_variables)
            observations (MaskedStateObservationSet): Observation object with state and mask
            R (torch.Tensor): Observation error covariance matrix (n_obs x n_obs)
            time_step_idx (int): Current time step index
        """
        if H.shape[0] == 0:  # No observations available
            return self.ensemble.fields["x"].squeeze(1)  # Return (n_ensemble, n_variables)

        # Extract observations from MaskedStateObservationSet
        # Get the mask for this time step
        mask_t = observations.mask.fields["x"][0, time_step_idx, :]  # shape: (n_variables,)

        # Extract observed values where mask is True
        obs_data = observations.state.fields["x"][0, time_step_idx, mask_t]  # shape: (n_obs,)

        # Convert ensemble State object to matrix form
        X = self.ensemble.fields["x"][:, -1, :].squeeze(1)  # Shape: (n_ensemble, n_variables)

        # Compute Kalman gain using the provided function
        K = self.compute_kalman_gain(X, H, R, no_B=True, no_inverse=True)  # Shape: (n_variables, n_obs)

        # Generate perturbed observations for all ensemble members at once
        # Create standard normal noise and scale it
        obs_std = torch.sqrt(torch.diag(R))  # Shape: (n_obs,)
        # Generate standard normal noise: (n_members, n_obs)
        standard_noise = torch.randn(self.n_members, len(obs_data))
        # Scale by observation standard deviations
        obs_noise_matrix = standard_noise * obs_std.unsqueeze(0)  # Broadcasting: (n_members, n_obs)

        # Broadcast observations and add noise for all ensemble members
        # obs_data shape: (n_obs,) -> (1, n_obs) -> (n_ensemble, n_obs)
        y_pert_matrix = obs_data.unsqueeze(0) + obs_noise_matrix  # Shape: (n_ensemble, n_obs)
        # Compute innovations for all ensemble members at once
        # H @ X.T gives observation predictions for all members: (n_obs, n_ensemble)
        # We need (n_ensemble, n_obs), so we use X @ H.T
        H_X = X @ H.T  # Shape: (n_ensemble, n_obs)
        innovations = y_pert_matrix - H_X  # Shape: (n_ensemble, n_obs)
        # Update all ensemble members at once
        # K @ innovations.T gives updates for all members: (n_variables, n_ensemble)
        # We need (n_ensemble, n_variables), so we use innovations @ K.T
        updates = innovations @ K.T  # Shape: (n_ensemble, n_variables)
        updated_ensemble = X + updates  # Shape: (n_ensemble, n_variables)

        # Update the last time step of self.ensemble.fields['x'] with updated_ensemble
        self.ensemble.fields["x"][:, -1, :] = updated_ensemble

        # Return updated ensemble as tensor for compatibility
        return updated_ensemble  # Shape: (n_ensemble, n_variables)

    def assimilate(
        self,
        m_dyn: Callable,
        observations: ObservationSet,
        obs_op: ObservationSet,
        x_init: State = None,
        dynamics_inputs: State = None,
        static_inputs: State = None,
        verbose: bool = False,
    ) -> tuple:
        """Main EnKF assimilation method.

        Args:
            m_dyn (Callable): Model dynamics function
            observations (ObservationSet): Set of observations to be assimilated
            obs_op (ObservationSet): Observation operator used to map between system states and observations
            x_init (State): Initial state for the ensemble mean
            dynamics_inputs (State): Extra inputs defined for each input state (e.g. TOA)
            static_inputs (State): Extra inputs that don't vary over time (e.g. bathymetry)
            verbose (bool): If True, print detailed progress and diagnostic information

        Returns:
            - ensemble_state (State): Full ensemble State object with history (Default option)
        """
        # Initialize ensemble
        if x_init is None:
            raise ValueError("x_init must be provided to initialize the ensemble")

        self.initialize_ensemble(x_init)

        # Get time step from observation operator - throw error if insufficient time steps
        if len(obs_op.time_axis) < 2:
            raise ValueError(
                f"Observation operator time_axis must have at least 2 time steps to compute dt, "
                f"but got {len(obs_op.time_axis)} time step(s). "
                f"Cannot determine time step for assimilation."
            )

        # Store initial ensemble state
        current_ensemble = self.ensemble.fields["x"][:, -1, :].squeeze(1)  # Shape: (n_ensemble, n_variables)

        for t_idx in range(len(obs_op.time_axis)):
            # Forecast step using m_dyn (except for first iteration)
            if t_idx > 0:
                dt = (
                    obs_op.time_axis[t_idx + 1] - obs_op.time_axis[t_idx]
                    if t_idx < len(obs_op.time_axis) - 1
                    else obs_op.time_axis[t_idx] - obs_op.time_axis[t_idx - 1]
                )
                self.forecast_step(m_dyn, dt, dynamics_inputs, static_inputs)

            # Linearize observation operator at current time step to get H matrix
            H_state = obs_op.linearize(idx_time=t_idx)
            H_t = H_state.fields["x"][0, 0, :, :]  # Extract H matrix for time t. Shape: (n_obs, n_variables)

            # Create observation error covariance matrix R_t
            if H_t.shape[0] > 0:  # Only if there are observations
                # Get the mask for this time step to identify observed variables
                mask_t = observations.mask.fields["x"][0, t_idx, :]  # Shape: (n_variables,)
                # Extract sigma values only at observed locations
                sigma_obs = obs_op.sigma.fields["x"][0, t_idx, mask_t]  # Shape: (n_obs,)
                # Create diagonal R matrix with variances (sigma^2) at observed locations only
                R_t = torch.diag(sigma_obs**2)  # Shape: (n_obs, n_obs)
                self.analysis_step(H_t, observations, R_t, t_idx)
            else:
                print("  No observations available, skipping analysis step")

            # Update current ensemble state
            current_ensemble = self.ensemble.fields["x"][:, -1, :].squeeze(1)  # Shape: (n_ensemble, n_variables)

        # Final summary
        final_spread = current_ensemble.std(dim=0).mean()

        if verbose:
            final_mean = current_ensemble.mean(dim=0)
            print("Final ensemble statistics:")
            print(f"  Number of members: {self.n_members}")
            print(f"  State dimension: {self.n_variables}")
            print(f"  Final spread: {final_spread:.4f}")
            print(f"  Final mean range: [{final_mean.min():.4f}, {final_mean.max():.4f}]")

        return self.ensemble

    def get_ensemble_statistics(self):
        """Compute ensemble statistics.

        Returns:
            dict: Dictionary containing ensemble mean, std, and individual members
        """
        if self.ensemble is None:
            return None

        # Extract ensemble matrix from State object
        # Get the last time step: shape (n_ensemble, n_variables)
        ensemble_matrix = self.ensemble.fields["x"][:, -1, :]

        return {
            "mean": ensemble_matrix.mean(dim=0),  # Mean across ensemble members
            "std": ensemble_matrix.std(dim=0),  # Std across ensemble members
            "members": ensemble_matrix,  # Individual ensemble members
            "spread": ensemble_matrix.std(dim=0).mean(),  # Average spread across variables
            "full_history": self.ensemble.fields["x"],  # Full ensemble history
            "time_axis": self.ensemble.time_axis,  # Time axis
            "n_time_steps": self.ensemble.fields["x"].shape[1],  # Number of time steps
        }


def enkf(
    m_dyn: Callable,
    observations: ObservationSet,
    obs_op: ObservationSet,
    x_init: State = None,
    dynamics_inputs: State = None,
    static_inputs: State = None,
    n_members: int = 20,
    initial_ensemble_std: float = 1.0,
    verbose: bool = False,
) -> tuple:
    """Perform ensemble Kalman filter data assimilation using EnKF.assimilate() method.

    This function is a wrapper around the EnKF.assimilate() method for backwards compatibility
    and consistent API design.

    Args:
        m_dyn (Callable): Model dynamics function
        observations (ObservationSet): Set of observations to be assimilated
        obs_op (ObservationSet): Observation operator used to map between system states and observations
        x_init (State): Initial state for the ensemble mean
        dynamics_inputs (State): Extra inputs defined for each input state (e.g. TOA)
        static_inputs (State): Extra inputs that don't vary over time (e.g. bathymetry)
        n_members (int): Number of ensemble members
        initial_ensemble_std (float): Standard deviation for ensemble perturbations
        verbose (bool): If True, print detailed progress and diagnostic information

    Returns:
            - ensemble_state (State): Full ensemble State object with history (Default option)
    """
    # Simply call the assimilate method of the EnKF object
    enkf_obj = EnKF(
        n_members=n_members, n_variables=x_init.fields["x"].shape[-1], initial_ensemble_std=initial_ensemble_std
    )
    return enkf_obj.assimilate(
        m_dyn=m_dyn,
        observations=observations,
        obs_op=obs_op,
        x_init=x_init,
        dynamics_inputs=dynamics_inputs,
        static_inputs=static_inputs,
        verbose=verbose,
    )
