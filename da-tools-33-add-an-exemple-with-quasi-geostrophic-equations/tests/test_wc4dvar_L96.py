from functools import partial
from unittest import TestCase

import torch
from da_tools.observation.operators import random_sparse_noisy_obs
from da_tools.probability.distributions import DiagonalGaussian
from da_tools.system.state import State
from da_tools.util.initialization import naive_initialization
from da_tools.variational.base import assemble_analysis
from da_tools.variational.weak_constraint_4dvar import wc4dvar_single_window, wc4dvar_sliding_window
from mdml_sim.lorenz96 import L96Simulator
from tensordict import TensorDict


def setup_task(
    n_variables=40,
    time_step=0.01,
    nb_steps=800,
    nb_steps_burnt=400,
    noise_amplitude=1.0,
    p_obs=0.25,
    forcing=8.0,
):
    groundtruth = generate_groundtruth(
        n_variables, time_step, nb_steps, nb_steps_burnt=nb_steps_burnt, forward_operator=L96Simulator(forcing=forcing)
    )
    groundtruth, obs_op, observations = random_sparse_noisy_obs(groundtruth, noise_amplitude, p_obs)
    T = groundtruth.time_axis.nelement()

    model_error_shape = (1, T - 1, n_variables)
    model_error_distribs = DiagonalGaussian(
        torch.zeros(*model_error_shape),
        torch.ones(*model_error_shape),
    )

    m_dyn = partial(next_step_function, L96Simulator(forcing=forcing))

    initialization = naive_initialization(observations)

    return groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization


def generate_groundtruth(
    n_variables, time_step, nb_steps, nb_steps_burnt=400, forward_operator=L96Simulator(forcing=8)
):
    initial_state = torch.randn(1, n_variables)
    nb_steps_total = nb_steps + nb_steps_burnt
    forecast_steps = torch.arange(0, nb_steps_total) * torch.tensor(time_step, dtype=torch.float64)
    time_series = forward_operator.integrate(time=forecast_steps, state=initial_state).squeeze()
    time_series = time_series[nb_steps_burnt:]
    return State(time_series.reshape(1, *time_series.shape), time_axis=forecast_steps[:nb_steps])


def next_step_function(forward_operator, x: State, dt: float, dynamic_inputs: State, static_inputs: State):
    B, T = x.fields.batch_size[:2]
    x_tensor = x.fields.reshape(-1)["x"]  # combine batch and time dimensions, leave others intact
    integrated = forward_operator.integrate(torch.arange(2) * dt, x_tensor)[:, 1]
    integrated = integrated.unsqueeze(0)  # re-add batch dimension
    new_fields = TensorDict(x=integrated, batch_size=(1, T))
    return State(new_fields, time_axis=x.time_axis + dt)


class TestVariational(TestCase):
    def test_singlewindow(self):
        groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization = setup_task()

        assimilated_series = wc4dvar_single_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs,
            x_init=initialization,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_steps=50,
            verbose=False,
        )

        # sum MSE over space and time:
        MSE = ((assimilated_series - groundtruth).fields["x"] ** 2).mean()

        print(f"MSE of assimilated trajectory: {MSE}")
        self.assertTrue(MSE < 0.02, msg="The performance is worse than usual observed results.")

    def test_slidingwindow(
        self, shift_fraction=0.4, discard_partial=False, window_duration_steps=200, nb_steps=800, nb_steps_burnt=400
    ):
        groundtruth, obs_op, observations, model_error_distribs, m_dyn, initialization = setup_task(
            nb_steps=nb_steps, nb_steps_burnt=nb_steps_burnt
        )

        window_duration = window_duration_steps * (groundtruth.time_axis[1] - groundtruth.time_axis[0])

        x_eachwin, windows, idx_ranges = wc4dvar_sliding_window(
            m_dyn,
            observations,
            obs_op,
            model_error_distribs=model_error_distribs,
            x_init=initialization[:, :window_duration_steps],
            window_duration=window_duration,
            shift_fraction=shift_fraction,
            discard_partial=discard_partial,
            background_prior=None,
            optimizer_pars={"lr": 1},
            alpha=1e4,
            n_steps=50,
            verbose=False,
        )

        is_weak = True

        analysis = assemble_analysis(
            x_eachwin,
            idx_ranges,
            obs_op.time_axis,
            is_weak,
            m_dyn=m_dyn,
        )

        MSE = ((analysis - groundtruth).fields["x"] ** 2).mean()
        print(f"MSE of assimilated trajectory: {MSE}")
