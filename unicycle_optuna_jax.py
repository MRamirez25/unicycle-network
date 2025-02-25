import optuna
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from unicycle_network_jax_class import UnicycleModel

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_int("n_steps", 50, 200)  # Number of unicycle network steps
    n_units = 100  # Keep fixed for now, could also be optimized
    n_inp = 1
    n_out = 10
    dt = 0.01

    # Instantiate the model with initial parameters
    model = UnicycleModel(
        n_units=n_units,
        n_inp=n_inp,
        n_out=n_out,
        dt=dt,
        lin_stiff_min=0.5,
        lin_stiff_max=1.0,
        ang_stiff_min=0.1,
        ang_stiff_max=0.3,
        lin_damping_min=0.1,
        lin_damping_max=0.2,
        ang_damping_min=0.1,
        ang_damping_max=0.2,
        eq_dist_min=0.5,
        eq_dist_max=1.0
    )

    # Initialize optimizer for readout layer only
    params = model.readout_layer.init(jax.random.PRNGKey(0))["params"]
    readout_params = params["readout_layer"]
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(readout_params)

    # Define the training loop
    @jax.jit
    def train_step(readout_params, opt_state, u_lin, u_ang, init_state, target):
        def loss_fn(params):
            model.readout_layer.params = model.readout_layer.set(params)
            return model.loss_fn(u_lin, u_ang, init_state, target, dt, n_steps)

        grads = jax.grad(loss_fn)(readout_params)
        updates, opt_state = optimizer.update(grads, opt_state, readout_params)
        readout_params = optax.apply_updates(readout_params, updates)
        return readout_params, opt_state

    # Define dummy input data and initial state
    u_lin = jnp.zeros((1, n_inp))  # Dummy input
    u_ang = jnp.zeros((1, n_inp))  # Dummy input
    init_state = (
        jnp.random.uniform(0.1, 0.6, (n_units,)),
        jnp.random.uniform(0.1, 0.2, (n_units,)),
        jnp.random.uniform(-jnp.pi, jnp.pi, (n_units,)),
        jnp.random.uniform(0.1, 1.0, (n_units,)),
        jnp.random.uniform(-1.0, 1.0, (n_units,))
    )
    target = jnp.zeros((n_out,))  # Dummy target

    # Training loop
    epochs = 100  # Number of epochs to train
    for epoch in range(epochs):
        readout_params, opt_state = train_step(readout_params, opt_state, u_lin, u_ang, init_state, target)

    # Compute final loss on a validation set or after training
    final_loss = model.loss_fn(u_lin, u_ang, init_state, target, dt, n_steps)
    return final_loss.item()

# Run the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Display the best parameters found
print("Best parameters:", study.best_params)
print("Best loss:", study.best_value)
