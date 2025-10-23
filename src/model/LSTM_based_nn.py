import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Optional


class LSTMModel(nn.Module):
    lstm_hidden_size: int
    num_lstm_layers: int
    linear_layer_sizes: Sequence[int]
    mean_aggregation: bool
    final_output_size: int
    dropout_rate: float

    def setup(self):
        # LSTM layers
        self.lstm_layers = [
            nn.RNN(
                nn.LSTMCell(features=self.lstm_hidden_size),
                name=f'lstm_{i}'
            )
            for i in range(self.num_lstm_layers)
        ]

        # Theta projector (will only be used when theta is provided)
        self.base_theta_projector = nn.Dense(
            features=self.lstm_hidden_size,
            name="base_theta_projector"
        )

        # Linear layers
        self.linear_layers = [
            nn.Dense(features=size) for size in self.linear_layer_sizes
        ]

        # Dropout layer
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Final output layer
        self.output_layer = nn.Dense(features=self.final_output_size)

    def __call__(self, x, theta: Optional[jax.Array] = None, train: bool = False):
        # Expand 2D input to 3D if necessary
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=-1)

        # Process through LSTM layers
        for lstm in self.lstm_layers:
            x = lstm(x)

        # Aggregate outputs
        if self.mean_aggregation:
            x = jnp.mean(x, axis=1)
        else:
            x = x[:, -1, :]

        # Handle theta if provided
        if theta is not None:
            theta_projected = self.base_theta_projector(theta)
            x = jnp.concatenate([x, theta_projected], axis=-1)

        # Pass through linear layers with ELU activation
        for linear_layer in self.linear_layers[:-1]:
            x = linear_layer(x)
            x = nn.elu(x)
            x = self.dropout(x, deterministic=not train)

        # Final linear layer without dropout
        x = self.linear_layers[-1](x)
        x = nn.elu(x)

        return self.output_layer(x)


# Example usage
if __name__ == "__main__":
    # Model hyperparameters
    lstm_hidden_size = 64
    num_lstm_layers = 2
    linear_layer_sizes = (32, 16, 8)
    mean_aggregation = False
    final_output_size = 5
    dropout_rate = 0.1

    # Create two instances of the same model
    model_without_theta = LSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    model_with_theta = LSTMModel(
        lstm_hidden_size=lstm_hidden_size,
        num_lstm_layers=num_lstm_layers,
        linear_layer_sizes=linear_layer_sizes,
        mean_aggregation=mean_aggregation,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    # Generate dummy data
    key = jax.random.PRNGKey(0)
    # batch_size, seq_len, fake dimension
    dummy_input = jax.random.normal(key, (32, 10, 1))
    dummy_theta = jax.random.normal(key, (32, 5))  # batch_size, theta_size

    # Initialize parameters
    params_without_theta = model_without_theta.init(key, dummy_input)
    params_with_theta = model_with_theta.init(key, dummy_input, dummy_theta)

    # Define JIT-ed apply functions
    @jax.jit
    def apply_model_without_theta(params, x):
        return model_without_theta.apply(params, x)

    @jax.jit
    def apply_model_with_theta(params, x, theta):
        return model_with_theta.apply(params, x, theta)

    # Apply the models
    outputs_without_theta = apply_model_without_theta(
        params_without_theta, dummy_input)
    outputs_with_theta = apply_model_with_theta(
        params_with_theta, dummy_input, dummy_theta)
