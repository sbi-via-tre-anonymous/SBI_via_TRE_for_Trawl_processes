import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional


class DenseModel(nn.Module):
    linear_layer_sizes: Sequence[int]
    final_output_size: int
    dropout_rate: float

    def setup(self):
        # Dense layers
        self.dense_layers = [
            nn.Dense(features=size, name=f'dense_{i}')
            for i, size in enumerate(self.linear_layer_sizes)
        ]

        # Theta projector (only used when theta is provided)
        self.theta_projector = nn.Dense(
            features=self.linear_layer_sizes[0],
            name="theta_projector"
        )

        # Dropout layer
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        # Final output layer
        self.output_layer = nn.Dense(
            features=self.final_output_size,
            name="output"
        )

    def __call__(self, x, theta: Optional[jax.Array] = None, train: bool = False):

        # First dense layer
        x = self.dense_layers[0](x)
        x = nn.elu(x)

        # Handle theta if provided
        if theta is not None:
            theta_projected = self.theta_projector(theta)
            x = jnp.concatenate([x, theta_projected], axis=-1)

        # Pass through remaining dense layers
        for dense_layer in self.dense_layers[1:]:
            x = dense_layer(x)
            x = nn.elu(x)
            x = self.dropout(x, deterministic=not train)

        return self.output_layer(x)


# Example usage
if __name__ == "__main__":
    # Model hyperparameters
    linear_layer_sizes = [64, 32, 16]
    final_output_size = 5
    dropout_rate = 0.1

    # Create model instances
    model = DenseModel(
        linear_layer_sizes=linear_layer_sizes,
        final_output_size=final_output_size,
        dropout_rate=dropout_rate
    )

    # Generate dummy data
    key = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(key, (32, 100))  # batch_size, input_size
    dummy_theta = jax.random.normal(key, (32, 5))    # batch_size, theta_size

    # Initialize parameters
    params = model.init(key, dummy_input)
    params_with_theta = model.init(key, dummy_input, dummy_theta)

    # For training (with dropout)
    output_train = model.apply(
        params,
        dummy_input,
        train=True,
        rngs={'dropout': jax.random.PRNGKey(1)}
    )

    # For inference (without dropout)
   # output_inference = model.apply(
   #     params,
   #     dummy_input,
   #     train=False
   # )

    # With theta
    output_with_theta = model.apply(
        params_with_theta,
        dummy_input,
        theta=dummy_theta,
        train=False
    )
