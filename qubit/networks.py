import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


class Lambda(nn.Module):
    def __init__(self, tau, end_point_protocol):
        """
        Args:
            tau: The current time of the cycle
            end_point_protocol: A 2D List of shape (num_steps + 1, 2) containing the (T, V) at each step
            The protocol at the last index should be the same as the protocol at the first index.
        """
        super().__init__()
        self.register_buffer("tau", torch.tensor(tau))
        num_steps = len(end_point_protocol) - 1
        self.register_buffer("num_steps", torch.tensor(num_steps))
        self.register_buffer("end_point_protocol",
                             torch.tensor(end_point_protocol))
        self.register_buffer("protocol_slopes",
                             torch.diff(self.end_point_protocol, axis=0)*num_steps)

        # Get the minimum of the protocol at each step
        clamp_protocols = torch.stack((torch.stack((self.end_point_protocol[:-1], self.end_point_protocol[1:])).min(0).values,
                                       torch.stack((self.end_point_protocol[:-1], self.end_point_protocol[1:])).max(0).values), dim=-2)
        # clamp_protocols is a tensor of shape(num_steps, 2, 2), where the second dimension corresponds to the min and max values of the protocol
        self.register_buffer("clamp_protocols", clamp_protocols)

        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 2)

    def get_linear_gradient(self, protocol_step):
        """Gets the gradient of a constant linear protocol between the endpoints. Used 
        for regularization. 
        Note: self.protocol_slopes is the gradient of this protocol in time units of protocol/(t/tau).
              This function returns the gradient of the protocol in original units of time (i.e. protocol/t)
        Args:
            protocol_step: The step of the protocol
        """
        if protocol_step > 3:
            raise ValueError("protocol_step too high")

        protocol_dot = (self.protocol_slopes[protocol_step] / (self.tau))
        return protocol_dot

    def forward(self, t):
        """Returns the current protocol at time t. This function currently only works if num_steps=4.
        Args:
            t: A tensor of shape (n_steps, 1) representing the current time of the protocol. t must be in [0, tau]
        Returns:
            protocol: A tensor of shape (n_steps, 2) representing the current (T, V) of the protocol
        """
        # Work in units of t/tau
        t = t/self.tau
        scaling = nn.Sigmoid()(self.fc1(t))
        scaling = self.fc2(scaling)
        # Multiply by this quintic to ensure smoother gradients
        scaling = scaling * ((t - 0) * (t - 0.25) *
                             (t - 0.50) * (t - 0.75) * (t - 1)) * 10

        protocol = ((t < 0.25) * torch.clamp((scaling + self.end_point_protocol[0] + t * self.protocol_slopes[0]),
                                             min=self.clamp_protocols[0][0],
                                             max=self.clamp_protocols[0][1])
                    + (0.25 <= t) * (t < 0.5) * torch.clamp((scaling + self.end_point_protocol[1] + (t - 0.25) * self.protocol_slopes[1]),
                                                            min=self.clamp_protocols[1][0],
                                                            max=self.clamp_protocols[1][1])
                    + (0.5 <= t) * (t < 0.75) * torch.clamp((scaling + self.end_point_protocol[2] + (t - 0.5) * self.protocol_slopes[2]),
                                                            min=self.clamp_protocols[2][0],
                                                            max=self.clamp_protocols[2][1])
                    + (0.75 <= t) * (t <= 1.00) * torch.clamp((scaling + self.end_point_protocol[3] + (t - 0.75) * self.protocol_slopes[3]),
                                                              min=self.clamp_protocols[3][0],
                                                              max=self.clamp_protocols[3][1]))
        return protocol



class Model:
    def __init__(self, tau, end_point_protocol, device=torch.device("cpu")):
        """
        Args:
            tau: The current time of the cycle
            end_point_protocol: A 2D List of shape (num_steps + 1, 2) containing the (T, V) at each step
            The protocol at the last index should be the same as the protocol at the first index.
        """
        self.device = device
        self.lambda_ = Lambda(tau, end_point_protocol).to(device)
        self.lambda_optimizer = Adam(self.lambda_.parameters(), lr=5e-6)

    def save_lambda_network(self, folder_name="./", iteration=0):
        """Saves lambda networks. Saves as latest iteration of protocol (i.e. lambda)
           and also checkpoints current protocol (i.e. lambda_iteration)
        """
        torch.save({"model_state_dict": self.lambda_.state_dict(),
                    "optimizer_state_dict": self.lambda_optimizer.state_dict()
                    }, folder_name + "lambda")

        torch.save({"model_state_dict": self.lambda_.state_dict(),
                    "optimizer_state_dict": self.lambda_optimizer.state_dict()
                    }, folder_name + "lambda_" + str(iteration))

    def load_lambda_network(self, folder_name="./", iteration=None):
        """Loads lambda networks. If iteration is None loads latest protocol.
           If iteration is not none loads a checkpointed protocol
        """
        filename = folder_name + "lambda"
        if iteration is not None:
            filename += "_" + str(iteration)
        lambda_checkpoint = torch.load(filename, map_location=self.device)
        self.lambda_.load_state_dict(lambda_checkpoint["model_state_dict"])
        self.lambda_optimizer.load_state_dict(
            lambda_checkpoint["optimizer_state_dict"])
