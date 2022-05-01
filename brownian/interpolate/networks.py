
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class LambdaI(nn.Module):
    def __init__(self, is_isothermal_step, init_protocol=[0.1], final_protocol=[1.]):
        super().__init__()
        self.is_isothermal_step = is_isothermal_step

        if self.is_isothermal_step:
            assert init_protocol[0] == final_protocol[0]
            self.register_buffer('T', torch.tensor(init_protocol[0:1]))
            self.register_buffer('init_protocol',
                                 torch.tensor(init_protocol[1:]))
            self.register_buffer('final_protocol',
                                 torch.tensor(final_protocol[1:]))
            final_layer_dim = 1
        else:
            self.register_buffer('init_protocol',
                                 torch.tensor(init_protocol))
            self.register_buffer('final_protocol',
                                 torch.tensor(final_protocol))
            final_layer_dim = 2

        # Probably a more elegant way to do this
        if self.is_isothermal_step:
            minimum_bound = torch.minimum(self.init_protocol,
                                          self.final_protocol)
            maximum_bound = torch.maximum(self.init_protocol,
                                          self.final_protocol)
        else:
            minimum_bound = torch.cat((torch.minimum(self.init_protocol[0:1], self.final_protocol[0:1]),
                                       torch.minimum(self.init_protocol[1:], self.final_protocol[1:])))
            maximum_bound = torch.cat((torch.maximum(self.init_protocol[0:1], self.final_protocol[0:1]),
                                       torch.maximum(self.init_protocol[1:], self.final_protocol[1:])))
        self.register_buffer("maximum_bound", maximum_bound)
        self.register_buffer("minimum_bound", minimum_bound)

        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, final_layer_dim)

    def forward(self, t_frac):
        """
        Returns:
            protocol: A torch tensor containing [T, k], representing the current protocol
        """
        x = F.relu(self.fc1(t_frac))
        x = self.fc2(x)
        x = (self.final_protocol * (t_frac)
             + self.init_protocol * (1 - t_frac)
             + x * (t_frac - 0) * (t_frac - 1))
        x = torch.clamp(x.abs(), min=self.minimum_bound, max=self.maximum_bound)
        if (self.is_isothermal_step):
            protocol = torch.cat([self.T.repeat(x.shape), x], dim=-1)
        else:
            protocol = x
        return protocol


class Lambda(nn.Module):
    def __init__(self, protocol, is_isothermal):
        """
        Args:
            protocol: A list of shape (num_steps, 2) 
                      containing the temperature and harmonic potential (k) of each step as floats
            is_isothermal: A list of shape (num_steps) containing a bool of whether each step is an isothermal one
        """
        super().__init__()
        num_steps = len(protocol)
        lambdas = []
        for i in range(num_steps):
            lambdas.append(LambdaI(is_isothermal_step=is_isothermal[i],
                                   init_protocol=protocol[i],
                                   final_protocol=protocol[(i + 1) % num_steps]))
        self.lambdas = nn.ModuleList(lambdas)

    def forward(self, i, t_frac):
        """
        Returns:
            protocol: A torch tensor containing [T, k], representing the current protocol
        """
        protocol = self.lambdas[i](t_frac)
        return protocol


class Model:
    def __init__(self, protocol, is_isothermal, device=torch.device("cpu")):
        self.device = device
        self.lambda_ = Lambda(protocol, is_isothermal).to(device)
        self.lambda_optimizer = Adam(self.lambda_.parameters(), lr=1e-3)

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
