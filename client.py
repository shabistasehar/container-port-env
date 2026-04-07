from openenv.core.env_client import EnvClient

from models import ContainerAction, ContainerObservation


class ContainerPortEnv(EnvClient[ContainerAction, ContainerObservation]):
    action_cls = ContainerAction
    observation_cls = ContainerObservation


__all__ = ["ContainerPortEnv"]
