import hydra
from omegaconf import DictConfig

from RieNets.spdnets.training.spdnets_training import training

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='SPDMLR.yaml'

@hydra.main(config_path='./conf/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    training(cfg,args)

if __name__ == "__main__":
    main()