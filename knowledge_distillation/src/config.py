"""Configuration management for the knowledge distillation framework."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import yaml


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str = "kd_experiment"
    seed: int = 42
    output_dir: str = "artifacts"
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    download: bool = True
    augmentation: bool = True
    normalize: bool = True
    validation_split: float = 0.0
    
    # Dataset-specific parameters
    image_size: Optional[int] = None
    num_classes: Optional[int] = None


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str = ""
    pretrained: bool = False
    num_classes: Optional[int] = None
    dropout: float = 0.0
    
    # Training parameters
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    optimizer: str = "sgd"
    scheduler: str = "multistep"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeacherConfig(ModelConfig):
    """Teacher model configuration."""
    freeze_features: bool = False
    warmup_epochs: int = 0


@dataclass
class StudentConfig(ModelConfig):
    """Student model configuration."""
    supervised_baseline: bool = False
    supervised_epochs: int = 5


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""
    alpha: float = 0.5  # Weight for hard labels
    temperature: float = 4.0
    loss_type: str = "kl_div"  # kl_div, mse, cosine
    
    # Advanced distillation options
    feature_distillation: bool = False
    attention_distillation: bool = False
    progressive_distillation: bool = False
    
    # Loss weights for multi-loss distillation
    loss_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    log_interval: int = 100
    save_interval: int = 5
    eval_interval: int = 1
    
    # Logging backends
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "knowledge-distillation"
    wandb_entity: Optional[str] = None
    
    # What to log
    log_gradients: bool = False
    log_weights: bool = False
    log_activations: bool = False


@dataclass
class KnowledgeDistillationConfig:
    """Complete configuration for knowledge distillation."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str) -> KnowledgeDistillationConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Use OmegaConf for better configuration handling
    conf = OmegaConf.create(config_dict)
    
    # Convert to structured config using the dataclass
    structured_conf = OmegaConf.structured(KnowledgeDistillationConfig)
    merged_conf = OmegaConf.merge(structured_conf, conf)
    
    return OmegaConf.to_object(merged_conf)


def save_config(config: KnowledgeDistillationConfig, save_path: str):
    """Save configuration to YAML file."""
    config_dict = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(save_path, 'w') as f:
        f.write(config_dict)


def merge_configs(base_config: KnowledgeDistillationConfig, 
                 override_config: Dict[str, Any]) -> KnowledgeDistillationConfig:
    """Merge base configuration with overrides."""
    base_conf = OmegaConf.structured(base_config)
    override_conf = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_conf, override_conf)
    return OmegaConf.to_object(merged)
