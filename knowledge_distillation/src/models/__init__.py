"""Model implementations for knowledge distillation."""

from .base import BaseModel, TeacherModel, StudentModel
from .teachers import (
    ResNetTeacher, VGGTeacher, DenseNetTeacher, EfficientNetTeacher,
    TEACHER_MODELS, create_teacher_model
)
from .students import (
    CNN3Student, CNN5Student, CNN7Student, MobileNetV2Student, 
    MobileNetV3Student, ShuffleNetV2Student, SqueezeNetStudent,
    STUDENT_MODELS, create_student_model
)

__all__ = [
    # Base classes
    'BaseModel', 'TeacherModel', 'StudentModel',
    
    # Teacher models
    'ResNetTeacher', 'VGGTeacher', 'DenseNetTeacher', 'EfficientNetTeacher',
    'TEACHER_MODELS', 'create_teacher_model',
    
    # Student models
    'CNN3Student', 'CNN5Student', 'CNN7Student', 'MobileNetV2Student',
    'MobileNetV3Student', 'ShuffleNetV2Student', 'SqueezeNetStudent',
    'STUDENT_MODELS', 'create_student_model',
]


def get_available_models():
    """Get all available teacher and student models."""
    return {
        'teachers': list(TEACHER_MODELS.keys()),
        'students': list(STUDENT_MODELS.keys())
    }


def create_model(model_type: str, model_name: str, num_classes: int, **kwargs):
    """Create a model of the specified type and name."""
    if model_type.lower() == 'teacher':
        return create_teacher_model(model_name, num_classes, **kwargs)
    elif model_type.lower() == 'student':
        return create_student_model(model_name, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'teacher' or 'student'.")
