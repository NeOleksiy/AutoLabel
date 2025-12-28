from dataclasses import dataclass

@dataclass
class TaskConfig:
    """Configuration for a specific task"""
    name: str
    prompt_template: str
    description: str
    output_format: str
    requires_categories: bool = True
    requires_visual_prompt: bool = False
    requires_keypoint_type: bool = False