from pydantic import BaseModel, Field

from src.models.enums import TaskStatus


class TaskSubmitResponse(BaseModel):
    status: TaskStatus = Field(default=TaskStatus.waiting)
    task_uuid: str
    message: str


class TaskStatusResponse(BaseModel):
    status: TaskStatus = Field(default=TaskStatus.waiting)
    task_uuid: str
    result_text: str
