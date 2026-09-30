"""元数据表模型：datasets / annotation_tasks / task_items / models / buckets / model_libraries / status_mapping。

设计约定：
- 每个 JSON 元数据实体一张表，键名与现有 JSON 文件字段保持一致（迁移零改键）
- 高频字段独立成列便于查询/排序；低频扩展键统一兜底进 extra JSONB，防字段遗漏丢数据
- M5 状态机统一（Phase 0）：三张主表均带 unified_status 列（旧列保留），status_mapping 定义旧值→新值映射
"""
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, BigInteger, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB


class Base(DeclarativeBase):
    pass


class DatasetRecord(Base):
    __tablename__ = "datasets"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, default="")
    archive: Mapped[str] = mapped_column(String, default="")
    size: Mapped[int] = mapped_column(BigInteger, default=0)
    uploaded_at: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="uploaded")
    stage: Mapped[str] = mapped_column(String, default="")
    training_status: Mapped[str] = mapped_column(String, default="incomplete")
    sealed_at: Mapped[str] = mapped_column(String, nullable=True)
    trained_at: Mapped[str] = mapped_column(String, nullable=True)
    last_trained_round: Mapped[int] = mapped_column(Integer, default=0)
    model_id: Mapped[str] = mapped_column(String, nullable=True)
    version: Mapped[str] = mapped_column(String, default="v1")
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    label_count: Mapped[int] = mapped_column(Integer, default=0)
    has_labels: Mapped[bool] = mapped_column(Boolean, default=False)
    split_ratio: Mapped[dict] = mapped_column(JSONB, default=dict)
    classes: Mapped[list] = mapped_column(JSONB, nullable=True)
    class_filter: Mapped[dict] = mapped_column(JSONB, nullable=True)
    label_validation: Mapped[dict] = mapped_column(JSONB, nullable=True)
    txt_to_json: Mapped[dict] = mapped_column(JSONB, nullable=True)
    model_auto_match: Mapped[dict] = mapped_column(JSONB, nullable=True)
    prepared_at: Mapped[str] = mapped_column(String, nullable=True)
    annotation_task_id: Mapped[str] = mapped_column(String, nullable=True)
    description: Mapped[str] = mapped_column(String, default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    bucket_pending: Mapped[int] = mapped_column(Integer, default=0)
    bucket_sealed: Mapped[int] = mapped_column(Integer, default=0)
    bucket_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    bucket_archived_at: Mapped[str] = mapped_column(String, nullable=True)
    # M5：统一状态机（新枚举值，旧列 status/stage/training_status 保留，双写期后统一读取）
    unified_status: Mapped[str] = mapped_column(String, nullable=True)
    # 兜底：未显式建模的扩展键（防迁移后丢字段）
    extra: Mapped[dict] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                 onupdate=datetime.utcnow)


class TaskRecord(Base):
    __tablename__ = "annotation_tasks"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String, default="")
    version: Mapped[str] = mapped_column(String, default="v1")
    classes: Mapped[list] = mapped_column(JSONB, nullable=True)
    model_id: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default="")
    imported_annotations: Mapped[int] = mapped_column(Integer, default=0)
    # M5：统一状态机（标注任务域统一枚举）
    unified_status: Mapped[str] = mapped_column(String, nullable=True)
    extra: Mapped[dict] = mapped_column(JSONB, default=dict)


class TaskItemRecord(Base):
    __tablename__ = "task_items"
    __table_args__ = ({"comment": "标注任务的图片列表"},)

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    image_path: Mapped[str] = mapped_column(String, default="")
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    annotated: Mapped[bool] = mapped_column(Boolean, default=False)
    # P2 标注完成判定：是否存在标注（approved 且有 label 内容）落表，配合 annotated 联合判定
    has_label: Mapped[bool] = mapped_column(Boolean, default=False)


class ModelRecord(Base):
    __tablename__ = "models"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    model_code: Mapped[str] = mapped_column(String, default="")
    name: Mapped[str] = mapped_column(String, default="")
    display_name: Mapped[str] = mapped_column(String, default="")
    business: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="active")
    source: Mapped[str] = mapped_column(String, default="")
    empty: Mapped[bool] = mapped_column(Boolean, default=False)
    classes: Mapped[list] = mapped_column(JSONB, nullable=True)
    labels_file: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(String, default="")
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    # 模型标签字典（labels_dict.json 内容：labels 列表）
    labels: Mapped[dict] = mapped_column(JSONB, nullable=True)
    # M5：统一状态机（模型域统一枚举）
    unified_status: Mapped[str] = mapped_column(String, nullable=True)
    # model_libraries：归属库（可空，未归库模型按旧 business 检索逻辑降级）
    model_library_id: Mapped[str] = mapped_column(String, nullable=True)
    extra: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[str] = mapped_column(String, default="")
    updated_at: Mapped[str] = mapped_column(String, default="")


class BucketRecord(Base):
    __tablename__ = "buckets"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    bucket_version_count: Mapped[int] = mapped_column(Integer, default=0)
    receiving: Mapped[dict] = mapped_column(JSONB, default=dict)
    sealed: Mapped[list] = mapped_column(JSONB, default=list)


class ModelLibraryRecord(Base):
    """算法库：模型的逻辑容器（base_algorithm / 标签字典 / 当前生产模型指针）。"""
    __tablename__ = "model_libraries"

    model_library_id: Mapped[str] = mapped_column(String, primary_key=True)
    display_name: Mapped[str] = mapped_column(String, default="")
    description: Mapped[str] = mapped_column(String, default="")
    business_scenario: Mapped[str] = mapped_column(String, default="")
    base_algorithm: Mapped[str] = mapped_column(String, default="")
    architecture_family: Mapped[str] = mapped_column(String, default="")
    label_dict_id: Mapped[str] = mapped_column(String, nullable=True)
    # 当前生产模型指针（P8 半自动发布更新；唯一生产模型）
    current_production_model_id: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = mapped_column(String, default="")
    updated_at: Mapped[str] = mapped_column(String, default="")


class StatusMappingRecord(Base):
    """M5 状态机映射表：旧枚举值 → 统一新枚举值（domain + 旧值唯一）。"""
    __tablename__ = "status_mapping"

    domain: Mapped[str] = mapped_column(String, primary_key=True)
    old_value: Mapped[str] = mapped_column(String, primary_key=True)
    new_value: Mapped[str] = mapped_column(String, default="")
    rule: Mapped[str] = mapped_column(String, default="")


class WeightMappingRecord(Base):
    """2.4 weight_mappings：跨代迁移的层名映射规则（源算法 → 目标算法）。

    规范：训练模块 §6.3。初始内置 ultralytics 同族映射（strict=False + 按层名/形状匹配），
    后续可扩展更多算法组合。mapping_rules 格式 {"source_layer_name": "target_layer_name"}。
    """
    __tablename__ = "weight_mappings"

    source_architecture: Mapped[str] = mapped_column(String, primary_key=True)
    target_architecture: Mapped[str] = mapped_column(String, primary_key=True)
    mapping_rules: Mapped[dict] = mapped_column(JSONB, default=dict)
    skip_layers: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[str] = mapped_column(String, default="")