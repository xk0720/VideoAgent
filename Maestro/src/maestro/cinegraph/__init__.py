"""cinegraph(2026-08-07 用户令):ViMax 生成核的严格移植 × Maestro 纪律层。

一致性哲学换轨:不再追求镜间像素连续(无钉帧/无出场矢量/无桥),
改为【图像层焊死一致性】——机位树 + 首帧派生 + 参考图选择 + 多视图
肖像注册表;视频模型只负责把首帧动起来。

来源:VideoAgent/ViMax(agents/camera_image_generator.py、
reference_image_selector.py、character_portraits_generator.py、
pipelines/script2video_pipeline.py)——核心逻辑逐段移植,模型调用
换成本库客户端(LLM=crew、图像=flux/seedream、视频=可灵)。

保留自有:分镜及全部出门闸、台词/音效音频法、reviewer/verifier、
prompt enhancer(接点连续性职责在本模式天然失效)。
"""
from .loop import generate_movie_cinegraph  # noqa: F401
