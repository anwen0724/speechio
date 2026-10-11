# 功能规格

本目录维护具体功能的行为与验收标准。编写和使用方式见[Spec 说明](../docs-templates/spec.md)。

产品范围由 [PRD](../PRD.md) 维护，结构约束由 [Architecture](../ARCHITECTURE.md) 维护。项目阶段见 [ROADMAP](../ROADMAP.md)，实际结果与未完成事项见 [STATUS](../STATUS.md)。

## 规格入口

| 功能规格 | 对应阶段 | 内容 |
| --- | --- | --- |
| [运行环境与识别基线验证](runtime-and-corpus-baseline.md) | P0 | 环境、模型准备、语料、整段／快照／原生流式验证及测量契约。 |

## 使用与维护

开发实施 Plan 应引用对应 Spec，不另建一套验收规则。行为改变时先维护负责该功能的 Spec，再同步受影响的实施安排。

规格存在不表示功能已实现。运行、测量和验收结论统一记录在 STATUS，不在本目录积累执行进度。
