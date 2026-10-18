"""运行角色的实例拥有权，集中隐藏身份共享、运行库兼容与释放。"""

from threading import Lock

from speechio.runtime.transcriber import Transcriber


class EngineResources:
    """不同角色只取得模型和锁，不重复维护实例容器及生命周期规则。"""

    def __init__(self, primary, primary_lock):
        self._owners = [(primary, primary_lock)]
        self._assigned = set()

    def assign(self, *, model=None, engine=None):
        """缺省共用主实例；注入按身份共享，显式本地配置创建独立实例。"""
        if model is None:
            selected = engine if engine is not None else self._owners[0][0].engine
            for owner, lock in self._owners:
                if owner.engine is selected:
                    self._assigned.add(owner)
                    return owner, lock
        owner = Transcriber(model, engine=engine)
        owned = owner, Lock()
        self._owners.append(owned)
        self._assigned.add(owner)
        return owned

    @property
    def injected(self):
        return any(owner.config is None for owner, _ in self._owners)

    def prepare(self, *, defer_primary=False):
        """流式可独立完成时，未被其他角色使用的主模型延迟到整段调用时准备。"""
        primary = self._owners[0][0]
        for owner, _ in self._owners:
            if defer_primary and owner is primary and owner not in self._assigned:
                continue
            owner.prepare()

    def close(self):
        """仅在所有执行线程退出后调用；每个拥有者保存自己的关闭结果。"""
        return tuple(error for owner, _ in self._owners for error in owner.close().errors)
