"""运行角色的实例拥有权，集中隐藏身份共享、运行库兼容与释放。"""

from threading import Lock

from speechio.runtime.transcriber import Transcriber


class EngineResources:
    """不同角色只取得模型和锁，不重复维护实例容器及生命周期规则。"""

    def __init__(self, primary, primary_lock):
        self._owners = [(primary, primary_lock)]

    def assign(self, *, model=None, engine=None):
        """缺省共用主实例；注入按身份共享，显式本地配置创建独立实例。"""
        if model is None:
            selected = engine if engine is not None else self._owners[0][0].engine
            for owner, lock in self._owners:
                if owner.engine is selected:
                    return owner, lock
        owner = Transcriber(model, engine=engine)
        owned = owner, Lock()
        self._owners.append(owned)
        return owned

    @property
    def injected(self):
        return any(owner.config is None for owner, _ in self._owners)

    def prepare(self):
        for owner, _ in self._owners:
            owner.prepare()

    def close(self):
        """仅在所有执行线程退出后调用；每个拥有者保存自己的关闭结果。"""
        return tuple(error for owner, _ in self._owners for error in owner.close().errors)
