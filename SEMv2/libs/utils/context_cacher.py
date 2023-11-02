class ContextCacher:
    def __init__(self):
        self.infos = dict()
    
    def reset(self):
        self.infos.clear()

    def cache_info(self, key, info):
        self.infos[key] = info

    def get_info(self, key):
        return self.infos[key]
    

global_context_cacher = ContextCacher()
