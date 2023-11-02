class Vocab:
    key_words = [
        '</line>',
        '</none>', # []
        '</bold>', # ['<b>', ' ', '</b>']
        '</space>' # [' ']
    ]

    def __init__(self):
        self._words_ids_map = dict()
        self._ids_words_map = dict()

        for word_id, word in enumerate(self.key_words):
            self._words_ids_map[word] = word_id
            self._ids_words_map[word_id] = word
            
        self.line_id = self._words_ids_map['</line>']
        self.none_id = self._words_ids_map['</none>']
        self.bold_id = self._words_ids_map['</bold>']
        self.space_id = self._words_ids_map['</space>']
        self.blank_ids = [self.none_id, self.bold_id, self.space_id]
    
    def __len__(self):
        return len(self._words_ids_map)

    def word_to_id(self, word):
        return self._words_ids_map[word]

    def words_to_ids(self, words):
        return [self.word_to_id(word) for word in words]

    def id_to_word(self, word_id):
        return self._ids_words_map[word_id]
    
    def ids_to_words(self, words_id):
        return [self.id_to_word(word_id) for word_id in words_id]