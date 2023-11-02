import os
import pickle
import threading


def merge_record_file(files, dst_file):
    cmd = 'cat'
    for file in files:
        cmd += ' %s' % file
    cmd += ' > %s' % dst_file
    os.system(cmd)


class ListRecordCacher:
    OFFSET_LENGTH = 8
    def __init__(self, cache_path):
        self._record_pos_list = list()
        self._cache_file = open(cache_path, 'wb')
        self._cached_bytes = b'\x00' * self.OFFSET_LENGTH

    def add_record(self, record):
        record_bytes = pickle.dumps(record)
        return self.add_record_bytes(record_bytes)
    
    def add_record_bytes(self, record_bytes):
        bytes_size = len(record_bytes)
        offset_bytes = bytes_size.to_bytes(
            length=self.OFFSET_LENGTH,
            byteorder='big', signed=False
        )
        total_bytes = offset_bytes + record_bytes

        cur_record_pos = None
        if len(self._record_pos_list) == 0:
            cur_record_pos = [self.OFFSET_LENGTH*2, bytes_size]
        else:
            cur_record_pos = [sum(self._record_pos_list[-1]) + self.OFFSET_LENGTH, bytes_size]
        self._record_pos_list.append(cur_record_pos)

        self._cached_bytes += total_bytes
        if len(self._cached_bytes) > 1024*1024:
            self._cache_file.seek(0, 2)
            self._cache_file.write(self._cached_bytes)
            self._cached_bytes = b''
        
    def flush(self):
        if len(self._cached_bytes) > 0:
            self._cache_file.seek(0, 2)
            self._cache_file.write(self._cached_bytes)
            self._cached_bytes = b''

    def _wirte_record_pos_list(self):
        self.flush()
        self._cache_file.seek(0, 2)
        offset = self._cache_file.tell()
        offset_bytes = offset.to_bytes(
            length=self.OFFSET_LENGTH,
            byteorder='big', signed=False
        )
        self._cache_file.seek(0)
        self._cache_file.write(offset_bytes)
        
        data_bytes = pickle.dumps(self._record_pos_list)
        bytes_size = len(data_bytes)
        offset_bytes = bytes_size.to_bytes(
            length=self.OFFSET_LENGTH,
            byteorder='big', signed=False
        )
        total_bytes = offset_bytes + data_bytes
        self._cache_file.seek(0, 2)
        self._cache_file.write(total_bytes)

    def close(self):
        if not self._cache_file.closed:
            self._wirte_record_pos_list()
            self._cache_file.close()

    def __del__(self):
        self.close()


class ListRecordLoader:
    OFFSET_LENGTH = 8
    def __init__(self, load_path):
        self._sync_lock = threading.Lock()
        self._size = os.path.getsize(load_path)
        self._load_path = load_path
        self._open_file()
        self._scan_file()

    def _open_file(self):
        self._pid = os.getpid()
        self._cache_file = open(self._load_path, 'rb')

    def _check_reopen(self):
        if (self._pid != os.getpid()):
            self._open_file()

    def _scan_file(self):
        record_pos_list = list()
        pos = 0
        while True:
            if pos >= self._size:
                break
            self._cache_file.seek(pos)
            offset = int().from_bytes(
                self._cache_file.read(self.OFFSET_LENGTH),
                byteorder='big', signed=False
            )
            offset = pos + offset
            self._cache_file.seek(offset)

            byte_size = int().from_bytes(
                self._cache_file.read(self.OFFSET_LENGTH),
                byteorder='big', signed=False
            )
            record_pos_list_bytes = self._cache_file.read(byte_size)
            sub_record_pos_list = pickle.loads(record_pos_list_bytes)
            assert isinstance(sub_record_pos_list, list)
            sub_record_pos_list = [[item[0]+pos, item[1]] for item in sub_record_pos_list]
            record_pos_list.extend(sub_record_pos_list)
            pos = self._cache_file.tell()
        
        self._record_pos_list = record_pos_list

    def get_record(self, idx):
        self._check_reopen()
        record_bytes = self.get_record_bytes(idx)
        record = pickle.loads(record_bytes)
        return record

    def get_record_bytes(self, idx):
        offset, length = self._record_pos_list[idx]
        self._sync_lock.acquire()
        try:
            self._cache_file.seek(offset)
            record_bytes = self._cache_file.read(length)
        finally:
            self._sync_lock.release()
        return record_bytes

    def __len__(self):
        return len(self._record_pos_list)
