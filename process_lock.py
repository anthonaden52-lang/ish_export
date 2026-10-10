"""Small OS-released process lock for single-instance collectors and jobs."""
import os
from pathlib import Path


class ProcessLock:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(self.fd).st_size == 0:
                    os.write(self.fd, b"\0")
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, OSError) as exc:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(f"Another process holds {self.path}.") from exc
        return self

    def release(self):
        if self.fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
