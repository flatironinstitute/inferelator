import subprocess
import tempfile
import os
import time
from inferelator_ng import kvs_controller
from inferelator_ng.tests.test_mi import Test2By2, Test2By3

# TODO: Actually implement this well
temp_fd, temp_file_name = tempfile.mkstemp()
os.close(temp_fd)
KVS_CMD = ['python', '-m', 'kvsstcp.kvsstcp', "--addrfile", temp_file_name]
proc = subprocess.Popen(KVS_CMD)
time.sleep(2)

with open(temp_file_name, mode='r') as temp_fh:
    addr, port = temp_fh.read().strip().split(":")

os.environ["KVSSTCP_HOST"] = addr
os.environ["KVSSTCP_PORT"] = port
os.remove(temp_file_name)

class TestMIKVS2by2(Test2By2):
    kvs = kvs_controller.KVSController(suppress_warnings=True)


class TestMIKVS2by3(Test2By3):
    kvs = kvs_controller.KVSController(suppress_warnings=True)