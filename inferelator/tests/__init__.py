from inferelator.distributed.inferelator_mp import MPControl

MPControl.set_multiprocess_engine("local")
MPControl.connect()