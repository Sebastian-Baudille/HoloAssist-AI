import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nic/git/HoloAssist-AI/ur3e_rl_ws/install/ur3e_rl_env'
