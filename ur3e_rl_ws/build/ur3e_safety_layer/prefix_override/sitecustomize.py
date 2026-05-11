import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws/install/ur3e_safety_layer'
