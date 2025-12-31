##!/bin/bash
#
## 设置 Python 脚本路径和日志文件
#PYTHON_SCRIPT="test.py"
#LOG_FILE="script_output.log"
#
## 无限循环直到 Python 脚本成功执行
#while true; do
#    echo "Running Python script..."
#
#    # 运行 Python 脚本，并将输出保存到日志文件
#    python3 $PYTHON_SCRIPT --generate_type oov_u >> $LOG_FILE 2>&1
#    # 检查 Python 脚本是否成功执行
#    if [ $? -eq 0 ]; then
#        echo "Python script completed successfully."
#        break  # 如果成功，退出循环
#    else
#        echo "Python script failed. Restarting in 3 seconds..."
#        sleep 3  # 如果失败，等待3秒钟后重启
#    fi
#done
python test.py --generate_type iv_u
python test.py --generate_type oov_s
python test.py --generate_type oov_u