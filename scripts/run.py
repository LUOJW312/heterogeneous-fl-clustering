import argparse
import client
import config
import logging
import os
import server


# Set up parser  设置解析器
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging   使追踪事件、诊断问题和调试应用变得容易
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation.    运行联合学习模拟"""

    # Read configuration file   读取配置文件
    fl_config = config.Config(args.config)

    # Initialize server  初始化服务器
    fl_server = {
        "basic": server.Server(fl_config),  #基本的联邦学习服务器
        "accavg": server.AccAvgServer(fl_config),  #执行准确性加权联合平均的联合学习服务器
        "directed": server.DirectedServer(fl_config),  #在选择期间使用配置文件进行指导的联合学习服务器。
        "kcenter": server.KCenterServer(fl_config),  #在选择期间执行 KCenter 分析的联合学习服务器。
        "kmeans": server.KMeansServer(fl_config),  #在选择期间执行 KMean 分析的联合学习服务器
        "magavg": server.MagAvgServer(fl_config),  #执行 magnetude 加权联合平均的联合学习服务器
        # "dqn": server.DQNServer(fl_config), # DQN server disabled
        # "dqntrain": server.DQNTrainServer(fl_config), # DQN server disabled
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
