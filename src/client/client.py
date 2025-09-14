import logging
import torch
import torch.nn as nn
import torch.optim as optim


class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):    #客户端的唯一标识符
        self.client_id = client_id

    def __repr__(self):   #客户端的字符串表示：客户端ID，数据样本数，标签集合
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), set([label for _, label in self.data]))

    # Set non-IID data configurations  #独立同分布数据配置
    def set_bias(self, pref, bias):  #接口
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):   #客户端数据分片
        self.shard = shard

    # Server interactions   #服务器交互
    def download(self, argv):    #服务器下载数据或配置
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):   #将数据上传到服务器
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv

    # Federated learning phases  #联邦学习阶段
    def set_data(self, data, config):
        # Extract from config
        do_test = self.do_test = config.clients.do_test   #config中下载数据，设为训练集
        test_partition = self.test_partition = config.clients.test_partition  #测试集

        # Download data
        self.data = self.download(data)

        # Extract trainset, testset (if applicable)
        data = self.data
        if do_test:  # Partition for testset if applicable
            self.trainset = data[:int(len(data) * (1 - test_partition))]
            self.testset = data[int(len(data) * (1 - test_partition)):]
        else:
            self.trainset = data

    def configure(self, config):   #配置客户端
        import fl_model  # pylint: disable=import-error

        # Extract from config
        model_path = self.model_path = config.paths.model

        # Download from server
        config = self.download(config)

        # Extract machine learning task from config   #下载全局模型
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

        # Download most recent global model
        path = model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Create optimizer  #配置优化器
        self.optimizer = fl_model.get_optimizer(self.model)

    def run(self):
        # Perform federated learning task   运行联邦任务
        {
            "train": self.train()
        }[self.task]

    def get_report(self):
        # Report results to server.  结果上传到服务器
        return self.upload(self.report)

    # Machine learning tasks
    def train(self):
        import fl_model  # pylint: disable=import-error

        logging.info('Training on client #{}'.format(self.client_id))

        # Perform model training
        trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
        fl_model.train(self.model, trainloader,
                       self.optimizer, self.epochs)

        # Extract model weights and biases
        weights = fl_model.extract_weights(self.model)

        # Generate report for server
        self.report = Report(self)
        self.report.weights = weights

        # Perform model testing if applicable
        if self.do_test:
            testloader = fl_model.get_testloader(self.testset, 1000)
            self.report.accuracy = fl_model.test(self.model, testloader)

    def test(self):
        # Perform model testing
        raise NotImplementedError


class Report(object):
    """Federated learning client report."""

    def __init__(self, client):
        self.client_id = client.client_id
        self.num_samples = len(client.data)
