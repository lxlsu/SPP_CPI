"""
加入指纹信息。
"""

# from fingerModel import *
from sppFingerModel import *
from sklearn import metrics
import datetime
from sppDatapre import *
import torch.utils.data

# torch.cuda.set_device(1)
# torch.cuda.set_device(1)


def train(trainArgs):
    """
    args:
        model           : {object} model
        lr              : {float} learning rate
        train_loader    : {DataLoader} training data loaded into a dataloader
        doTest          : {bool} do test or not
        test_proteins   : {list} proteins list for test
        testDataDict    : {dict} test data dict
        seqContactDict  : {dict} seq-contact dict
        optimizer       : optimizer
        criterion       : loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
        epochs          : {int} number of epochs
        use_regularizer : {bool} use penalization or not
        penal_coeff     : {int} penalization coeff
        clip            : {bool} use gradient clipping or not
    Returns:
        accuracy and losses of the model
    """
    losses = []
    accs = []
    testResults = {}
    # 保存下来，用于画图
    loss_total = []
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    """
    modelArgs['d_a'] = 32
    d_a = modelArgs['d_a']
    modelArgs['protein_input_dim'] = 512
    modelArgs['protein_fc'] = 32        # 蛋白质全连接层的输出维度
    modelArgs['in_channels'] = 8  # 从原来的8改为16
    modelArgs['cnn_channels'] = 32
    cnn_channels = modelArgs['cnn_channels']
    modelArgs['cnn_layers'] = 4
    modelArgs['r'] = 10
    modelArgs['dense_hid'] = 64        # = modelArgs['d_a']+ modelArgs['protein_fc']
    """
    n_time_str = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    info = "spp_4_2_1" + "_p_" + str(p_input_dim) + "_" + str(doc2vec_epoch) + "_finger_" + str(finger_len) + "_in_" \
           + str(modelArgs['in_channels']) + "_cnn_" + str(cnn_b1) + "_" + str(cnn_b2) \
           + "_layer_" + str(modelArgs['block_num1']) + "_" + str(modelArgs['block_num2']) \
           + "_hidden_size_" + str(modelArgs['hidden_nodes']) \
           + "_spp_out_" + str(modelArgs['spp_out_dim']) + "_fc_" + str(modelArgs['fc_final']) \
           + "_lr_" + str(learning_rate)
    print("时间", n_time_str)
    print("参数", info)

    test_result = "../data/" + DataName + "/result_finger/" + n_time_str + "_" + info + "_Test_" + DataName + ".txt"
    dev_result = "../data/" + DataName + "/result_finger/" + n_time_str + "_" + info + "_Dev_" + DataName + ".txt"
    train_loss = "../data/" + DataName + "/log/" + n_time_str + "_" + info + "_Train_loss" + DataName + ".log"
    with open(test_result, "a+") as test_f:   # 写入测试集数据
        test_f.write("testAuc, testPrecision, testRecall, testAcc, testLoss\n") #
    with open(dev_result, "a+") as dev_f:   # 写入测试集数据
        dev_f.write("devAuc, devPrecision, devRecall, devAcc, devLoss\n")

    for i in range(trainArgs['epochs']):
        print("Running EPOCH", i + 1)
        total_loss = 0
        n_batches = 0
        correct = 0
        train_loader = trainArgs['train_loader']
        optimizer = trainArgs['optimizer']
        criterion = trainArgs["criterion"]
        attention_model = trainArgs['model']

        print("数据总数", len(train_loader))

        for batch_idx, (cmp_dm, finger, doc2vecProtein, y) in enumerate(train_loader):
            # input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            # attention_model.hidden_state = attention_model.init_hidden()
            cmp_dm = cmp_dm.type(torch.FloatTensor)   # 类型
            cmp_dm = create_variable(cmp_dm)

            finger = finger.type(torch.FloatTensor)  # 类型
            finger = create_variable(finger)
            # print(lines.shape, "lines.shape")

            doc2vecProtein = create_variable(doc2vecProtein)

            # print(contactmap.shape, "contactmap.shape")
            y_pred = attention_model(cmp_dm, finger, doc2vecProtein)  # 输出结果,输入化合物3d距离，蛋白质doc2vec
            # penalization AAT - I
            """
            if trainArgs['use_regularizer']:
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size,
                                                                 att.size(1), att.size(1))).cuda()

                penal = attention_model.l2_matrix_norm(att @ attT - identity)
            """
            # 二分类
                # binary classification
                # Adding a very small value to prevent BCELoss from outputting NaN's
            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                    y.type(torch.DoubleTensor)).data.sum()
            """
            if trainArgs['use_regularizer']:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor)) + (
                            trainArgs['penal_coeff'] * penal.cpu() / train_loader.batch_size)
            else:
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))
            """
            loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))

            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()  # retain_graph=True

            # gradient clipping
            if trainArgs['clip']:
                torch.nn.utils.clip_grad_norm_(attention_model.parameters(), 0.5)
            optimizer.step()
            n_batches += 1
            if batch_idx % 10 == 0:
                with open(train_loss, "a+") as t_loss:
                    t_loss.write("Epoch " + str(i + 1) + "\t" + str(batch_idx)+ "\t" + str(loss.item())+"\n")
                print(DataName, " [epoch %d]" % (i+1), batch_idx, loss.item())

        avg_loss = total_loss / n_batches
        acc = correct.numpy() / (len(train_loader.dataset))

        losses.append(avg_loss)
        accs.append(acc)

        print("avg_loss is", avg_loss)
        print("train ACC = ", acc)

        # if (trainArgs['doSave']):
        """
        model_name = n_time_str + "_" + info + "_" + DataName + '_%d.pkl' % (i + 1)
        print("存储模型", model_name)
        # torch.save(attention_model.state_dict(), './model_pkl//' + model_name)
        torch.save(attention_model, "../data/" + DataName + "/model/" + model_name)
        """
        if (trainArgs['doTest']):
            testArgs = {}
            testArgs['model'] = attention_model
            testArgs['criterion'] = trainArgs['criterion']
            testArgs['use_regularizer'] = trainArgs['use_regularizer']
            testArgs['penal_coeff'] = trainArgs['penal_coeff']
            testArgs['clip'] = trainArgs['clip']

            # attention_model.eval()
            # 测试集
            testResult = testPerProteinDataset72(testArgs)
            print(
                "test [len(dev_loader) %d] [Epoch %d/%d] [AUC : %.3f] "
                "[precision : %.3f] [recall : %.3f] [loss : %.3f]"
                % (len(dev_loader), i, trainArgs['epochs'], testResult[0], testResult[1], testResult[2], testResult[3])
            )
            with open(test_result, "a+") as test_f:
                test_f.write('\t'.join(map(str, testResult)) + '\n')

            # 验证集
            devResult = valPerProteinDataset72(testArgs)
            print(
                "validate [len(dev_loader) %d] [Epoch %d/%d] "
                "[AUC : %.3f] [precision : %.3f] [recall : %.3f] [loss : %.3f]"
                % (len(dev_loader), i+1, trainArgs['epochs'], devResult[0], devResult[1], devResult[2], devResult[3])
            )
            with open(dev_result, "a+") as dev_f:
                dev_f.write('\t'.join(map(str, devResult)) + '\n')

            # attention_model.train()

    return losses, accs, testResults


def testPerProteinDataset72(testArgs):
    testArgs['test_loader'] = test_loader
    testAcc, testRecall, testPrecision, testAuc, testLoss = test(testArgs)
    # result = [testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3, roce4]
    result = [testAuc, testPrecision, testRecall, testAcc, testLoss]

    return result


def valPerProteinDataset72(testArgs):
    testArgs['test_loader'] = dev_loader
    testAcc, testRecall, testPrecision, testAuc, testLoss = test(testArgs)
    # result = [testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3, roce4]
    result = [testAuc, testPrecision, testRecall, testAcc, testLoss]
    return result


def test(testArgs):
    test_loader = testArgs['test_loader']
    criterion = testArgs["criterion"]
    attention_model = testArgs['model']
    losses = []
    accuracy = []
    print('test begin ...')
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    with torch.no_grad():
        for batch_idx, (cmp_dm, finger, protein, y) in enumerate(test_loader):
            # input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            # attention_model.hidden_state = attention_model.init_hidden()
            cmp_dm = cmp_dm.type(torch.FloatTensor)  # 类型
            cmp_dm = create_variable(cmp_dm)

            finger = finger.type(torch.FloatTensor)  # 类型
            finger = create_variable(finger)

            protein = create_variable(protein)

            y_pred = attention_model(cmp_dm, finger, protein)
            if not bool(attention_model.type):
                # binary classification
                # Adding a very small value to prevent BCELoss from outputting NaN's
                pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
                correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                    y.type(torch.DoubleTensor)).data.sum()
                all_pred = np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis=0)
                all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis=0)
                """
                if trainArgs['use_regularizer']:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor)) + (
                                C * penal.cpu() / train_loader.batch_size)
                else:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))
                """
                loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y.type(torch.DoubleTensor))
            total_loss += loss.data
            n_batches += 1
    testSize = round(len(test_loader.dataset), 3)
    testAcc = round(correct.numpy() / (n_batches * test_loader.batch_size), 3)
    testRecall = round(metrics.recall_score(all_target, np.round(all_pred)), 3)
    testPrecision = round(metrics.precision_score(all_target, np.round(all_pred)), 3)
    testAuc = round(metrics.roc_auc_score(all_target, all_pred), 3)
    print("AUPR = ", metrics.average_precision_score(all_target, all_pred))
    testLoss = round(total_loss.item() / n_batches, 5)
    # print("test size =", testSize, "  test acc =", testAcc, "  test recall =", testRecall, "  test precision =",
    #       testPrecision, "  test auc =", testAuc, "  test loss = ", testLoss)

    return testAcc, testRecall, testPrecision, testAuc, testLoss

if __name__ == "__main__":
    # /home/student/Project/SPP_CPI_copy/DUDE-foldTest1_top4
    # testFoldPath = '../data/DUDE/dataPre/DUDE-foldTest1'
    # testFoldPath = '../data/DUDE-foldTest1/'
    print('get train datas....')

    print('get seq-contact dict....')
    # seqContactDict = getSeqContactDict(contactPath, contactDictPath)
    print('get letters....')

    print('train loader....')
    # trainDataSet:[[smile,seq,label],....]    seqContactDict:{seq:contactMap,....}
    # trainFoldPath = '../data/trainDataset72/trainDataset72Fold1/trainDataset72Fold1_filter'
    DataName = "human"
    dataset_name = "dataset_filter_2"
    trainFoldPath = "../data/" + DataName + "/" + dataset_name
    TotalDataset = getTrainDataSet(trainFoldPath)

    vector_len = 200
    k_gram = 3
    w_dows = 4
    doc2vec_epoch = 15
    # 3_576_4_10"
    p_name = str(k_gram) + "_" + str(vector_len) + "_" + str(w_dows) + "_" + str(doc2vec_epoch)

    protein_file_path = "../data/" + DataName + "/" + \
                        dataset_name + "_proteins_doc2vec_" + p_name
    trian_proteinDataset = load_tensor(protein_file_path, torch.FloatTensor)    # 蛋白质特征

    smiles_file_path = trainFoldPath + "_smiles"

    matrix_tensor = load_tensor(smiles_file_path, torch.FloatTensor)  # 加载距离矩阵， 化合物特征。

    b_size = 1
    finger_len = 50
    data_process_method = 'n'

    total_dataset = ProDataset(dataSet=TotalDataset, matrix=matrix_tensor, proteins=trian_proteinDataset,
                               bits=finger_len, method=data_process_method)
    # train_loader = DataLoader(dataset=total_dataset, batch_size=1, shuffle=True, drop_last=True)

    # 划分数据集
    train_size = int(0.8 * len(total_dataset))
    other_size = len(total_dataset) - train_size
    # 训练数据集
    train_dataset, other_dataset = torch.utils.data.random_split(total_dataset, [train_size, other_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    # 验证数据集， 测试数据集
    test_size = int(0.5 * other_size)
    dev_size = other_size - test_size
    dev_dataset, test_dataset = torch.utils.data.random_split(other_dataset, [dev_size, test_size])

    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=b_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size, shuffle=True)

    print('model args...')

    modelArgs = {}
    modelArgs['batch_size'] = 1
    modelArgs['protein_input_dim'] = vector_len
    modelArgs['protein_fc'] = vector_len           # dd 蛋白质全连接层的输出维度
    modelArgs['finger'] = finger_len
    modelArgs['d_a'] = 32
    # d_a = modelArgs['d_a']
    modelArgs['in_channels'] = 64  # 从原来的8改为16
    modelArgs['cnn_channel_block1'] = 128     # 通道数
    modelArgs['cnn_channel_block2'] = 128
    cnn_b1 = modelArgs['cnn_channel_block1']
    cnn_b2 = modelArgs['cnn_channel_block2']
    modelArgs['block_num1'] = 4       # resual block， CNN层数
    modelArgs['block_num2'] = 4

    modelArgs['r'] = 20
    modelArgs['cnn_layers'] = 4
    modelArgs['hidden_nodes'] = 256
    modelArgs['spp_out_dim'] = 100
    modelArgs['fc_final'] = 100       # = modelArgs['cnn_channels']+ modelArgs['protein_fc']

    p_input_dim = modelArgs['protein_input_dim']
    modelArgs['task_type'] = 0  # 0表示二分类，1表示多酚类
    modelArgs['n_classes'] = 1

    print('train args...')

    trainArgs = {}
    trainArgs['model'] = SPP_CPI(modelArgs, block=ResidualBlock).to(device)
    trainArgs['epochs'] = 35
    trainArgs['lr'] = 0.0001
    learning_rate = trainArgs['lr']
    trainArgs['train_loader'] = train_loader
    trainArgs['doTest'] = True
    trainArgs['use_regularizer'] = False
    trainArgs['penal_coeff'] = 0.03
    trainArgs['clip'] = True
    trainArgs['criterion'] = torch.nn.BCELoss()
    trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(), lr=trainArgs['lr'])
    trainArgs['doSave'] = True
    print('train args over...')

    losses, accs, testResults = train(trainArgs)
