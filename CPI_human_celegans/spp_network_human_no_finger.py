"""
no finger information
"""

from sppModel import *
from sklearn import metrics
import datetime
from sppDatapre import *
import torch.utils.data

# torch.cuda.set_device(1)
# torch.cuda.set_device(1)


def train(trainArgs):

    losses = []
    accs = []
    testResults = {}
    # 保存下来，用于画图
    loss_total = []
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'

    n_time_str = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    info = "spp_4_2_1_fc_protein_2_elu" + "_protein_" + str(p_input_dim) + "_" + str(doc2vec_epoch) + "_" \
           + str(modelArgs['protein_fc']) + "_in_" \
           + str(modelArgs['in_channels']) + "_cnn_" + str(cnn_b1) + "_" + str(cnn_b2) \
           + "_layer_" + str(modelArgs['block_num1']) + "_" + str(modelArgs['block_num2']) \
           + "_spp_fc_" + str(modelArgs['hidden_nodes']) + "_" + str(modelArgs['spp_out_dim']) + "_combine_fc_" + str(modelArgs['fc_final']) \
           + "_lr_" + str(learning_rate)
    print("时间", n_time_str)
    print("参数", info)
    if data_process_method == 'n':
        result_file_name = "result_conv5_3_normal"
    elif data_process_method == 's':
        result_file_name = "result_conv5_3_standard"
    else:
        result_file_name = "result_conv5_3"
    test_result = "../data/" + DataName + "/"+result_file_name + "/" + n_time_str + "_" + info + "_Test_" + DataName + ".txt"
    dev_result = "../data/" + DataName + "/"+result_file_name + "/" + n_time_str + "_" + info + "_Dev_" + DataName + ".txt"
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

        for batch_idx, (cmp_dm, finger, doc2vecProtein, y) in enumerate(train_loader):

            cmp_dm = cmp_dm.type(torch.FloatTensor)
            cmp_dm = create_variable(cmp_dm)

            finger = finger.type(torch.FloatTensor)
            finger = create_variable(finger)
            # print(lines.shape, "lines.shape")

            doc2vecProtein = create_variable(doc2vecProtein)


            y_pred = attention_model(cmp_dm, finger, doc2vecProtein)

            correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                    y.type(torch.DoubleTensor)).data.sum()

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

        if (trainArgs['doTest']):
            testArgs = {}
            testArgs['model'] = attention_model
            testArgs['criterion'] = trainArgs['criterion']
            testArgs['use_regularizer'] = trainArgs['use_regularizer']
            testArgs['penal_coeff'] = trainArgs['penal_coeff']
            testArgs['clip'] = trainArgs['clip']

            attention_model.eval()

            testResult = testPerProteinDataset72(testArgs)
            print(
                "test [len(dev_loader) %d] [Epoch %d/%d] [AUC : %.3f] "
                "[precision : %.3f] [recall : %.3f] [loss : %.3f]"
                % (len(dev_loader), i, trainArgs['epochs'], testResult[0], testResult[1], testResult[2], testResult[3])
            )
            with open(test_result, "a+") as test_f:
                test_f.write('\t'.join(map(str, testResult)) + '\n')

            devResult = valPerProteinDataset72(testArgs)
            print(
                "validate [len(dev_loader) %d] [Epoch %d/%d] "
                "[AUC : %.3f] [precision : %.3f] [recall : %.3f] [loss : %.3f]"
                % (len(dev_loader), i+1, trainArgs['epochs'], devResult[0], devResult[1], devResult[2], devResult[3])
            )
            with open(dev_result, "a+") as dev_f:
                dev_f.write('\t'.join(map(str, devResult)) + '\n')

        attention_model.train()

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

            cmp_dm = cmp_dm.type(torch.FloatTensor)
            cmp_dm = create_variable(cmp_dm)

            finger = finger.type(torch.FloatTensor)
            finger = create_variable(finger)

            protein = create_variable(protein)

            y_pred = attention_model(cmp_dm, finger, protein)
            if not bool(attention_model.type):

                pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
                correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),
                                    y.type(torch.DoubleTensor)).data.sum()
                all_pred = np.concatenate((all_pred, y_pred.data.cpu().squeeze(1).numpy()), axis=0)
                all_target = np.concatenate((all_target, y.data.cpu().numpy()), axis=0)

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

    return testAcc, testRecall, testPrecision, testAuc, testLoss

if __name__ == "__main__":

    print('get train datas....')

    print('get seq-contact dict....')
    # seqContactDict = getSeqContactDict(contactPath, contactDictPath)
    print('get letters....')

    print('train loader....')

    DataName = "celegans"
    dataset_name = "dataset_filter_2"
    trainFoldPath = "../data/" + DataName + "/" + dataset_name
    TotalDataset = getTrainDataSet(trainFoldPath)

    vector_len = 1024
    k_gram = 3
    w_dows = 4
    doc2vec_epoch = 7
    # 3_576_4_10"
    p_name = str(k_gram) + "_" + str(vector_len) + "_" + str(w_dows) + "_" + str(doc2vec_epoch)

    protein_file_path = "../data/" + DataName + "/" + \
                        dataset_name + "_proteins_doc2vec_" + p_name
    trian_proteinDataset = load_tensor(protein_file_path, torch.FloatTensor)

    smiles_file_path = trainFoldPath + "_smiles"

    matrix_tensor = load_tensor(smiles_file_path, torch.FloatTensor)

    b_size = 1
    finger_len = 34
    data_process_method = 'n'

    total_dataset = ProDataset(dataSet=TotalDataset, matrix=matrix_tensor, proteins=trian_proteinDataset,
                               bits=finger_len, method=data_process_method)
    # train_loader = DataLoader(dataset=total_dataset, batch_size=1, shuffle=True, drop_last=True)

    train_size = int(0.8 * len(total_dataset))
    other_size = len(total_dataset) - train_size

    train_dataset, other_dataset = torch.utils.data.random_split(total_dataset, [train_size, other_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True)

    test_size = int(0.5*other_size)
    dev_size = other_size - test_size
    dev_dataset, test_dataset = torch.utils.data.random_split(other_dataset, [dev_size, test_size])

    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=b_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size, shuffle=True)

    print('model args...')

    modelArgs = {}
    modelArgs['batch_size'] = 1
    modelArgs['protein_input_dim'] = vector_len
    modelArgs['protein_fc'] = 36
    modelArgs['d_a'] = 32
    # d_a = modelArgs['d_a']
    modelArgs['in_channels'] = 32
    modelArgs['cnn_channel_block1'] = 128
    modelArgs['cnn_channel_block2'] = 128
    cnn_b1 = modelArgs['cnn_channel_block1']
    cnn_b2 = modelArgs['cnn_channel_block2']
    modelArgs['block_num1'] = 16
    modelArgs['block_num2'] = 16

    modelArgs['r'] = 20
    # modelArgs['cnn_layers'] = 4
    modelArgs['hidden_nodes'] = 256
    modelArgs['spp_out_dim'] = 64
    modelArgs['fc_final'] = 100

    p_input_dim = modelArgs['protein_input_dim']
    modelArgs['task_type'] = 0
    modelArgs['n_classes'] = 1

    print('train args...')

    trainArgs = {}
    trainArgs['model'] = SPP_CPI(modelArgs, block=ResidualBlock).to(device)
    trainArgs['epochs'] = 30
    trainArgs['lr'] = 0.0001
    learning_rate = trainArgs['lr']
    trainArgs['weight_decay'] = 1e-4
    trainArgs['train_loader'] = train_loader
    trainArgs['doTest'] = True
    trainArgs['use_regularizer'] = False
    trainArgs['penal_coeff'] = 0.03
    trainArgs['clip'] = True
    trainArgs['criterion'] = torch.nn.BCELoss()

    trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(),
                                              lr=trainArgs['lr'])
    trainArgs['doSave'] = True
    print('train args over...')

    losses, accs, testResults = train(trainArgs)

"""

"""