"""
SPP_CPI
"""
from spp_datapre import *
from sklearn import metrics
import datetime
from spp_finger_model import *
import os


def train(trainArgs):

    losses = []
    accs = []
    testResults = {}
    loss_total = []
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    n_time_str = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    info = "spp_fc_protein_2_" + str(fold_num) + "_protein_" + str(p_input_dim) + "_" + str(doc2vec_epoch) + "_" \
           + str(modelArgs['protein_fc']) + "_in_channel_" \
           + str(modelArgs['in_channels']) + "_cnn_channel_" + str(cnn_b1) + "_" + str(cnn_b2) \
           + "_cnn_layer_" + str(modelArgs['block_num1']) + "_" + str(modelArgs['block_num2']) \
           + "_spp_out_dim_" + str(modelArgs['spp_out_dim']) + "_fc_" + str(modelArgs['fc_final']) \
           + "_lr_" + str(learning_rate)
    train_log = "./log/" + n_time_str + info + "_Test_" + foldName + ".txt"
    train_loss = "./log/" + n_time_str + info + "_Train_loss" + foldName + ".log"
    with open(train_log, "a+") as t_f:
        t_f.write("testAcc, testRecall, testPrecision, testAuc, "
                  "testLoss, all_pred, all_target, roce1, roce2, roce3, roce4\n")

    for i in range(0, trainArgs['epochs']):
        print("Running EPOCH", i + 1)
        total_loss = 0
        n_batches = 0
        correct = 0
        train_loader = trainArgs['train_loader']
        optimizer = trainArgs['optimizer']
        criterion = trainArgs["criterion"]
        attention_model = trainArgs['model']

        print("数据总数", len(train_loader))

        for batch_idx, (lines, finger, contactmap, y) in enumerate(train_loader):
            # input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            # attention_model.hidden_state = attention_model.init_hidden()
            lines = lines.type(torch.FloatTensor)
            lines = create_variable(lines)
            finger = finger.type(torch.FloatTensor)
            finger = create_variable(finger)
            # print(lines.shape, "lines.shape")

            contactmap = create_variable(contactmap)

            y_pred = attention_model(lines, finger, contactmap)

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
                print("[epoch %d]" % (i+1), batch_idx, loss.item())

        avg_loss = total_loss / n_batches
        acc = correct.numpy() / (len(train_loader.dataset))

        losses.append(avg_loss)
        accs.append(acc)

        print("avg_loss is", avg_loss)
        print("train ACC = ", acc)

        # if (trainArgs['doSave']):
        model_name = n_time_str + info + "_" + foldName + '_%d.pkl' % (i + 1)
        torch.save(attention_model.state_dict(), './model_pkl_30/' + model_name)
        # torch.save(attention_model, "./model/trainDataset72/" + model_name)

        if (trainArgs['doTest']):
            testArgs = {}
            testArgs['model'] = attention_model
            testArgs['test_proteins'] = trainArgs['test_proteins']
            testArgs['testDataDict'] = trainArgs['testDataDict']
            testArgs['seqContactDict'] = trainArgs['seqContactDict']
            testArgs['criterion'] = trainArgs['criterion']
            testArgs['use_regularizer'] = trainArgs['use_regularizer']
            testArgs['penal_coeff'] = trainArgs['penal_coeff']
            testArgs['clip'] = trainArgs['clip']
            testArgs['d_name'] = trainArgs['d_name']
            testArgs['testDataset'] = trainArgs['testDataset']

            # d_name = "test_dataset_30_fold1"        #
            # testDataset = "New File"    # d_name + "_filter"
            testResult = testPerProtein(testArgs)
            # testResult = testPerProtein(testArgs)
            #         result[x] = [testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target, roce1, roce2, roce3,
            #                      roce4]
            # testResults[i] = testResult
            with open(train_log, "a+") as t_f:
                t_f.write('\t'.join(map(str, testResult)) + '\n')
    return losses, accs, testResults


def getROCE(predList, targetList, roceRate):
    """
    getROCE(all_pred, all_target, 0.5)
    :param predList:
    :param targetList:
    :param roceRate:
    :return:
    """
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x: x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if (targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if (fp1 > ((roceRate * n) / 100)):
                break
    roce = (tp1 * n) / (p * fp1)
    return roce


def testPerProteinDataset72(testArgs, path_name, fileName):
    # file = "../../data/DUDE-foldTest1/"
    # testDataPath = "./DUDE-foldTest1_top4/"
    testDataPath = "../../data/testDataset30/" + path_name + "/"
    filePath = testDataPath + fileName
    # test_protein_file = testDataPath + fileName + "_protein_seq_proteins_doc2vec_3_512_4_7"
    test_protein_file = testDataPath + fileName + "_protein_seq_proteins_doc2vec_" + parater_name

    testDataSet = getTrainDataSet(filePath)    # [smile, protein, label]
    test_protein = load_tensor(test_protein_file, torch.FloatTensor)
    test_dataset = ProDataset(dataSet=testDataSet, proteins=test_protein, bits=finger_len)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True)
    testArgs['test_loader'] = test_loader
    testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target, roce1, roce2, roce3, roce4 = test(
        testArgs)
    result = [testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3, roce4]
    return result


def testPerProtein(testArgs):
    result = {}

    # testDataPath = test_file      # 根目录
    # testArgs['test_proteins'] = ["xiap_2jk7A_full"]
    for x in testArgs['test_proteins']:
        print('\n current test protein:', x.split('_')[0])
        # data = testArgs['testDataDict'][x]
        # [smile,contactMap,label],....]
        x_split = x.split('_')

        p_name = x_split[0] + "_" + x_split[1] + "_" + x_split[2]
        testDataPath = test_file + p_name + "/"
        testPath = testDataPath + p_name + "_filter"
        test_protein_file = testPath + "_protein_seq_proteins_doc2vec_" + parater_name

        testDataSet = getTrainDataSet(testPath)
        test_protein = load_tensor(test_protein_file, torch.FloatTensor)
        test_dataset = ProDataset(dataSet=testDataSet, proteins=test_protein, bits=256)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True)
        testArgs['test_loader'] = test_loader
        testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target, roce1, roce2, roce3, roce4 = test(
            testArgs)
        result[x] = [testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3,
                     roce4]
    return result


def test(testArgs):
    test_loader = testArgs['test_loader']
    criterion = testArgs["criterion"]
    attention_model = testArgs['model']
    losses = []
    accuracy = []
    print('test begin ...', len(test_loader))
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    with torch.no_grad():
        for batch_idx, (lines, finger, contactmap, y) in enumerate(test_loader):
            # input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            # attention_model.hidden_state = attention_model.init_hidden()
            lines = lines.type(torch.FloatTensor)
            lines = create_variable(lines)

            finger = finger.type(torch.FloatTensor)
            finger = create_variable(finger)

            contactmap = create_variable(contactmap)
            y_pred = attention_model(lines, finger, contactmap)
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
    print("test size =", testSize, "  test acc =", testAcc, "  test recall =", testRecall, "  test precision =",
          testPrecision, "  test auc =", testAuc, "  test loss = ", testLoss)
    roce1 = round(getROCE(all_pred, all_target, 0.5), 2)
    roce2 = round(getROCE(all_pred, all_target, 1), 2)
    roce3 = round(getROCE(all_pred, all_target, 2), 2)
    roce4 = round(getROCE(all_pred, all_target, 5), 2)
    print("roce0.5 =", roce1, "  roce1.0 =", roce2, "  roce2.0 =", roce3, "  roce5.0 =", roce4)
    return testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target, roce1, roce2, roce3, roce4


if __name__ == "__main__":
    contactPath = '../../data/DUDE/contactMap'
    contactDictPath = '../../data/DUDE/dataPre/DUDE-contactDict'
    smileLettersPath = '../../data/DUDE/voc/combinedVoc-wholeFour.voc'
    seqLettersPath = '../../data/DUDE/voc/sequence.voc'
    print('get train data....')

    fold_num = 3
    testFold = "DUDE_foldTest" + str(fold_num) + "_30"   # DUDE_foldTest1_30
    testFoldPath = "../../data/DUDE/dataPre/" + testFold
    testProteinList = getTestProteinList(testFoldPath)  # whole foldTest
    # testProteinList = ['kpcb_2i0eA_full']# a protein of fold1Test
    # testProteinList = ['tryb1_2zebA_full','mcr_2oaxE_full', 'cxcr4_3oduA_full']  # protein of fold3Test
    DECOY_PATH = '../../data/DUDE/decoy_smile'
    ACTIVE_PATH = '../../data/DUDE/active_smile'
    print('get protein-seq dict....')
    # dataDict = getDataDict(testProteinList, ACTIVE_PATH, DECOY_PATH, contactPath)

    print('train loader....')
    # trainDataSet:[[smile,seq,label],....]    seqContactDict:{seq:contactMap,....}
    # trainFoldPath = '../../data/trainDataset72/trainDataset72Fold1/trainDataset72Fold1_filter'
    foldName = "DUDE_foldTrain" + str(fold_num)
    root_path = "../../data/trainDataset72/"
    suffix = "_filter_72_2"
    trainFoldPath = root_path + foldName + suffix  # "dataset_small_filter"
    trainDataSet = getTrainDataSet(trainFoldPath)

    smiles_file_path = trainFoldPath + "_smiles"
    print(smiles_file_path)

    matrix_tensor = load_tensor(smiles_file_path, torch.FloatTensor)  # 加载距离矩阵。

    vector_len = 200
    k_gram = 3
    w_dows = 4
    doc2vec_epoch = 15
    # 3_576_4_10"
    parater_name = str(k_gram) + "_" + str(vector_len) + "_" + str(w_dows) + "_" + str(doc2vec_epoch)
    #  ata/DUDE/dataPre
    protein_file_path = trainFoldPath + "_protein_seq_proteins_doc2vec_" + parater_name
    trian_proteinDataset = load_tensor(protein_file_path, torch.FloatTensor)

    # test_proteinDataset =
    finger_len = 50

    train_dataset = ProDataset(dataSet=trainDataSet, matrix=matrix_tensor,proteins=trian_proteinDataset, bits=finger_len)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, drop_last=True)

    modelArgs = {}
    modelArgs['batch_size'] = 1
    modelArgs['protein_input_dim'] = vector_len
    modelArgs['protein_fc'] = vector_len
    modelArgs['finger'] = finger_len
    modelArgs['d_a'] = 32
    # d_a = modelArgs['d_a']
    modelArgs['in_channels'] = 64
    modelArgs['cnn_channel_block1'] = 128
    modelArgs['cnn_channel_block2'] = 128
    cnn_b1 = modelArgs['cnn_channel_block1']
    cnn_b2 = modelArgs['cnn_channel_block2']
    modelArgs['block_num1'] = 4
    modelArgs['block_num2'] = 4

    modelArgs['r'] = 20
    modelArgs['cnn_layers'] = 4
    modelArgs['hidden_nodes'] = 256
    modelArgs['spp_out_dim'] = 100
    modelArgs['fc_final'] = 100       # = modelArgs['cnn_channels']+ modelArgs['protein_fc']

    p_input_dim = modelArgs['protein_input_dim']
    modelArgs['task_type'] = 0
    modelArgs['n_classes'] = 1

    print('train args...')

    trainArgs = {}
    trainArgs['model'] = SPP_CPI(modelArgs, block=ResidualBlock).to(device)
    trainArgs['epochs'] = 30
    trainArgs['lr'] = 0.0001
    trainArgs['weight_decay'] = 1e-6
    learning_rate = trainArgs['lr']
    trainArgs['train_loader'] = train_loader
    trainArgs['doTest'] = False
    trainArgs['test_proteins'] = testProteinList      # testProteinList
    trainArgs['testDataDict'] = ""
    trainArgs['seqContactDict'] = ""        # seqContactDict
    trainArgs['use_regularizer'] = False
    trainArgs['penal_coeff'] = 0.03
    trainArgs['clip'] = True
    trainArgs['criterion'] = torch.nn.BCELoss()
    trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(),
                                              lr=trainArgs['lr'])
    trainArgs['doSave'] = True
    trainArgs['saveNamePre'] = 'DUDE30Res-fold1-'   # 没用
    # d_name = "test_dataset_30_fold1"        #
    trainArgs['d_name'] = ""
    # trainArgs['testDataset'] = "New File"  # d_name + "_filter"
    trainArgs['testDataset'] = trainArgs['d_name'] + "_filter"

    print('train args over...')
    test_file = "../../data/" + testFold + "/"     # 测试数据集

    models_path = "./model_pkl/"
    if fold_num == 1:
        model_name_fold = ""
    elif fold_num == 2:
        model_name_fold = ""
    else:
        # model_name_fold = "2020_0824_2205spp_fc_protein_2_3_protein_200_15_200_in_channel_64_cnn_channel_128_128" \
        #                   "_cnn_layer_4_4_spp_out_dim_100_fc_100_lr_0.0001_DUDE_foldTrain3_13"
        model_name_fold = ""

    if os.path.exists(models_path + model_name_fold + ".pkl"):
        print("模型加载ing")
        trainArgs['model'].load_state_dict(torch.load(models_path + model_name_fold + ".pkl", map_location="cuda"))
        losses, accs, testResults = train(trainArgs)
    else:
        print("模型不存在")
        losses, accs, testResults = train(trainArgs)


    # path = "../../data/doc2vec"

"""

"""