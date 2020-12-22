"""

"""
from spp_datapre import *
from sklearn import metrics
import datetime
from spp_finger_model import *


# torch.cuda.set_device(1)
# torch.cuda.set_device(1)

def train(trainArgs, models_path, model_name):
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
    n_time_str = datetime.datetime.now().strftime(ISOTIMEFORMAT)

    train_log = "./result_30/" + n_time_str + "_" + model_name + "_Test_.txt"
    # train_loss = "./log/" + n_time_str + info + "_Train_loss" + foldName + ".log"
    with open(train_log, "a+") as t_f:  # 写入测试集数据
        t_f.write("testAcc, testRecall, testPrecision, testAuc, "
                  "testLoss, all_pred, all_target, roce1, roce2, roce3, roce4\n")
    for i in range(trainArgs['epochs']):
        print("Running EPOCH", i + 1)
        attention_model = trainArgs['model']
        attention_model.load_state_dict(torch.load(models_path + model_name + ".pkl", map_location="cuda"))
        # load_state_dict(torch.load(model_path, map_location="cuda"))    # 加载已有模型。
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
            attention_model.eval()

            # d_name = "test_dataset_30_fold1"        #
            # testDataset = "New File"    # d_name + "_filter"
            testResult = testPerProtein(testArgs, train_log)
            # testResult = testPerProtein(testArgs)
            #         result[x] = [testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target, roce1, roce2, roce3,
            #                      roce4]
            # testResults[i] = testResult

    return losses, accs, testResults


def getROCE(predList, targetList, roceRate):
    """
    getROCE(all_pred, all_target, 0.5)
    :param predList:
    :param targetList:
    :param roceRate:
    :return:
    """
    p = sum(targetList)  # 正样本数
    n = len(targetList) - p  # 负样本数
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

    testDataSet = getTrainDataSet(filePath)  # [smile, protein, label]
    test_protein = load_tensor(test_protein_file, torch.FloatTensor)
    test_dataset = ProDataset(dataSet=testDataSet, proteins=test_protein, bits=finger_len)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True)
    testArgs['test_loader'] = test_loader
    testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target, roce1, roce2, roce3, roce4 = test(
        testArgs)
    result = [testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3, roce4]
    return result


def testPerProtein(testArgs, train_log):
    result = {}

    # testDataPath = test_file      # 根目录
    # testArgs['test_proteins'] = ["xiap_2jk7A_full"]
    num = 1
    for x in testArgs['test_proteins']:
        print('\n current test protein:', x.split('_')[0], num)
        num += 1
        # data = testArgs['testDataDict'][x]
        # [smile,contactMap,label],....]
        x_split = x.split('_')

        p_name = x_split[0] + "_" + x_split[1] + "_" + x_split[2]
        testDataPath = test_file + p_name + "/"
        testPath = testDataPath + p_name + suffix
        test_protein_file = testPath + "_protein_seq_proteins_doc2vec_" + parater_name

        smiles_test_file_path = testPath + "_smiles"
        print(smiles_test_file_path)

        matrix_test_tensor = load_tensor(smiles_test_file_path, torch.FloatTensor)  # 加载距离矩阵。

        testDataSet = getTrainDataSet(testPath)
        test_protein = load_tensor(test_protein_file, torch.FloatTensor)
        test_dataset = ProDataset(dataSet=testDataSet, matrix=matrix_test_tensor,
                                  proteins=test_protein, bits=finger_len)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True)
        testArgs['test_loader'] = test_loader
        testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target, roce1, roce2, roce3, roce4 = test(
            testArgs)
        result[x] = [testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3,
                     roce4]
        testResult = [testAcc, testRecall, testPrecision, testAuc, roce1, roce2, roce3, roce4, testLoss]
        with open(train_log, "a+") as t_f:
            t_f.write('\t'.join(map(str, testResult)) + '\n')
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
            lines = lines.type(torch.FloatTensor)  # 类型
            lines = create_variable(lines)

            finger = finger.type(torch.FloatTensor)  # 类型
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
    # /home/student/Project/drugVQA_copy/DUDE-foldTest1_top4
    # testFoldPath = '../../data/DUDE/dataPre/DUDE-foldTest1'
    # testFoldPath = '../../data/DUDE-foldTest1/'

    contactPath = '../../data/DUDE/contactMap'
    contactDictPath = '../../data/DUDE/dataPre/DUDE-contactDict'
    smileLettersPath = '../../data/DUDE/voc/combinedVoc-wholeFour.voc'
    seqLettersPath = '../../data/DUDE/voc/sequence.voc'
    print('get train datas....')

    print('get seq-contact dict....')
    # seqContactDict = getSeqContactDict(contactPath, contactDictPath)
    print('get letters....')
    smiles_letters = getLetters(smileLettersPath)
    sequence_letters = getLetters(seqLettersPath)  # 无用
    fold_number = 3
    testFold = "DUDE_foldTest" + str(fold_number) + "_30"
    testFold_file = "DUDE_foldTest" + str(fold_number)
    testFoldPath = "../../data/DUDE/dataPre/" + testFold
    testProteinList = getTestProteinList(testFoldPath)  # whole foldTest
    # testProteinList = ['kpcb_2i0eA_full']# a protein of fold1Test
    # testProteinList = ['tryb1_2zebA_full','mcr_2oaxE_full', 'cxcr4_3oduA_full']  # protein of fold3Test
    DECOY_PATH = '../../data/DUDE/decoy_smile'
    ACTIVE_PATH = '../../data/DUDE/active_smile'
    print('get protein-seq dict....')
    # dataDict = getDataDict(testProteinList, ACTIVE_PATH, DECOY_PATH, contactPath)

    N_CHARS_SMI = len(smiles_letters)  # 无用
    N_CHARS_SEQ = len(sequence_letters)  # 无用

    print('train loader....')
    # trainDataSet:[[smile,seq,label],....]    seqContactDict:{seq:contactMap,....}
    # trainFoldPath = '../../data/trainDataset72/trainDataset72Fold1/trainDataset72Fold1_filter'
    foldName = "DUDE_foldTrain" + str(fold_number)

    root_path = "../../data/DUDE/dataPre/"
    trainFoldPath = root_path + "dataset_small_filter"  # foldName + "_filter"
    # trainDataSet = getTrainDataSet(trainFoldPath)

    vector_len = 200
    k_gram = 3
    w_dows = 4
    doc2vec_epoch = 15
    # 3_576_4_10"
    suffix = "_filter_2"
    parater_name = str(k_gram) + "_" + str(vector_len) + "_" + str(w_dows) + "_" + str(doc2vec_epoch)
    #  ata/DUDE/dataPre
    protein_file_path = root_path + foldName + "_protein_seq_proteins_doc2vec_" + parater_name
    # trian_proteinDataset = load_tensor(protein_file_path, torch.FloatTensor)

    # test_proteinDataset =
    finger_len = 50

    # train_dataset = ProDataset(dataSet=trainDataSet, proteins=trian_proteinDataset, bits=finger_len)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, drop_last=True)

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
    trainArgs['model'] = DrugVQA(modelArgs, block=ResidualBlock).to(device)
    trainArgs['epochs'] = 1
    trainArgs['lr'] = 0.0001
    trainArgs['weight_decay'] = 1e-6
    learning_rate = trainArgs['lr']
    trainArgs['train_loader'] = ""
    trainArgs['doTest'] = True
    trainArgs['test_proteins'] = testProteinList  # testProteinList
    trainArgs['testDataDict'] = ""
    trainArgs['seqContactDict'] = ""  # seqContactDict
    trainArgs['use_regularizer'] = False
    trainArgs['penal_coeff'] = 0.03
    trainArgs['clip'] = True
    trainArgs['criterion'] = torch.nn.BCELoss()
    trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(), lr=trainArgs['lr'],
                                              weight_decay=trainArgs['weight_decay'])
    trainArgs['doSave'] = True
    trainArgs['saveNamePre'] = 'DUDE30Res-fold1-'  # 没用
    # d_name = "test_dataset_30_fold1"        #
    trainArgs['d_name'] = ""
    # trainArgs['testDataset'] = "New File"  # d_name + "_filter"
    trainArgs['testDataset'] = trainArgs['d_name'] + "_filter"

    print('train args over...')
    test_file = "../../data/" + testFold_file + "/"  # 测试数据集
    model_path = "./model_pkl_30/"

    if fold_number == 1:
        model_name_fold = "2020_0826_1151spp_fc_protein_2_1_protein_200_15_200_in_channel_64_cnn_channel_128_128" \
                          "_cnn_layer_4_4_spp_out_dim_100_fc_100_lr_0.0001_DUDE_foldTrain1_3"
    elif fold_number == 2:
        model_name_fold = "2020_0826_1152spp_fc_protein_2_2_protein_200_15_200_in_channel_64_cnn_channel_128_128" \
                          "_cnn_layer_4_4_spp_out_dim_100_fc_100_lr_0.0001_DUDE_foldTrain2_3"
    else:
        # model_name_fold = "2020_0824_2205spp_fc_protein_2_3_protein_200_15_200_in_channel_64_cnn_channel_128_128" \
        #                   "_cnn_layer_4_4_spp_out_dim_100_fc_100_lr_0.0001_DUDE_foldTrain3_13"
        model_name_fold = "2020_0826_1456spp_fc_protein_2_3_protein_200_15_200_in_channel_64_cnn_channel_128_128" \
                          "_cnn_layer_4_4_spp_out_dim_100_fc_100_lr_0.0001_DUDE_foldTrain3_3"

    losses, accs, testResults = train(trainArgs, model_path, model_name_fold)

    # path = "../../data/doc2vec"

"""

"""
