from spp_datapre_ddi import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
import datetime
from spp_finger_model_ddi import *
import os
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
    loss_total = []
    ISOTIMEFORMAT = '%Y_%m%d_%H%M'
    n_time_str = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    info = "_BIO_spp_no_finger_ddi_3" + "_in_channels_" +\
           str(modelArgs['in_channels']) + "_cnn_channels_" + \
           str(cnn_channels) + "_cnn_layers_" +\
           str(modelArgs['cnn_layers']) + \
           "_spp_out_dim_" + str(modelArgs['spp_out_dim']) + "_fc_final_" + str(modelArgs['fc_final']) + "_lr_" + str(learning_rate)

    test_result = "../data/BIOSNAP/result/" + n_time_str + info + "_Test_" + foldName + ".txt"
    train_loss = "../data/BIOSNAP/log/" + n_time_str + info + "_Train_loss" + foldName + ".log"
    with open(test_result, "a+") as t_f:
        t_f.write("testAcc, testAuc, testPrecision, testPAuc, testPRAUC, testRecall, testF1, testLoss\n")
    for i in range(trainArgs['epochs']):
        print("Running EPOCH", i + 1)
        total_loss = 0
        n_batches = 0
        correct = 0
        train_loader = trainArgs['train_loader']
        optimizer = trainArgs['optimizer']
        criterion = trainArgs["criterion"]
        attention_model = trainArgs['model']
        if os.path.exists(model_path):
            attention_model.load_state_dict(torch.load(model_path))

        for batch_idx, (lines, contactmap, y) in enumerate(train_loader):
            # input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            # attention_model.hidden_state = attention_model.init_hidden()
            lines = lines.type(torch.FloatTensor)
            lines = lines.to(device)
            # print(lines.shape, "lines.shape")
            contactmap = contactmap.type(torch.FloatTensor)
            contactmap = contactmap.to(device)
            # print(contactmap.shape, "contactmap.shape")

            y_pred = attention_model(lines, contactmap)
            # penalization AAT - I
            """
            if trainArgs['use_regularizer']:
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size,
                                                                 att.size(1), att.size(1))).cuda()

                penal = attention_model.l2_matrix_norm(att @ attT - identity)
            """

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
                print("[epoch %d]" % (i+1), batch_idx, loss.item())

        avg_loss = total_loss / n_batches
        acc = correct.numpy() / (len(train_loader.dataset))

        losses.append(avg_loss)
        accs.append(acc)

        print("avg_loss is", avg_loss)
        print("train ACC = ", acc)

        # if (trainArgs['doSave']):
        model_name = n_time_str + info + "_" + foldName + '_%d.pkl' % (i + 1)
        print("存储模型", model_name)
        torch.save(attention_model.state_dict(), '../data/model_pkl/' + model_name)
        # torch.save(attention_model, "../data/model/" + model_name)

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
            testResult = testPerProteinDataset72(testArgs, testArgs['d_name'], testArgs['testDataset'])

            with open(test_result, "a+") as t_f:
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
    # file = "../data/DUDE-foldTest1/"
    # testDataPath = "./DUDE-foldTest1_top4/"
    testDataPath = "../data/BIOSNAP/"
    filePath = testDataPath + "sup_test_pair_filter_2"

    drugA_test_file_path = filePath + "_drugA"
    drugB_test_file_path = filePath + "_drugB"
    # test_protein_file = testDataPath + fileName + "_protein_seq_proteins_doc2vec_3_512_4_7"
    # test_protein_file = testDataPath + fileName + "_protein_seq_proteins_doc2vec_3_512_4_7"
    d_a = load_tensor(drugA_test_file_path, torch.FloatTensor)
    d_b = load_tensor(drugB_test_file_path, torch.FloatTensor)

    testDataSet = getTrainDataSet(filePath)    # [smile, protein, label]
    # test_protein = load_tensor(test_protein_file, torch.FloatTensor)
    test_dataset = ProDataset(dataSet=testDataSet, druga=d_a, drugb=d_b)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True)
    testArgs['test_loader'] = test_loader
    testAcc, testAuc, testPrecision, testPAuc,testPRAUC, testRecall, testF1, testLoss = test(testArgs)
    result = [testAcc, testAuc, testPrecision, testPAuc,testPRAUC, testRecall, testF1, testLoss]
    return result


def test(testArgs):
    test_loader = testArgs['test_loader']
    criterion = testArgs["criterion"]
    attention_model = testArgs['model']
    print('test begin ...')
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    with torch.no_grad():
        for batch_idx, (drug1, drug2, y) in enumerate(test_loader):
            # input, seq_lengths, y = make_variables(lines, properties, smiles_letters)
            # attention_model.hidden_state = attention_model.init_hidden()
            drug1 = drug1.type(torch.FloatTensor)  # 类型
            drug1 = drug1.to(device)

            drug2 = drug2.type(torch.FloatTensor)  # 类型
            drug2 = drug2.to(device)
            y_pred = attention_model(drug1, drug2)
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
    precision, recall, _ = precision_recall_curve(all_target, all_pred)
    testPAuc = round(metrics.average_precision_score(all_target, all_pred), 3)
    PRAUC = round(auc(recall, precision), 3)
    testF1 = round(metrics.f1_score(all_target, np.round(all_pred)), 3)

    # print("AUPR = ", metrics.average_precision_score(all_target, all_pred))
    testLoss = round(total_loss.item() / n_batches, 5)

    print("  test Roc_auc =", testAuc,
          "  test precision =", testPrecision,
          "  test Pre_AUC =", testPAuc, " PRAUC ", PRAUC,
          "  test F1 =", testF1,
          "  test loss = ", testLoss)
    # roce1 = round(getROCE(all_pred, all_target, 0.5), 2)
    # roce2 = round(getROCE(all_pred, all_target, 1), 2)
    # roce3 = round(getROCE(all_pred, all_target, 2), 2)
    # roce4 = round(getROCE(all_pred, all_target, 5), 2)
    # print("roce0.5 =", roce1, "  roce1.0 =", roce2, "  roce2.0 =", roce3, "  roce5.0 =", roce4)
    return testAcc, testAuc, testPrecision, testPAuc, PRAUC, testRecall, testF1, testLoss


if __name__ == "__main__":
    # /home/student/Project/drugVQA_copy/DUDE-foldTest1_top4
    # testFoldPath = '../data/DUDE/dataPre/DUDE-foldTest1'
    # testFoldPath = '../data/DUDE-foldTest1/'

    print('get train datas....')
    print('get protein-seq dict....')

    print('train loader....')
    # trainDataSet:[[smile,seq,label],....]    seqContactDict:{seq:contactMap,....}
    # trainFoldPath = '../data/trainDataset72/trainDataset72Fold1/trainDataset72Fold1_filter'
    foldName = "sup_train_val_pair_filter_2"
    trainFoldPath = "../data/BIOSNAP/" + foldName
    trainDataSet = getTrainDataSet(trainFoldPath)

    drugA_file_path = trainFoldPath + "_drugA"
    drugB_file_path = trainFoldPath + "_drugB"

    drugA = load_tensor(drugA_file_path, torch.FloatTensor)
    drugB = load_tensor(drugB_file_path, torch.FloatTensor)


    train_dataset = ProDataset(dataSet=trainDataSet, druga=drugA, drugb=drugB)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, drop_last=True)

    # train_size = int(0.01 * len(train_dataset))
    # other_size = len(train_dataset) - train_size

    # train_dataset, other_dataset = torch.utils.data.random_split(train_dataset, [train_size, other_size])
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    print('model args...')

    modelArgs = {}
    modelArgs['batch_size'] = 1
    # modelArgs['protein_input_dim'] = 512
    # modelArgs['protein_fc'] = 32
    # modelArgs['d_a'] = 32
    # d_a = modelArgs['d_a']
    modelArgs['in_channels'] = 32
    modelArgs['cnn_channels'] = 128
    cnn_channels = modelArgs['cnn_channels']
    # modelArgs['r'] = 20
    modelArgs['cnn_layers'] = 4
    modelArgs['spp_out_dim'] = 64
    modelArgs['fc_final'] = 64       # = modelArgs['cnn_channels']+ modelArgs['protein_fc']

    # p_input_dim = modelArgs['protein_input_dim']
    modelArgs['task_type'] = 0
    modelArgs['n_classes'] = 1

    print('train args...')

    trainArgs = {}
    trainArgs['model'] = DrugVQA(modelArgs, block=ResidualBlock).to(device)
    trainArgs['epochs'] = 15
    trainArgs['lr'] = 0.0001
    learning_rate = trainArgs['lr']
    trainArgs['weight_decay'] = 1e-6
    trainArgs['train_loader'] = train_loader
    trainArgs['doTest'] = True
    trainArgs['test_proteins'] = ""       # testProteinList
    trainArgs['testDataDict'] = ""
    trainArgs['seqContactDict'] = ""        # seqContactDict
    trainArgs['use_regularizer'] = False
    trainArgs['penal_coeff'] = 0.03
    trainArgs['clip'] = True
    trainArgs['criterion'] = torch.nn.BCELoss()
    trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(),
                                              lr=trainArgs['lr'], weight_decay=trainArgs['weight_decay'])
    trainArgs['doSave'] = True
    # d_name = "test_dataset_30_fold1"        #
    trainArgs['d_name'] = "testDataset30Fold3"
    # trainArgs['testDataset'] = "New File"  # d_name + "_filter"
    trainArgs['testDataset'] = trainArgs['d_name'] + "_filter"

    print('train args over...')
    model_path = "../data//BIOSNAP/model_pkl/2020_0808_1304_spp_no_finger_ddi_in_channels_32_cnn_channels_128_cnn_layers_4" \
                 "_spp_out_dim_64_fc_final_96_lr_0.0001_sup_train_val_pair_filter_new_4.pkl"

    losses, accs, testResults = train(trainArgs)

"""
"""